from __future__ import annotations

import warnings
from typing import List
from typing import Optional

# import catboost
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from rich.console import Console
from rich.progress import track
from rich.table import Table
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

console = Console()

# https://gist.github.com/neelabalan/33ab34cf65b43e305c3f12ec6db05938
def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table = Table(show_header=True, header_style="bold magenta"),
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    console.print(rich_table)
    # return rich_table


class Model:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer_low = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    categorical_transformer_high = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            # 'OrdianlEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
            ("encoding", OrdinalEncoder()),
        ]
    )

    def fit(self, X, y):
        ...

    def scores(self, X, y):
        ...

    def get_card_split(self, df: pd.DataFrame, cols: List[str], n: int = 11):
        """
        Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)

        Parameters
            df: DataFrame from which the cardinality of the columns is calculated.
            cols: Categorical columns to list
            n: The value of 'n' will be used to split columns.
        Returns
            card_low: Columns with cardinality < n
            card_high: Columns with cardinality >= n
        """
        cond = df[cols].nunique() > n
        card_high = cols[cond]
        card_low = cols[~cond]
        return card_low, card_high


class Classifiers(Model):
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn

    Parameters
        random_state: int
            Set the random state for reproducability
    """

    excluded_classifiers = [
        "ClassifierChain",
        "ComplementNB",
        "GradientBoostingClassifier",
        "GaussianProcessClassifier",
        "HistGradientBoostingClassifier",
        "MLPClassifier",
        "LogisticRegressionCV",
        "MultiOutputClassifier",
        "MultinomialNB",
        "OneVsOneClassifier",
        "OneVsRestClassifier",
        "OutputCodeClassifier",
        "RadiusNeighborsClassifier",
        "VotingClassifier",
        "StackingClassifier",
    ]

    def __init__(
        self,
        random_state: int = 42,
    ):
        self.models = {}
        self.random_state = random_state
        self.classifiers = [
            est
            for est in all_estimators()
            if (issubclass(est[1], ClassifierMixin) and (est[0] not in self.excluded_classifiers))
        ] + [
            ("XGBClassifier", xgboost.XGBClassifier),
            ("LGBMClassifier", lightgbm.LGBMClassifier),
            # ('CatBoostClassifier',catboost.CatBoostClassifier)
        ]

    def scores(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Get the scores from all models
        Parameters
            X_test : Testing vectors, where rows is the number of samples
                and columns is the number of features.
            y_test : Testing vectors, where rows is the number of samples
                and columns is the number of features.
        """
        score_list = []
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)
        if not self.models:
            raise Exception("Models not fitted")
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            score_list.append(
                {
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, y_pred, normalize=True),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred, average="weighted"),
                }
            )
            try:
                score_list[-1].update({"roc_auc_score": roc_auc_score(y_test, y_pred)})
            except Exception as ex:
                score_list[-1].update({"roc_auc_score": None})
                print(f"ROC AUC couldn't be calculated for {model_name} due to {ex}")

        scores_df = pd.DataFrame(score_list)
        scores_df.sort_values(by="balanced_accuracy", ascending=False, inplace=True)
        return scores_df

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Classifiers:
        """Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
            X_train: Training vectors, where rows is the number of samples
                and columns is the number of features.
            y_train:,Training vectors, where rows is the number of samples
                and columns is the number of features.
        Returns
            Self
        """
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = self.get_card_split(X_train, categorical_features)

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", self.numeric_transformer, numeric_features),
                ("categorical_low", self.categorical_transformer_low, categorical_low),
                (
                    "categorical_high",
                    self.categorical_transformer_high,
                    categorical_high,
                ),
            ]
        )
        for name, model in track(self.classifiers, description="Processing..."):
            args = {}
            if "random_state" in model().get_params().keys():
                args = {"random_state": self.random_state}
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model(**args))])
            try:
                # print(model_name)
                pipe.fit(X_train, y_train)
                self.models[name] = pipe
            except Exception as ex:
                print(f"Error in fitting model {name} due to - {ex}")
        return self


class Regressors(Model):
    """
    This module helps in fitting regression models that are available in Scikit-learn

    Parameters
        random_state: int
            Set the random state for reproducability
    """

    excluded_regressors = [
        "TheilSenRegressor",
        "ARDRegression",
        "CCA",
        "IsotonicRegression",
        "StackingRegressor",
        "MultiOutputRegressor",
        "MultiTaskElasticNet",
        "MultiTaskElasticNetCV",
        "MultiTaskLasso",
        "MultiTaskLassoCV",
        "PLSCanonical",
        "PLSRegression",
        "RadiusNeighborsRegressor",
        "RegressorChain",
        "VotingRegressor",
    ]

    def __init__(
        self,
        random_state=42,
    ):
        self.models = {}
        self.random_state = random_state
        self.regressors = [
            est
            for est in all_estimators()
            if (issubclass(est[1], RegressorMixin) and (est[0] not in self.excluded_regressors))
        ] + [
            ("XGBRegressor", xgboost.XGBRegressor),
            ("LGBMRegressor", lightgbm.LGBMRegressor),
            # ('CatBoostRegressor',catboost.CatBoostRegressor)
        ]

    def __adjusted_rsquared(self, r2: float, n: float, p: float):
        return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

    def scores(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Get the scores from all models
        """
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test)

        score_list = []
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            r_squared = r2_score(y_test, y_pred)
            score_list.append(
                {
                    "model": model_name,
                    "r_squared": r_squared,
                    "adjusted_r_squared": self.__adjusted_rsquared(r_squared, X_test.shape[0], X_test.shape[1]),
                    "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                }
            )
        scores_df = pd.DataFrame(score_list)
        scores_df.sort_values(by="adjusted_r_squared", ascending=False)

        return scores_df

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> Regressors:
        """Fit Regression algorithms to X_train and y_train, predict and score on X_test, y_test.

        Parameters
            X_train: Training vectors, where rows is the number of samples
                and columns is the number of features.
            y_train: Training vectors, where rows is the number of samples
                and columns is the number of features.
        Returns
            Self
        """

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = self.get_card_split(X_train, categorical_features)

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", self.numeric_transformer, numeric_features),
                ("categorical_low", self.categorical_transformer_low, categorical_low),
                (
                    "categorical_high",
                    self.categorical_transformer_high,
                    categorical_high,
                ),
            ]
        )

        for name, model in track(self.regressors, description="Processing..."):
            args = {}
            if "random_state" in model().get_params().keys():
                args = {"random_state": self.random_state}
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", model(**args))])
            pipe.fit(X_train, y_train)
            self.models[name] = pipe

        return self
