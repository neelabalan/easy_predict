from easy_predict import Classifiers
from easy_predict import Regressors
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
from easy_predict import df_to_table


def test_classifier():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = Classifiers()
    # Fit all classifier models
    models = clf.fit(X_train, y_train)
    # Rich Table
    df_to_table(models.scores(X_test, y_test))


def test_regressor():
    boston = datasets.load_boston()
    X, y = shuffle(boston.data, boston.target)
    X = X.astype(np.float32)

    offset = int(X.shape[0] * 0.9)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]

    reg = Regressors()
    reg.fit(X_train, y_train)
    df_to_table(reg.scores(X_test, y_test))


# test_regressor()
# test_classifier()
