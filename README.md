# Easy Predict

> Based on [LazyPredict](https://github.com/shankarpandala/lazypredict) 


## Usage

```python 
from easy_predict import Regressors
from easy_predict import df_to_table

from sklearn.model_selection import train_test_split
from sklearn import datasets

boston = datasets.load_boston()
X, y = boston.data, boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = Regressors()
reg.fit(X_train, y_train)
df_to_table(reg.scores(X_test, y_test))
```

```
Processing... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
┏━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃    ┃ model                         ┃ r_squared              ┃ adjusted_r_squared   ┃ rmse               ┃
┡━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ 0  │ AdaBoostRegressor             │ 0.8649163406391276     │ 0.844960800051726    │ 3.482098831897125  │
│ 1  │ BaggingRegressor              │ 0.838297459638218      │ 0.8144095843575002   │ 3.8097642868776838 │
│ 2  │ BayesianRidge                 │ 0.6597287144165369     │ 0.6094613654098889   │ 5.52652750663049   │
│ 3  │ DecisionTreeRegressor         │ 0.6451139869519652     │ 0.5926876441153237   │ 5.643962590723844  │
│ 4  │ DummyRegressor                │ -0.0006162634958384317 │ -0.14843457514863267 │ 9.47705636660635   │
│ 5  │ ElasticNet                    │ 0.5588365719166664     │ 0.4936647018589012   │ 6.292734902446766  │
│ 6  │ ElasticNetCV                  │ 0.6607461551762512     │ 0.6106291099181975   │ 5.518258921930619  │
│ 7  │ ExtraTreeRegressor            │ 0.6276467891067996     │ 0.572640064770304    │ 5.781189917546827  │
│ 8  │ ExtraTreesRegressor           │ 0.8481784895131056     │ 0.8257503118275417   │ 3.6915294627714657 │
.
.
```

> Also check `test_easy_predict.py`