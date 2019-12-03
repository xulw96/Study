import numpy as np
import pandas as pd

# interface
data = pd.DataFrame({'x0': [1, 2, 3, 4, 5], 'x1': [0.01, -0.01, 0.25, -4.1, 0],
                     'y': [-1.5, 0, 3.6, 1.3, -2]})
data.values  # convert back to Numpy array
data['category'] = pd.Categorical(['a', 'b', 'a', 'a', 'b'], categories=['a', 'b'])
dummies = pd.get_dummies(data.category, prefix='category')
data_with_dummies = data.drop('category', axis=1).join(dummies)
print(data_with_dummies)

# Patsy
import patsy
y, X = patsy.dmatrices('y ~ x0 + x1', data)
coef, resid, _, _ = np.linalg.lstsq(X, y, rcond=None)
coef = pd.Series(coef.squeeze(), index=X.design_info.column_names)  # passing name to the coef

y, X = patsy.dmatrices('y ~ standardize(x0) + center(x1)', data)  # do transformation
new_X = patsy.build_design_matrices([X.design_info], data)  # apply same transformation to new data
y, X = patsy.dmatrices('y ~ I(x0 + x1)', data)  # wrap into I to add columns

# statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
def dnorm(mean, variance, size=1):  # generate data
    if isinstance(size, int):
        size = size,  # transform into an iterable
    return mean + np.sqrt(variance) * np.random.randn(*size)
np.random.seed(618)
N = 100
X = np.c_[dnorm(0, 0.4, size=N), dnorm(0, 0.6, size=N), dnorm(0, 0.2, size=N)]
eps = dnorm(0, 0.1, size=N)
beta = [0.1, 0.3, 0.5]
y = np.dot(X, beta) + eps

X_model = sm.add_constant(X)  # add an intercept column
model = sm.OLS(y, X)
results = model.fit()
print(results.params, results.summary())

data = pd.DataFrame(X, columns=['col0', 'col1', 'col2'])  # give variable names
results = smf.ols('y ~ col0 + col1 + col2', data=data).fit()
results.predict(data[:5])

init_x = 4
values = [init_x, init_x]
N = 100
b0 = 0.8
b1 = -0.4
noise = dnorm(0, 0.1, N)
for i in range(N):
    new_X = values[-1] * b0 + values[-2] * b1 + noise[i]
    values.append(new_X)
MAXLAGS = 5
model = sm.tsa.AR(values)
results = model.fit(MAXLAGS)

# Scikit_learn
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
train['Age'] = train['Age'].fillna(train['Age'].median())  # fill NaN
test['Age'] = test['Age'].fillna(train['Age'].median())
train['IsFemale'] = (train['Sex'] == 'female').astype(int)  # encoding sex info
test['IsFemale'] = (test['Sex'] == 'female').astype(int)
predictors = ['Pclass', 'IsFemale', 'Age']  # select predictors to build model
X_train = train[predictors].values
X_test = test[predictors].values
y_train = train['Survived'].values
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
from sklearn.linear_model import LogisticRegressionCV
model = LogisticRegressionCV(10)
model.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
model = LogisticRegressionCV(10)
score = cross_val_score(model, X_train, y_train, cv=4)
