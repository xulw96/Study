import wooldridge
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

pd.options.display.max_columns = None
pd.options.display.max_rows = None

# c1
df = wooldridge.dataWoo('401K')
df.describe()
X = np.array(df['prate']).reshape(-1, 1)
y = np.array(df['mrate']).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, r^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df)))
reg.predict(np.array([[3.5]]))

# c2
df = wooldridge.dataWoo('CEOSAL2')
df.describe()
len(df[df['ceoten'] == 0].index)
df['ceoten'].max()
X = np.array(df['ceoten']).reshape(-1, 1)
y = np.array(np.log(df['salary'])).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df)))

# c3
df = wooldridge.dataWoo('SLEEP75')
X = np.array(df['totwrk']).reshape(-1, 1)
y = np.array(df['sleep']).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df)))

# c4
df = wooldridge.dataWoo('WAGE2')
df.describe()
X = np.array(np.log(df['wage'])).reshape(-1, 1)
y = np.array(df['IQ']).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df)))
X = np.array(np.log(df['wage'])).reshape(-1, 1)
y = np.array(np.log(df['IQ'])).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df)))

# c5
df = wooldridge.dataWoo('RDCHEM')
X = np.array(np.log(df['rd'])).reshape(-1, 1)
y = np.array(np.log(df['sales'])).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df)))

# c6
df = wooldridge.dataWoo('MEAP93')
X = np.array(np.log(df['expend'])).reshape(-1, 1)
y = np.array(df['math10']).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df)))

# c7
df = wooldridge.dataWoo('CHARITY')
df.describe()
len(df[df['gift'] == 0]) / len(df)
X = np.array(df['mailsyear']).reshape(-1, 1)
y = np.array(df['gift']).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df.index)))

# c8
X = np.random.uniform(0, 10, 500)
U = np.random.normal(0, 36, 500)
y = 1 + 2 * X + U
X = np.array(X).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df.index)))
y_predict = reg.predict(X)
residual = y - y_predict
residual_mul_x = residual * X
error_mul_x = U * X

# c9
df = wooldridge.dataWoo('COUNTYMURDERS')
len(df[df['murders'] == 0])
X = np.array(df['execs']).reshape(-1, 1)
y = np.array(df['murders']).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df.index)))

# c10
df = wooldridge.dataWoo('CATHOLIC')
df.describe()
X = np.array(df['read12']).reshape(-1, 1)
y = np.array(df['math12']).reshape(-1, 1)
reg = LinearRegression().fit(X, y)
print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y), len(df.index)))

