import wooldridge
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

pd.options.display.max_columns = None


def lin_reg(X, y, df):
    reg = LinearRegression().fit(X, y)
    print('coef is {}, intercept is {}, R^2 is {}, n={}'.format(reg.coef_, reg.intercept_, reg.score(X, y),
                                                                len(df.index)))
    return reg


# c1
df = wooldridge.dataWoo('BWGHT')
X = np.array(df['cigs']).reshape(-1, 1)
y = np.array(df['bwght']).reshape(-1, 1)
reg = lin_reg(X, y, df)
X = np.array(list(zip(df['cigs'], df['faminc'])))
reg = lin_reg(X, y, df)
# c2
df = wooldridge.dataWoo('HPRICE1')
X = np.array(list(zip(df['sqrft'], df['bdrms'])))
y = np.array(df['price'])
reg = lin_reg(X, y, df)
reg.predict([[2438, 4]])
# c3
df = wooldridge.dataWoo('CEOSAL2')
sales = np.log(df['sales'])
mktval = np.log(df['mktval'])
X = np.array(list(zip(sales, mktval)))
y = np.array(np.log(df['salary']))
reg = lin_reg(X, y, df)
profits = df['profits']
X = np.array(list(zip(sales, mktval, profits)))
reg = lin_reg(X, y, df)
ceoten = df['ceoten']
X = np.array(list(zip(sales, mktval, profits, ceoten)))
reg = lin_reg(X, y, df)
np.corrcoef(np.log(df['mktval']), df['profits'])
# c5
df = wooldridge.dataWoo('WAGE1')
X = np.column_stack(df['tenure'], df['exper'])
y = np.array(df['educ'])
reg = lin_reg(X, y, df)
r1 = reg.predict(X) - y
y = np.array(np.log(df['wage']))
X = r1.reshape(-1, 1)
reg = lin_reg(X, y, df)
X = np.array(list(zip(df['educ'], df['exper'], df['tenure'])))
reg = lin_reg(X, y, df)
#c10
# ability^2 = np.multiply(ability, ability)
