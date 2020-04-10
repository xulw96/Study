import wooldridge
import pandas as pd
import statsmodels.api as sm
import numpy as np

# c1
df = wooldridge.dataWoo('INTDEF')


def set_dumm(df):
    if df['year'] < 1979:
        df['dumm'] = 0
    else:
        df['dumm'] = 1
    return df


df = df.apply(set_dumm, axis=1)
model = sm.formula.ols('i3 ~ Q("inf") + Q("def") + dumm', data=df).fit()

# c7
df = wooldridge.dataWoo('INTDEF')
inf_lag = sm.tsa.add_lag(df['inf'])
def_lag = sm.tsa.add_lag(df['def'])
X = np.concatenate((inf_lag, def_lag), axis=1)
X = sm.add_constant(X)
y = df['i3'][1:]
model = sm.OLS(y, X).fit()
model.f_test('x2=0, x4=0')

# c11
df = wooldridge.dataWoo('TRAFFIC2')
df = sm.tsa.add_trend(df, 't')
