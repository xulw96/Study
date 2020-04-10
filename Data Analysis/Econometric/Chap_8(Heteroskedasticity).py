import statsmodels.api as sm
import numpy as np
import wooldridge
import pandas as pd


# c4
df = wooldridge.dataWoo('VOTE1')
df.dropna(inplace=True)
X = df[['prtystrA', 'democA', 'expendA', 'expendB']]
X['expendA'] = np.log(X['expendA'])
X['expendB'] = np.log(X['expendB'])
X = sm.add_constant(X)
y = df['voteA']
model = sm.OLS(y, X, missing='drop').fit()
sm.stats.diagnostic.het_white(model.resid, X)

# c13
df = wooldridge.dataWoo('FERTIL2')
df.dropna(inplace=True)
model = sm.formula.rlm('children ~ age + age^2 + educ + electric + urban', data=df).fit()
