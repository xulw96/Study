import statsmodels.api as sm
import wooldridge
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

# c1
df = wooldridge.dataWoo('WAGE1')
model = sm.formula.ols(formula='wage ~ educ + exper + tenure', data=df).fit()
X = df[['educ', 'exper', 'tenure']]
res1 = df['wage'] - model.predict(X)

model = sm.formula.ols(formula='np.log(wage) ~ educ + exper + tenure', data=df).fit()
res2 = np.log(df['wage']) - model.predict(X)

fig, axes = plt.subplots(2, 1)
sb.distplot(res1, kde=False, ax=axes[0])
sb.distplot(res2, kde=False, ax=axes[1])

# c3
df = wooldridge.dataWoo('BWGHT').dropna()
model = sm.formula.ols(formula='bwght ~ cigs + parity + faminc', data=df).fit()
res = model.resid
X = df[['cigs', 'parity', 'faminc', 'motheduc', 'fatheduc']]
model = sm.OLS(res, X).fit()
R2 = model.rsquared
LM = len(df) * R2
