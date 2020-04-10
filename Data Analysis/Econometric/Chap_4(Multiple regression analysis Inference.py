import wooldridge
import statsmodels.api as sm
import numpy as np
import pandas as pd

# c1
df = wooldridge.dataWoo('VOTE1')
'''y = df['voteA']
X = pd.DataFrame([np.log(df['expendA']), np.log(df['expendB']), np.log(df['prtystrA'])]).T
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()'''
model = sm.formula.ols(formula='voteA ~ np.log(expendA) + np.log(expendB) + np.log(prtystrA)', data=df).fit()
print(model.summary)

# c6
df = wooldridge.dataWoo('WAGE2')
model = sm.formula.ols(formula='np.log(wage) ~ educ + exper +tenure', data=df).fit()
print(model.summary)
f_test = model.f_test('educ=exper')
print(f_test)

# c8
df = wooldridge.dataWoo('401KSUBS')
df = df[df['fsize'] == 1]
model = sm.formula.ols('nettfa ~ inc + age', data=df).fit()
f_test = model.f_test('age = 1')
