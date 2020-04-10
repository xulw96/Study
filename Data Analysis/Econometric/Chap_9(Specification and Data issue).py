import wooldridge
import pandas as pd
import numpy as np
import statsmodels.api as sm

# c5
df = wooldridge.dataWoo('RDCHEM')
lad = sm.formula.quantreg('rdintens ~ sales + np.square(sales) + profmarg', data=df).fit(q=0.5)
student_resid = lad.outlier_test()
outliers = student_resid[student_resid['student_resid'] > 1.96].index

ols1 = sm.formula.ols('rdintens ~ sales + np.square(sales) + profmarg', data=df).fit()
ols2 = sm.formula.ols('rdintens ~ sales + profmarg', data=df).fit()
sm.stats.diagnostic.compare_cox(ols1, ols2)