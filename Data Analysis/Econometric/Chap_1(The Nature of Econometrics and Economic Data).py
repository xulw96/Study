import wooldridge

# c3.5
df = wooldridge.dataWoo('MEAP01')
corr = df['math4'].corr(df['read4'])
