# date and time
from datetime import datetime
now = datetime.now()
print(now.year, now.month, now.day)
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)
print(delta)  # a tuple of '(day, seconds)'
from datetime import timedelta
start = datetime(2011, 1, 1)
print(start - 2 * timedelta(12))

stamp = datetime(2011, 1, 3)
str(stamp)
stamp.strftime('%Y-%m-%d')  # specify format
value = '2011-01-03'
datetime.strptime(value, '%Y-%m-%d')  # specify strings to parse date
datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]

from dateutil.parser import parse
parse('2011-01-03')  # easilly parse date
parse('Jan 31, 1997 10:45 PM')
parse('6/12/2011', dayfirst=True)  # day before month
import pandas as pd
datestrs = ['2011-07-06 12:00:00', '2011-08-06 00:00:00']
pd.to_datetime(datestrs + [None])  # pandas method to convert datetime; handles NaT(not a time)

# Time Series
import numpy as np
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),
         datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = pd.Series(np.random.randn(6), index=dates)  # a Time series
print(ts.index)  # a DatetimeIndex object with datetime64[ns] data type
print(ts.index[0])  # a Timestamp object for each value

ts['1/10/2011'], ts['20110110']  # get value by passing datetime
long_ts = pd.Series(np.random.randn(1000), index=pd.date_range('2011-1-1', periods=1000))
long_ts['2011']  # using only year to select
long_ts[:datetime(2011, 1, 7)]  # select date before/after a certain date by slicing
long_ts['1/6/2011':'1/11/2011']  # select date range  (chronologically ordered time series)
long_ts.truncate(after='1/9/2011')  # data range by indexing

dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = pd.DataFrame(np.random.randn(100, 4), index=dates,
                       columns=['Colorado', 'Texas', 'New York', 'Ohio'])
long_df.loc['5-2001']  # works for DataFrame too

dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000',
                         '1/3/2000'])
dup_ts = pd.Series(np.arange(5), index=dates)
grouped = dup_ts.groupby(level=0)  # fix duplicate value by grouping on level 0

# frequencies
index = pd.date_range('2012-04-01', '2012-06-01')  # DatetimeIndex with range of particular frequency
pd.date_range(start='2012-04-01', periods=20, freq='BM')  # specify frequency type
pd.date_range('2012-05-02 12:56:31', periods=5, normalize=True)  # normalize date to midnight

from pandas.tseries.offsets import Hour, Minute
new_frequency = Hour(2) + Minute(30)  # new offsets
pd.date_range('2000-01-01', periods=10, freq='1h30min')
pd.date_range('2000-01-01', periods=10, freq='4h')
rng = pd.date_range('2012-01-01', '2012-09-01', freq='WOM-3FRI')  # third friday every month
print(rng)

ts = pd.Series(np.random.randn(4), index=pd.date_range('1/1/2000', periods=4, freq='M'))
pct_change = ts / ts.shift(1) - 1  # shift data; index not changed; getting NaN
ts.shift(2, freq='M')  # passing frequency enables shifting index; data not discarded

from pandas.tseries.offsets import Day, MonthEnd
now = datetime(2011, 11, 17)
print(now + 3 * Day())
print(now + 3 * MonthEnd(2))  # add anchored offset
roll = MonthEnd(2).rollforward(now)  # roll dates using offsets' method
ts = pd.Series(np.random.randn(20), index=pd.date_range('1/15/2000', periods=20, freq='4d'))
mean = ts.groupby(MonthEnd(2).rollforward).mean()  # creatively getting average for each month

# Time zone
import pytz
tz = pytz.timezone('Asia/Shanghai')  # a time zone object
pd.date_range('3/9/2012 9:30', periods=10, freq='D', tz='UTC')  # date ranges with time zone set; otherwise, time zone naive
ts_utc = ts.tz_localize('UTC')  # passing times series into localized time zone
ts_utc.tz_convert('Asia/Shanghai')  # convert time zone
ts.index.tz_localize('UTC')  # not only Series, but also DatetimeIndex's instance method

stamp = pd.Timestamp('2011-03-12 04:00')  # works for Timestamp too
stamp_utc = stamp.tz_localize('utc')
stamp_utc.tz_convert('Asia/Shanghai')
stamp_moscow = pd.Timestamp('2011-03-12 04:00', tz='Europe/Moscow')  # passing timezone when building it

rng = pd.date_range('3/7/2012 9:30', periods=10, freq='B')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
result = ts[:7].tz_localize('Asia/Shanghai') + ts[2:].tz_localize('Europe/Moscow')
print(result.index)  # integrate different UTC into 'utc'

# Periods
period = pd.Period(2007, freq='A-DEC')  # period is a timespan
print(period + 5)  # shifting by frequency
rng = pd.period_range('2000-01-01', '2000-06-30', freq='M')  # sequence of periods within the range; PeriodIndex

period.asfreq('M', how='start')  # convert Period into another frequency
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts.asfreq('M', how='start')  # convert a PeriodIndex

p = pd.Period('2012Q4', freq='Q-JAN')
p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60  # second-to-last business day of the quarter
rng = p4pm.to_timestamp()  # convert to Timestamp

rng = pd.date_range('2000-01-02', periods=3, freq='M')
ts = pd.Series(np.random.randn(3), index=rng)
ts.to_period()  # return back

data = pd.read_csv('./data/macrodata.csv')  # having a year and quarter attributes
index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')  # combine to form an index
data.index = index  # assign it
print(data.infl)

# Resampling and Frequency Conversion
rng = pd.date_range('2000-01-01', periods=12, freq='T')
ts = pd.Series(np.arange(12), index=rng)
ts.resample('5min', closed='right', label='left', loffset='-1s').sum()  # a groupby and aggregation
ts.resample('5min').ohlc()  # compute open, high, low, close

index=pd.date_range('1/1/2000', periods=2, freq='W-WED')
frame = pd.DataFrame(np.random.randn(2, 4), index=index, columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame.resample('D').asfreq()  # no aggregation
frame.resample('D').ffill(limit=2)  # fulfill the NaN

frame = pd.DataFrame(np.random.randn(24, 4), index=pd.period_range('1-2000', '12-2001', freq='M'),
                     columns=['Colorado', 'Texas', 'New York', 'Ohio'])
annual_frame = frame.resample('A-DEC').mean()
annual_frame.resample('Q-DEC', convention='end').ffill()

# moving window functions
import matplotlib.pyplot as plt
close_px_all = pd.read_csv('./data/stock_px_2.csv', parse_dates=True, index_col=0)
close_px = close_px_all[['AAPL', 'MSFT', 'XOM']].resample('B').ffill()
close_px.AAPL.plot()
close_px.AAPL.rolling(250).mean().plot()  # rolling is similar to groupby and resample
appl_std250 = close_px.AAPL.rolling(250, min_periods=10).std()  # minimum non-NaN value
appl_std250.plot()
plt.close()
expanding_mean = appl_std250.expanding().mean()  # expanding is another type
close_px.rolling('20D').mean()  # rolling by periods number or time offset

aapl_px = close_px.AAPL['2006':'2007']
ma60 = aapl_px.rolling(30, min_periods=20).mean()
ewm60 = aapl_px.ewm(span=30).mean()  # exponentially weighted functions; more weight on recent observations
ma60.plot(style='k--', label='Simple MA')
ewm60.plot(style='k-', label='EW MA')
plt.legend()
plt.close()

returns = close_px.pct_change()
spx_rets = close_px_all['SPX'].pct_change()
corr = returns.rolling(125, min_periods=100).corr(spx_rets)
corr.plot()
plt.close()

from scipy.stats import percentileofscore
score_at_2percent = lambda x: percentileofscore(x, 0.02)
result = returns.AAPL.rolling(250).apply(score_at_2percent, raw=True)  # user-defined aggregation methods
result.plot()
