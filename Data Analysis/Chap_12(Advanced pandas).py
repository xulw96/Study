import pandas as pd
import numpy as np

# Categorical
values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(['apple', 'orange'])
dim.take(values)  # assign integer with string;

fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
df = pd.DataFrame({'fruit': fruits, 'basket_id': np.arange(N), 'count': np.random.randint(3, 15, size=N),
                   'weight': np.random.uniform(0, 4, size=N)}, columns=['basket_id', 'fruit', 'count', 'weight'])
fruit_cat = df['fruit'].astype('category')  # convert into pd.Category
categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
ordered_cat = pd.Categorical.from_codes(codes, categories, ordered=True)  # building a Category with meaningful ordering

draws = np.random.randn(1000)
bins = pd.qcut(draws, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  # a categories object
print(bins.codes[:10], bins.categories[:10])
bins = pd.Series(bins, name='quartile')
results = pd.Series(draws).groupby(bins).agg(['count', 'min', 'max']).reset_index()  # a quatile analyse

N = 10000000
draws = pd.Series(np.random.randn(N))
labels = pd.Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))  # double slash for getting interger division
categories = labels.astype('category')
print(labels.memory_usage(), categories.memory_usage())

cat_s = pd.Series(list('abcd') * 2).astype('category')
print(cat_s.cat.codes, cat_s.cat.categories)
actual_categories = list('abcde')
cat_s.cat.set_categories(actual_categories)  # convert categories
cat_s.cat.remove_unused_categories()  # as name suggests
pd.get_dummies(cat_s)  # one-hot encoding

# Advanced GroupBy
df = pd.DataFrame({'key': ['a', 'b', 'c'] * 4, 'value': np.arange(12)})
g = df.groupby('key').value
g.transform('mean')  # built-int will be faster than using apply
g.transform(lambda x: x.rank(ascending=False))
g.transform(lambda x: (x - x.mean() / x.std()))  # similar to apply
normalized = ((df['value'] - g.transform('mean')) / g.transform('std'))  # to normalized data

time = pd.date_range('2017-05-20 00:00', freq='1min', periods=15)
df = pd.DataFrame({'time': time.repeat(3), 'key': np.tile(['a', 'b', 'c'], 15), 'value': np.arange(45)})
time_key = pd.Grouper(freq='5min')
resampled = df.set_index('time').groupby(['key', time_key]).sum()  # groupby with two index; Time must be the index
resampled.reset_index()

# Method Chaining
df2 = df.assign(k=v)  # equal to the following; enable method chaining
df2 = df.copy()
df2['k'] = v

result = (load_data()[lambda x: x.col2 < 0].assign(col1_demeaned=lambda x: x.col1 - x.col1.mean())
          .groupby('key').col1_demeaned.std())  # chained methods

def group_demean(df, by, cols):
    result = df.copy()
    g = df.groupby(by)
    for c in cols:
        result[c] = df[c] - g[c].transform('mean')
    return result
result = (df[df.col1 < 0].pipe(group_demean, ['key1', 'key2'], ['col1']))  # using pipeline to eanble function within method chaining


