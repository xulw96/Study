def main():
    # Groupby
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    df = pd.DataFrame(dict(key1=list('aabba'), key2=['one', 'two', 'one', 'two', 'one'],
                           data1=np.random.randn(5), data2=np.random.randn(5)))
    grouped = df['data1'].groupby(df['key1'])
    print(grouped.mean())  # indexed by unique value in keys
    mean = df['data1'].groupby([df['key1'], df['key2']]).mean()  # pass multiple array as list
    print(mean)  # unique pair of keys
    states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
    years = np.array([2005, 2005, 2006, 2005, 2006])
    mean = df['data1'].groupby([states, years])  # can passing any same-length array/Series to groupby
    mean = df.groupby('key1').mean()  # usually do on same object; nuisance column (not numeric) get ignored
    mean = df.groupby(['key1', 'key2']).size()  # ignore NaN

    for name, group in df.groupby('key1'):
        print(name, group, sep='\n')  # 'name' is the unique key value at that turn; 'group' is for other values
    for name, group in df.groupby(['key1', 'key2']):
        print(name, group, sep='\n')  # 'name' is a tuple for the keys' value
    pieces = dict(list(df.groupby('key1')))
    print(pieces['b'])  # build a dict to get pieces
    grouped = df.groupby(df.dtypes, axis=1)  # group on other axis
    for dtype, group in grouped:
        print(dtype, group, sep='\n')

    s_grouped = df.groupby(['key1', 'key2'])['data2']  # indexing a groupby object; aggregate the values
    print(s_grouped.mean())

    people = pd.DataFrame(np.random.randn(5, 5), columns=list('abcde'),
                          index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
    people.iloc[2:3, [1, 2]] = np.nan
    mapping = dict(a='red', b='red', c='blue', d='blue', e='red', f='orange')
    by_columns = people.groupby(mapping, axis=1)  # passing dict/Series for grouping information
    print(by_columns.sum())
    people.groupby(len).sum()  # pass index, return group names; function is ok to pass
    people.groupby([len, ['one', 'one', 'one', 'two', 'one']])  # everything gets into arrays internally. Fine to combine anything

    columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'], [1, 3, 5, 1, 3]],
                                        names=['city', 'tenor'])
    hier_df = pd.DataFrame(np.random.randn(4, 5), columns=columns)
    hier_df.groupby(level='city', axis=1).count()  # group by levels

    # Data aggregation
    grouped = df.groupby('key1')
    grouped['data1'].quantile(0.9)  # A Series method to be called for aggregation
    def peak_to_peak(arr):
        return arr.max() - arr.min()
    grouped.agg(peak_to_peak)  # pass a function for aggregation
    grouped.describe()  # this works as if aggregation

    tips = pd.read_csv('./data/tips.csv')
    tips['tip_pct'] = tips['tip'] / tips['total_bill']
    grouped = tips.groupby(['day', 'smoker'])
    grouped_pct = grouped['tip_pct']
    result = grouped_pct.agg(['mean', 'std', peak_to_peak])  # calling multi-function to return at the same time
    print(result)
    grouped_pct.agg([('foo', 'mean'), ('bar', np.std)])  # replace function with (name, function)
    grouped['tip_pct', 'total_bill'].agg(['count', 'mean', 'max'])  # apply function to multiple columns
    grouped.agg({'tip_pct': ['min', 'max', 'mean', 'std'], 'size': 'sum'})  # mapping function to columns

    tips.groupby(['day', 'smoker'], as_index=False, group_keys=False).mean()  # always able to reset_index() later

    # Apply
    def top(df, n=5, column='tip_pct'):  # select top five tip_pct value
        return df.sort_values(by=column)[-n:]  # should return a pandas object/ scalar value
    result = tips.groupby(['smoker', 'day']).apply(top, n=1, column='total_bill')  # overide arguments
    print(result)  # function on each row; glue with pandas.concat; labeling with group names;

    frame = pd.DataFrame({'data1': np.random.randn(1000), 'data2': np.random.randn(1000)})
    quartiles = pd.cut(frame.data1, 4)  # sample four equal length
    def get_stats(group):
        return {'min': group.min(), 'max': group.max(), 'count': group.count(), 'mean': group.mean()}
    grouped = frame.data2.groupby(quartiles)
    grouped.apply(get_stats).unstack()
    grouping = pd.qcut(frame.data1, 10, labels=False)  # just get quantile numbers
    grouped = frame.data2.groupby(grouping)
    grouped.apply(get_stats).unstack()

    # group specific func
    states = ['Ohio', 'New York', 'Vermont', 'Florida',
              'Oregon', 'Nevada', 'California', 'Idaho']
    group_key = ['East'] * 4 + ['West'] * 4
    data = pd.Series(np.random.randn(8), index=states)
    data[['Vermont', 'Nevada', 'Idaho']] = np.nan
    fill_values = {'East': 0.5, 'West': -1}  # specify value for different group
    fill_func = lambda g: g.fillna(fill_values[g.name])  # function to return value
    data.groupby(group_key).apply(fill_func)

    # sampling
    suits = list('HSCD')
    card_val = (list(range(1, 11)) + [10] * 3) * 4
    base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
    cards = []
    for suit in suits:
        cards.extend(str(num) + suit for num in base_names)  # extend add by iterating; append add by passing object
    deck = pd.Series(card_val, index=cards)
    def draw(deck, n=5):
        return deck.sample(n)  # randomly select n element
    get_suit = lambda card: card[-1]  # last letter is suit
    deck.groupby(get_suit, group_keys=False).apply(draw, n=2)  # groupby suit and sample two cards

    # grouped weighted aggragation
    df = pd.DataFrame({'category': list('aaaabbbb'), 'data': np.random.randn(8),
                       'weights': np.random.randn(8)})
    get_wavg = lambda g: np.average(g['data'], weights=g['weights'])
    df.groupby('category').apply(get_wavg)  # weighted average
    close_px = pd.read_csv('./data/stock_px_2.csv', parse_dates=True, index_col=0)
    print(close_px.info())
    spx_corr = lambda x: x.corrwith(x['SPX'])  # each column correlation with 'SPX'
    rets = close_px.pct_change().dropna()
    get_year = lambda x: x.year  # get year attribute from each datetime label
    rets.groupby(get_year).apply(spx_corr)  # yearly corrleation
    rets.groupby(get_year).apply(lambda g: g['AAPL'].corr(g['MSFT']))  # inter-column corr

    import statsmodels.api as sm
    def regress(data, yvar, xvars):  # apply Linear regression on group
        y = data[yvar]
        x = data[xvars]
        x['intercept'] = 1
        result = sm.OLS(y, x).fit()
        return result.params
    result = rets.groupby(get_year).apply(regress, 'AAPL', ['SPX'])
    print(result)

    # Pivot table & Cross tabulation
    tips.pivot_table(index=['day', 'smoker'])  # pivot() not handle aggragation; not tolerate duplicates
    tips.pivot_table(['tip_pct', 'size'], index=['time', 'day'],
                     columns='smoker', margins=True)  # add a total row/column to compute means
    tips.pivot_table('tip_pct', index=['time', 'smoker'], columns='day',
                     aggfunc=len, margins=True, fill_value=0)  # change aggregation method; fill NaN
    pd.crosstab([tips.time, tips.day], tips.smoker, margins=True)  # passing index and column; count frequency




if __name__ == '__main__':
    main()