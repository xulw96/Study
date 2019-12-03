def main():
    import pandas as pd
    from pandas import Series, DataFrame
    # Series
    data = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
    obj1 = pd.Series(data)  # passing dict to build a Series
    states = ['California', 'Ohio', 'Oregon', 'Texas']
    obj2 = pd.Series(obj1, index=states)  # getting result with corresponding index
    obj2.isnull()  # equal to pd.isnull(obj)
    print(obj1, '\n', obj2)
    obj1 + obj2  # data alignment with index; similar to join
    obj2.name = 'Population'  # Series has name attribute
    obj2.index = ['Bob', 'Steve', 'Jeff', 'Ryan']  # in-place index assignment

    # Dataframe
    data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
            'year': [2000, 2001, 2002, 2001, 2002, 2003],
            'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}   # build from equal length dict
    frame = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                         index=['one', 'two', 'three', 'four', 'five', 'six'])  # columns select from the keys; NaN for not found values
    frame['year']
    frame.year  # two ways to retrive column in dataframe
    frame.loc['three']  # retrive row in dataframe
    val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
    frame['debt'] = val  # assign with Series; the value will auto-align with index.
    frame['eastern'] = frame.state == 'Ohio'  # bulid a new column; assign True when state is 'Ohio'
    del frame['eastern']
    frame.T

    # index
    import numpy as np
    labels = pd.Index(np.arange(3))  # an Index object; can hold same element multiple times (unlike key in database); select all when specified
    obj = pd.Series([1.5, -2.5, 0], index=labels)
    obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])

    '''obj2 = obj.reindex(range(6), method='ffill')  # rearrange with new index; forward-fill for not-existing'''
    obj = pd.DataFrame(np.arange(9).reshape(3, 3), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
    obj.reindex(columns=['Texas', 'Utah', 'California'])
    print(obj.loc[['a', 'b', 'c', 'd'], ['Texas', 'Utah', 'California']])  # access group with both columns and index

    # drop, select
    index = pd.Index(['Ohio', 'Colorado', 'Utah', 'New York'])
    columns = ['one', 'two', 'three', 'four']
    data = pd.DataFrame(np.arange(16).reshape((4, 4)), index=index, columns=columns)
    data.drop('two', axis=1)  # axis=1; axis='columns' to point to the table column
    '''data.drop('c', inplace=True)'''  # in-place manipulation, no new object created!
    data[data['three'] > 5 ]  # select rows with a boolean array
    data.iloc[:, :3][data.three > 5]  # multiple selection
    data.iat[1, 2]  # select a single scalar value

    # operation
    df1 = pd.DataFrame(np.arange(12).reshape((3, 4)), columns=list('abcd'))
    df2 = pd.DataFrame(np.arange(20).reshape((4, 5)), columns=list('abcde'))
    df1.reindex(columns=df2.columns, fill_value=0)  # most function take in fill_value for NaN
    frame = pd.DataFrame(np.arange(12).reshape((4, 3)), columns=list('bde'),
                         index=['Utah', 'Ohio', 'Texas', 'Oregon'])
    series = frame.iloc[0]
    print(frame.sub(series, axis='index'))  # 'broadcasting' to each row; manually set to match index

    frame.apply(lambda x: x.max() - x.min())  # apply a function; input row or column
    def f(x):
        return pd.Series([x.min(), x.max()], index=['min', 'max'])
    frame.apply(f)  # can return multiple values
    frame.applymap(lambda x: '%.2f' % x)  # this is element-wise

    obj = pd.Series(range(4), index=['d', 'a', 'b', 'c'])
    frame = pd.DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'], columns=['d', 'a', 'b', 'c'])
    frame.sort_index(axis=1, ascending=False)
    frame.sort_values(by=['a', 'b'])

    obj = pd.Series([7, -5, 7, 4, 2, 0, 4])
    obj.rank(method='first')  # output the rank with same index
    frame = pd.DataFrame({'b': [4.3, 7, -3, 2], 'a': [0, 1, 0, 1], 'c': [-2, 5, 8, -2.5]})
    frame.rank(axis='columns')

    df = pd.DataFrame(np.random.randn(4, 3), index=['a', 'a', 'b', 'c'])  # duplicated index
    df.loc['b']  # get two rows

    # summary and description
    df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=['a', 'b', 'c', 'd'],
                      columns=['one', 'two'])
    df.sum()
    df.mean(axis='column', skipna=False)
    df.idxmax()  # return the corresponding index for the value
    df.cumsum()  # accumulations
    df.describe()

    # correlation and covariance
    import pandas_datareader.data as web
    all_data = {ticker: web.get_data_yahoo(ticker) for ticker in ['APPL', 'IBM', 'MSFT', 'GOOG']}
    price = pd.DataFrame({ticker: data['Adj Close'] for ticker, data in all_data.items()})
    volume = pd.DataFrame({ticker: data['Volume'] for ticker, data in all_data.items()})
    returns = price.pct_change()  # percent change
    returns.tail()  # last 5 rows
    returns['MSFT'].corr(returns['IBM'])  # compute correlation
    returns['MSFT'].cov(returns['IBM'])  # compute covariance
    returns.corr()  # a full correlation values table
    returns.cov()
    returns.corrwith(returns.IBM)  # pair-wise computation with manual input

    # unique, value_count, membership
    obj = pd.Series(list('cadaabbcc'))
    uniques = obj.unique()
    obj.value_counts()
    mask = obj.isin(['b', 'c'])  # return a boolean check series
    to_match = pd.Series(list('cabbca'))
    unique_vals = pd.Series(list('cba'))
    pd.Index(unique_vals).get_indexer(to_match)  # return array of index by slices
    data = pd.DataFrame({'Qu1': list('13434'), 'Qu2': list('23123'), 'Qu3': list('15244')})
    result = data.apply(pd.value_counts, fillna(0))  # wrangling


if __name__ == '__main__':
    main()