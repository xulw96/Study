def main():
    # Hierachical index
    import numpy as np
    import pandas as pd
    data = pd.Series(np.random.randn(9), index=[list('aaabbccdd'),
                                                [1, 2, 3, 1, 3, 1, 2, 2, 3]])
    print(data.index)  # MultiIndex
    data.loc[['b', 'c']]  # selection
    data.loc[:, 2]  # select with the second level index
    data.unstack()  # like a pivot table; rearrange to DataFrame
    data.unstack().stack()
    frame = pd.DataFrame(np.arange(12).reshape((4, 3)), index=[list('aabb'), [1, 2, 1, 2]],
                         columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
    frame.index.names = ['key1', 'key2']
    frame.columns.names = ['state', 'color']
    print(frame)
    frame.index.from_arrays([['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']],
                            names=['states', 'color'])  # another way to assign index
    print(frame)

    frame.swaplevel('key1', 'key2')  # index's levels interchanged
    frame.swaplevel(0, 1).sort_index(level=0)  # sort the values together with the index
    frame.sum(level='color', axis=1)  # statistics with level (perform on same index; groupby API)
    frame = pd.DataFrame({'a': range(7), 'b': range(7, 0, -1),
                          'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                          'd': [0, 1, 2, 0, 1, 2, 3]})
    frame = frame.set_index(['c', 'd'], drop=False)  # set columns to be index, keep the columns too
    '''frame = frame.reset_index()'''  # give back the column and create new 0-based index

    # combine and merge
    df1 = pd.DataFrame({'key': list('bbacaab'), 'data1': range(7)})
    df2 = pd.DataFrame({'key': list('abd'), 'data2': range(3)})
    pd.merge(df1, df2, on='key')  # specify merging key; many-to-one join; many-to-many will form Cartesian product
    df3 = pd.DataFrame({'lkey': list('bbacaab'), 'data1': range(7)})
    df4 = pd.DataFrame({'rkey': list('abd'), 'data2': range(3)})
    pd.merge(df3, df4, left_on='lkey', right_on='rkey', how='outer')  # specify key for each table; choose outer join
    df5 = pd.DataFrame({'key1': ['foo', 'foo', 'bar'],
                        'key2': ['one', 'two', 'one'],
                        'lval': [1, 2, 3]})
    df6 = pd.DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                        'key2': ['one', 'one', 'one', 'two'],
                        'rval': [4, 5, 6, 7]})
    pd.merge(df5, df6, on=['key1', 'key2'], how='outer')  # merge based on key combination, NaN for not existing value
    pd.merge(df5, df6, on='key1')  # auto-suffix (_x, _y) to same columns
    pd.merge(df5, df6, on='key1', suffixes=('_left', '_right'))

    lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                           'key2': [2000, 2001, 2002, 2001, 2002],
                           'data': np.arange(5)})
    righth = pd.DataFrame(np.arange(12).reshape((6, 2)),
                          index=[['Nevada', 'Nevada', 'Ohio', 'Ohio'],
                                   [2001, 2000, 2000, 2000, 2001, 2002]],
                          columns=['event1', 'event2'])
    pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')  # merge on index
    left2 = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=list('ace'), columns=['Ohio', 'Nevada'])
    right2 = pd.DataFrame([[7, 8], [9, 10], [11, 12], [13, 14]],
                          index=list('bcde'), columns=['Missouri', 'Alabama'])
    another = pd.DataFrame([[7, 8], [9, 10], [11, 12], [16, 17]],
                           index=list('acef'), columns=['New York', 'Oregon'])
    left2.join([right2, another], on='key', how='outer')  # join method to perform merge

    s1 = pd.Series([0 ,1], index=['a', 'b'])
    s2 = pd.Series([0, 1, 5, 6], index=['a', 'b', 'f', 'g'])
    s3 = pd.Series([5, 6], index=['f', 'g'])
    pd.concat([s1, s2], axis=1, join='outer')  # a general perpurse join/merge
    pd.concat([s1, s2], axis=1, join_axes=[['a', 'c', 'b', 'e']])  # specify index to be joined; NaN for not existing
    result = pd.concat([s1, s1, s3], keys=['one', 'two', 'three'])  # hierachical index for each concating obj;
    result = result.unstack()  # rearange to Dataframe
    pd.concat([s1, s2, s3], axis=1, keys=['one', 'two', 'three'])  # keys become headers for columns

    df1 = pd.DataFrame(np.arange(6).reshape((3, 2)), index=list('abc'), columns=['one', 'two'])
    df2 = pd.DataFrame(5 + np.arange(4).reshape((2, 2)), index=list('ac'), columns=['three', 'four'])  # add value locally to array
    pd.concat([df1, df2], axis=1, keys=['level1', 'level2'], names=['upper', 'lower'])  # same logic to Series; names for the levels
    pd.concat({'level1': df1, 'level2': df2}, axis=1)  # same thing as above
    pd.concat([df1, df2], ignore_index=True)  # concat on axis 0; ignore the original index

    a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=list('fedcba'))
    b = pd.Series(np.arange(len(a)), dtype=np.float64, index=list('fedcba'))
    np.where(pd.isnull(a), b, a)  # if pd.isnull(), then b; else a.
    b[:-2].combine_first(a[:2])  # replace NaN value with the input
    df1 = pd.DataFrame({'a': [1, np.nan, 5, np.nan],
                        'b': [np.nan, 2, np.nan, 6],
                        'c': range(2, 18, 4)})
    df2 = pd.DataFrame({'a': [5, 4, np.nan, 3, 7],
                        'b': [np.nan, 3, 5, 6, 8]})
    df1.combine_first(df2)  # same task done on DataFrame

    # reshaping and pivoting
    data = pd.DataFrame(np.arange(6).reshape((2, 3)), index=pd.Index(['Ohio', 'Colorado'], name='state'),
                        columns=pd.Index(['one', 'two', 'three'], name='number'))  # assign by object 'Index'
    result = data.stack()  # back to MultiIndex Series
    result.unstack(0)
    result.unstack('state')  # default to put innermost index to columns; but can pass argument to that
    s1 = pd.Series([0, 1, 2, 3], index=['a', 'b', 'c', 'd'])
    s2 = pd.Series([4, 5, 6], index=['c', 'd', 'e'])
    data2 = pd.concat([s1, s2], keys=['one', 'two'])
    data2.unstack()  # will generate NaN
    data2.unstack().stack(dropna=False)  # otherwise, auto-drop NaN value
    df = pd.DataFrame({'left': result, 'right': result + 5},
                      columns=pd.Index(['left', 'right'], name='side'))
    df.unstack('state')  # unstacked level will be in lowest level in columns
    df.unstack('state').stack('side')

    pivoted = data.pivot('date', 'item', 'value')  # first two for index/columns, third for value
    pivoted = data.pivot('date', 'item')  # ignore third one; getting hierarchical columns for multiple-values
    unstacked = data.set_index(['date', 'item']).unstack('item')  # set_index().unstack() equals to pivot()

    df = pd.DataFrame({'key': ['foo', 'bar', 'baz'],
                       'A': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
    melted = pd.melt(df, ['key'])  # 'key' as group indicator to concat other columns into one 'column_name: value' pair
    melted.pivot('key', 'variable', 'value')  # reverse melt
    pd.melt(df, id_vars=['key'], value_vars=['A', 'B'])  # melt only a subset; specify indicator and values


if __name__ == '__main__':
    main()