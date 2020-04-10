def main():
    # handle missing data
    import pandas as pd
    import numpy as np
    data = pd.DataFrame([1, np.nan, 3.5, np.nan, 7])
    data.dropna(how='all', axis=1)  # drop columns with all NaN
    data.dropna(thresh=2)  # drop when there is exactly 2 NaN

    df = pd.DataFrame(np.random.randn(7, 3))
    df.fillna({1: 0.5, 2: 0}, inplace=True)  # different replacement for different column
    df.fillna(method='ffill', limit=2)  # same method from reindexing
    df.fillna(df.mean())

    # data transformation
    data = pd.DataFrame({'k1': ['one', 'two'] * 3 + ['two'],
                         'k2': [1, 1, 2, 3, 3, 4, 4]})
    data.duplicated()  # return boolean series on whether row is a duplicate to previous
    data.drop_duplicates(['k1'], keep='last')  # drop based on a single column, keep the last one

    data = pd.DataFrame({'food': ['bacon', 'pulled pork','bacon', 'Pastrami',
                                  'corned beef', 'Bacon', 'pastrami', 'honey ham',
                                  'nova lox'], 'ouces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
    meat_to_animal = {'bacon': 'pig', 'pulled pork': 'pig', 'pastrami': 'cow',
                      'corned beef': 'cow', 'honey ham': 'pig', 'nova lox': 'salmon'}
    lowercased = data['food'].str.lower()  # return a lowercased version
    data['animal'] = lowercased.map(meat_to_animal)  # write a new column based on mapping dict
    data['animal'] = data['food'].map(lambda x: meat_to_animal[x.lower()])  # achieved by passing function

    data = pd.Series([1, -999, 2, -999, -1000, 3])
    data.replace({-999:np.nan, -1000: 0})

    data = pd.DataFrame(np.arange(12).reshape((3, 4)), index=['Ohio', 'Colorado', 'New York'],
                        columns=['one', 'two', 'three', 'four'])
    data.index = data.index.map(lambda x: x[:4].upper())  #  apply works on row/column; applymap (for DF) and map (for Series) works element-wise
    data.rename(index=str.title, columns=str.upper)
    data.rename(index={'OHIO': 'INDIANA'}, columns={'three': 'peekaboo'}, inplace=True)

    ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
    bins = [18, 25, 35, 60, 100]
    cats = pd.cut(ages, bins, right=False)  # put each item into bins
    print(cats)
    print(cats.codes)  # labels for the ages data
    print(cats.categories)
    pd.value_counts(cats)  # counts the bins
    group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
    pd.cut(ages, bins, labels=group_names)  # label for each bin

    data = np.random.randn(20)
    pd.cut(data, 4, precision=2)  # 4 for number of bins, it will auto-create bins' range; decimal precision
    pd.qcut(data, 4)  # based on quantiles, not max() - min()
    pd.qcut(data, [0, 0.1, 0.5, 0.9, 1])  # based on own quantiles

    data = pd.DataFrame(np.random.randn(1000, 4))
    col = data[2]  # select column 2
    col[np.abs(col) > 3]  # conditional select
    data[(np.abs(data) > 3).any(1)]  # select rows with at least one value > 3
    data[np.abs(data) > 3] = np.sign(data) * 3  # transform data

    df = pd.DataFrame(np.arange(5 * 4).reshape((5, 4)))
    samples = np.random.permutation(5)  # a random permuted list
    df.take(samples)  # using the permuted list to locate; .take() is equal to iloc-indexing
    df.sample(n=3, replace=True)  # randomly select 3 rows; allow repetition

    df = pd.DataFrame({'key': list(bbacab), 'data1': range(6)})
    pd.get_dummies(df['key'], prefix='key')  # a matrix indicating 1 for existing value; prefix to column names
    df_with_dummy = df[['data1']].join(dummies)  # the index as primary key

    # string manipulate
    val = 'a,b,  guido'
    val.split(',')
    val.strip()
    val.replace(',', ';')

    import re
    text = 'foo    bar\t baz \tqux'
    regex = re.compile('\s+', flag=re.IGNORECASE)  # case-insensitive
    regex.findall(text)  # find items matching the regex; return list of tuples (accessible with groups()) when pattern has groups
    regex.search(text)  # for the first appearance
    regex.match(text)  # for existing at the start of the string
    regex.sub('REDACTED', text)  # replace pattern with new string
    regex.sub(r'Username: \1, Domain: \2, Suffix: \3', text) # to ignore escaping char, prefixing r to the items, instead of 'c:\\x'

    data = pd.Series(['dave@google.com', 'steve@gmail.com', 'rob@gmail.com', np.nan],
                     index=['Dave', 'Steve', 'Rob', 'Wes'])
    data.str.contains('gmail')  # first using str; otherwise, get bug meeting NaN
    pattner = '([A-Z0-9._%+-]+)@([A-Z0-9.-]\\.([A-Z]{2, 4}'
    matches = data.str.match(pattern, flags=re.IGNORECASE)



if __name__ == '__main__':
    main()