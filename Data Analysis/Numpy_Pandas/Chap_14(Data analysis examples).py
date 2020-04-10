import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bitly
import json
with open('./data/example.txt') as data:
    records = [json.loads(line) for line in data]
frame = pd.DataFrame(records)
tz_counts = frame['tz'].value_counts()  # timezone info

subset = tz_counts[:10]
sns.barplot(y=subset.index, x=subset.values)
plt.close()

results = pd.Series([x.split()[0] for x in frame.a.dropna()])  # browser info
results.value_counts()

cframe = frame[frame.a.notnull()]  # check os system
cframe['os'] = np.where(cframe['a'].str.contains('Windows'), 'Windows', 'Not Windows')

bt_tz_os = cframe.groupby(['tz', 'os'])  # multiindex grouping
agg_counts = bt_tz_os.size().unstack().fillna(0)  # group counts with size; similar to value_counts

indexer = agg_counts.sum(1).argsort()
count_subset = agg_counts.take(indexer[-10:])
'''count_subset = agg_counts.sum(1).nlargest(10)'''

count_subset = count_subset.stack()
count_subset.name = 'total'
count_subset = count_subset.reset_index()
sns.barplot(x='total', y='tz', hue='os', data=count_subset)
plt.close()


def norm_total(group):  # normalization
    group['normed_total'] = group.total / group.total.sum()
    return group


results = count_subset.groupby('tz').apply(norm_total)
'''g = count_subset.groupby('tz')
results2 = count_subset.total / g.total.transform('sum')'''
sns.barplot(x='normed_total', y='tz', hue='os', data=results)
plt.close()

# MovieLens 1M
pd.options.display.max_rows = 10
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
mnames = ['movie_id', 'title', 'genres']
users = pd.read_csv('./data/users.dat', sep='::', header=None, names=unames, engine='python')
ratings = pd.read_csv('./data/ratings.dat', sep='::', header=None, names=rnames, engine='python')
movies = pd.read_csv('./data/movies.dat', sep='::', header=None, names=mnames, engine='python')
data = pd.merge(pd.merge(users, ratings), movies)  # join into one table

mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title > 250]  # manage only votes over 250
mean_ratings = mean_ratings.loc[active_titles]  # use the index to select
top_female_ratings = mean_ratings.sort_values(by='F', ascending=False)

mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']
sorted_by_diff = mean_ratings.sort_values(by='diff')
rating_std_by_title = data.groupby('title')['rating'].std()
rating_std_by_title = rating_std_by_title.loc[active_titles]
sorted_by_std = rating_std_by_title.sort_values(ascending=False)

# Baby names
years = range(1880, 2011)
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = './data/babynames/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year  # a year column for each file
    pieces.append(frame)
names = pd.concat(pieces, ignore_index=True)  # multi-list into a single DataFrame
total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
total_births.plot(title='Total births by sex and year')
plt.close()
def add_prop(group):
    group['prop'] = group.births / group.births.sum()
    return group
names = names.groupby(['year', 'sex']).apply(add_prop)

def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]
grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
top1000.reset_index(inplace=True, drop=True)

boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births', index='year', columns='name', aggfunc=sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False, title="Number of births per year")

df = boys[boys.year == 2010]
prop_cumsum = df.sort_values(by='prop', ascending=False).prop.cumsum()
prop_cumsum.values.searchsorted(0.5)  # get the 50% position of prop
def get_quantile_count(group, q=0.5):
    group = group.sort_values(by='prop', ascending=False)
    return group.prop.cumsum().values.searchsorted(0.5) + 1
diversity = top1000.groupby(['year', 'sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')

get_last_letter = lambda x: x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table('birth', index=last_letters, columns=['sex', 'year'], aggfunc=sum)
subtable = table.reindex(columns=[1910, 1960, 2010], level='year')
letter_prop = subtable / subtable.sum()
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
letter_prop['M'].plot(kind='bar', rot=0, ax=axes[0], title='Male')
letter_prop['F'].plot(kind='bar', rot=0, ax=axes[1], title='Female', legend=False)
plt.close()
letter_prop = table / table.sum()
dny_ts = letter_prop.loc[['d', 'n', 'y'], 'M'].T

all_names = pd.Series(top1000.name.unique())
lesley_like = all_names[all_names.str.lower().str.contains('lesl')]
filtered = top1000[top1000.name.isin(lesley_like)]
table = filtered.pivot_table('births', index='year', columns='sex', aggfunc='sum')
table = table.div(table.sum(1), axis=0)
table.plot(style={'M': 'k-', 'F': 'k--'})
plt.close()

# USDA Food
with open('./data/database.json') as data:
    db = json.load(data)
nutrients = pd.DataFrame(db[0]['nutrients'])
info_keys = ['description', 'group', 'id', 'manufacturer']
info = pd.DataFrame(db, columns=info_keys)
nutrients = nutrients.drop_duplicates()

col_mapping = {'description': 'food', 'group': 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
col_mapping = {'description': 'nutrient', 'group': 'fgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)
ndata = pd.merge(nutrients, info, on='id', how='outer')
result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)  # count the median value
result['Zinc', 'Zn'].sort_values().plot(kind='barh')
plt.close()

by_nutrient = ndata.groupby(['nutgroup', 'nutrient'])
get_maximum = lambda x: x.loc[x.value.idxmax()]
get_minimum = lambda x: x.loc[x.value.idxmin()]
max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]
max_foods.loc['Amino Acids']['food']

