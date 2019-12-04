def main():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    data = np.arange(10)
    plt.plot(data)
    plt.close()

    fig = plt.figure()  # Figure pane;
    ax1 = fig.add_subplot(2, 2, 1)  # return AxesSubplot object
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    plt.plot(np.random.randn(50).cumsum(), 'k--')  # auto-add to the last subfig
    ax1.hist(np.random.randn(100), bins=20, color='k', alpha=0.3)  # use the subplot object directly to draw
    ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.randn(30))
    plt.close()

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)  # with same axes limit; new figure pane
    for i in range(2):
        for j in range(2):
            axes[i, j].hist(np.random.randn(500), bins=50, color='k', alpha=0.5)  # axes[i ,j] to get each subplot
    plt.subplots_adjust(wspace=0, hspace=0)  # adjust spacing around Fig
    plt.close()

    fig = plt.figure()
    plt.plot(np.random.randn(30).cumsum(), 'ko--')
    plt.plot(np.random.randn(30).cumsum(), color='k', linestyle='dashed', marker='o')
    plt.close()
    fig = plt.figure()
    data = np.random.randn(30).cumsum()
    plt.plot(data, 'k--', label='Default')
    plt.plot(data, 'k-', drawstyle='steps-post', label='step-post')  # not linearly interpolate
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1 ,1)
    ax.plot(np.random.randn(1000).cumsum())
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels(['one', 'two', 'three', 'four', 'five'], rotation=30, fontsize='small')
    ax.set_title('My first plot')
    ax.set_xlabel('Stages')
    ax.set(**{'title': 'my first plot', 'xlabel': 'Stages'})  # batch setting, remember parsing positional keyward
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.random.randn(1000).cumsum(), 'k', label='one')
    ax.plot(np.random.randn(1000).cumsum(), 'k--', label='two')
    ax.plot(np.random.randn(1000).cumsum(), 'k.', label='three')
    ax.legend(loc='best')  # call to draw the legend on Fig
    plt.close()

    from datetime import datetime
    from pandas.plotting import register_matplotlib_converters
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    data = pd.read_csv('./data/spx.csv', index_col=0, parse_dates=True)
    register_matplotlib_converters()
    crisis_data = [(datetime(2007, 10, 11), 'Peak of bull market'),
                   (datetime(2008, 3, 12), 'Bear Stearns Fails'),
                   (datetime(2008, 9, 15), 'Lehman Bankruptcy')]
    ax.plot(data['SPX'], 'k-')
    for date, label in crisis_data:  # data.asof() grant neareast value for NaN
        ax.annotate(label, xy=(date, data['SPX'].asof(date) + 75),
                    xytext=(date, data['SPX'].asof(date) + 225),
                    arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
                    horizontalalignment='left', verticalalignment='top')  # xytext for arrow
    ax.set_xlim(['1/1/2007', '1/1/2011'])
    ax.set_ylim([600, 1800])
    ax.set_title('Important dates in the 2008-2009 financial crisis')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rec = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='k', alpha=0.3)
    circ = plt.Circle((0.7, 0.2), 0.15, color='b', alpha=0.3)
    pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color='g', alpha=0.5)
    ax.add_patch(rec)  # build the object and calling add_patch() to add it
    ax.add_patch(circ)
    ax.add_patch(pgon)
    plt.close()

    '''plt.savefig('example.svb', dpi=600, bbox_indches='tight')'''  # trim the whitespace around
    '''from io import BytesIO
    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()''' # save to file-like object

    '''plt.rc('figure', figsize=(10, 10)) 
    font_options = dict(family='monospace', weight='bold', size='small')
    plt.rc('font', **font_options)''' # change configuration

    # Pandas and Seaborn
    s = pd.Series(np.random.randn(10).cumsum(), index=np.arange(0, 100, 10))
    s.plot()  # call plot() method of Series/Dataframe
    plt.close()
    df = pd.DataFrame(np.random.randn(10, 4).cumsum(0), columns=list('ABCD'), index=np.arange(0, 100, 10))  # input to cumsum()
    df.plot()  # equal to df.plot.line()
    plt.close()

    fig, axes = plt.subplots(2, 1)
    data = pd.Series(np.random.rand(16), index=list('abcdefghijklmnop'))
    data.plot.bar(ax=axes[0], color='k', alpha=0.7)  # ax specify which subplot figure to plot on
    data.plot.barh(ax=axes[1], color='k', alpha=0.7)
    plt.close()

    fig, axes = plt.subplots(2, 1)
    df = pd.DataFrame(np.random.rand(6, 4),
                      index=['one', 'two', 'three', 'four', 'five', 'six'],
                      columns=pd.Index(list('ABCD'), name='Genus'))
    df.plot.bar(ax=axes[0])  # group bar, with columns names on legend
    df.plot.bar(ax=axes[1], stacked=True, alpha=0.5)  # stacked group bar
    plt.close()

    tips = pd.read_csv('./data/tips.csv')
    party_counts = pd.crosstab(tips['day'], tips['size'])  # count frequency with corresponding attribtue
    party_counts = party_counts.loc[:, 2:5]  # refine
    party_pcts = party_counts.div(party_counts.sum(1), axis=0)  # normalize each row; sum to 1
    party_pcts.plot.bar()
    plt.close()

    import seaborn as sns
    tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])
    sns.barplot(x='tip_pct', y='day', hue='time', data=tips, orient='h')  # need to input data source; hue for additional value
    sns.set(style='whitegrid')  # change config for whole
    plt.close()

    fig, axes = plt.subplots(3, 1)
    tips['tip_pct'].plot.hist(bins=50, ax=axes[0])  # histogram shows value frequency
    tips['tip_pct'].plot.density(ax=axes[1])
    tips['tip_pct'].plot.kde(ax=axes[2])  # specify using 'kernel density estimate' to compute
    plt.close()

    comp1 = np.random.normal(0, 1, size=200)
    comp2 = np.random.normal(10, 2, size=200)
    values = pd.Series(np.concatenate([comp1, comp2]))
    sns.distplot(values, bins=100, color='k')  # hist and density in one graph; distplot()
    plt.close()

    macro = pd.read_csv('./data/macrodata.csv')
    data = macro[['cpi', 'm1', 'tbilrate', 'unemp']]
    trans_data = np.log(data).diff().dropna()  # log difference of each value
    sns.regplot('m1', 'unemp', data=trans_data)  # scatter plot with regression line
    plt.title('Changes in log m1 versus log unemp')
    plt.close()
    fig = plt.Figure()
    sns.pairplot(trans_data, diag_kind='kde', plot_kws={'alpha': 0.2})  # scatter with hist
    plt.close()

    # Facet grid to explore multi-categories
    sns.catplot(x='day',  y='tip_pct', hue='time', col='smoker', kind='bar',
                data=tips[tips.tip_pct < 1])  # filter data; for many categories
    plt.close()
    sns.catplot(x='day', y='tip_pct', row='time', col='smoker', kind='bar',
                data=tips[tips.tip_pct < 1])  # from hue to a new row
    plt.close()
    sns.catplot(x='tip_pct', y='day', kind='box', data=tips[tips.tip_pct < 0.5])  # another type of fig
    plt.close()


if __name__ == '__main__':
    main()