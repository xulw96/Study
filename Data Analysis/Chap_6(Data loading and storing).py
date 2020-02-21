def main():
    # data reading and writing
    import pandas as pd
    import sys
    df = pd.read_csv('example.csv', header=None)
    df = pd.read_table('example.csv', sep='\s+')  # table need to specify the delimiter; 's+' is regex for all whitespace
    df = pd.read_csv('example.csv', names=['a', 'b', 'c', 'd', 'message'],
                     index_col=['message', 'a'])  # grant names to columns, specify column to be index; a hierachical index here
    df = pd.read_csv('example.csv', skiprows=[0, 2, 3])  # skip rows for reading
    sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
    df = pd.read_csv('example.csv', na_values=sentinels)  # specify values to be replaced NA; specify such condition for each column

    pd.options.display.max_rows = 10  # set for display only
    pd.read_csv('example.csv', nrows=5)  # set for reading
    chuncker =  pd.read_csv('example.csv', chunksize=1000)  # iterate over parts of file
    tot = pd.Series([])
    for piece in chuncker:
        tot = tot.add(piece['key'].value_counts(), fill_value=0)

    df.to_csv(sys.stdout, sep='|', na_rep='NULL')  # writing to the console, replace NaN
    df.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])  # control writing index and columns

    # csv file
    import csv
    with open('example.csv') as f:
        reader = csv.reader(f, delimiter='|')
        lines = list(reader)
        header, values = lines[0], lines[:1]  # unpack
        data_dict = {h: v for h, v in zip(header, values)}  # dict comprehension
    class my_dialect(csv.Dialect):  # define a specific reader
        lineterminator = '\n'
        delimiter = ';'
        quotechar = '"'
        quoting = csv.QUOTE_MINIMAL  # quote only fields with special character
    with open('example.csv') as f:
        reader = csv.reader(f, dialect=my_dialect)  # deploy my_dialect
        writer = csv.writer(f, dialect=my_dialect)
        writer.writerow(('1', '2', '3'))

    # JSON data
    import json
    result = json.loads(obj)  # transform json into python
    asjson = json.dump(result)  # backwards
    data = pd.read_json('example.json')
    asjson = data.to_json(orient='records')

    # XML and HTML
    html = 'https://www.fdic.gov/bank/individual/failed/banklist.html'
    tables = pd.read_html('html')
    failures = table[0]
    close_timestamps = pd.to_datetime(failures['Closing Date'])
    close_timestamps.dt.year.value_counts()

    from lxml import objectify
    path = 'performance_nmr.xml'
    parsed = objectify.parse(path)
    root = parsed.getroot()
    data = []
    skip_fields = ['PARENT_SEQ', 'INDICATOR_SEQ', 'DESIRED_CHANCE', 'DECIMAL_PLACES']
    for elt in root.INDICATOR:  # returns a generators for yielding each tag
        el_data = {}
        for child in elt.getchildren():
            if child.tag in skip_fields:
                continue
            el_data[child.tag] = child.pyval
        data.append(el_data)

    # excel
    writer = pd.ExcelWriter('example.xlsx')
    frame.to_excel(writer, 'sheet1')
    writer.save()
    frame.to_excel('example.xlsx')  # avoid ExcelWriter

    # Web API
    import requests
    url = 'https://api.github.com/repos/pandas-dev/pandas/issues'
    resp = requests.get(url)
    data = resp.json()
    data[0]['title']

    # SQL
    import sqlalchemy as sqla
    db = sqla.create_engine('mysql://root:0618@localhost/test', encoding='utf8')
    pd.read_sql('select * from test', db)


if __name__ == '__main__':
    main()