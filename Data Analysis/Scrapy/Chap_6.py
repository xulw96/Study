def main():
    from urllib.request import urlretrieve
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    import os

    # retrieve everthing found! Do not run this script!
    download_directory = 'download'
    base_url = 'http://pythonscraping.com'

    def get_absolute_url(base_url, source):
        if source.startswith('http://www.'):
            url = 'http://{}'.format(source[11:])
        elif source.startswith('http://'):
            url = source
        elif source.startswith('www.'):
            url = 'http://{}'.format(source)
        else:
            url = '{}/{}'.format(base_url, source)
        if base_url not in url:
            return None
        return url

    def get_download_path(base_url, absolute_url, download_directory):
        path = absolute_url.replace('www', '')
        path = path.replace(base_url, '')
        path = download_directory + path
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return path

    def retrive_data():
        html = urlopen('http://www.pythonscraping.com')
        bs = BeautifulSoup(html, 'html.parser')
        download_list = bs.find_all(src=True)
        for download in download_list:
            file_url = get_absolute_url(base_url, download['src'])
            if file_url is not None:
                print(file_url)
                urlretrieve(file_url, get_download_path(base_url, file_url, download_directory))

    # write into .csv file
    import csv
    def write_csv():
        html = urlopen('http://en.wikipedia.org/wiki/Comparison_of_text_editors')
        bs = BeautifulSoup(html, 'html.parser')
        table = bs.find_all('table', {'class': 'wikitable'})[0]
        rows = table.find_all('tr')
        with open('editors.csv', 'w+') as csv_file:
            writer = csv.writer(csv_file)
            for row in rows:
                csv_row = []
                for cell in row.find_all(['td', 'th']):
                    csv_row.append(cell.get_text())
                    writer.writerow(csv_row)

    # MySQL with Python
    import pymysql
    '''conn = pymysql.connect(host='localhost', user='root', passwd='0618', db='MySQL')
    cur = conn.cursor()
    cur.execute('USE scraping')
    cur.execute('SELECT * FROM pages WHERE id=1')
    print(cur.fetchone())  # 'fetchone obtain result from last query executed
    cur.close()
    conn.close()'''

    import random
    import re
    import datetime
    '''conn = pymysql.connect(host='localhost', user='root', passwd='0618', db='MySQL', charset='utf8')  # handle in utf8
    cur = conn.cursor()
    cur.execute('USE scraping')
    random.seed(datetime.datetime.now())'''
    def store(title, content):
        cur.execute('INSERT INTO pages (title, content) VALUES ("%s", "%s"), (title, content)')
        cur.commit()  # operate through connection
    def get_links(article_url):
        html = urlopen('http://en.wikipedia.org' + article_url)
        bs = BeautifulSoup(html, 'html.parser')
        title = bs.find('h1').get_text()
        content = bs.find('div', {'id': 'mw-content-text'}).find('p').get_text()
        store(title, content)
        return bs.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))

    '''links = get_links('/wiki/Kevin_Bacon')
    try:
        while len(links) > 0:
            new_article = links[random.randint(0, len(links) - 1)].attrs['href']  # a random hyperlink from the results
            print(new_article)
            links = get_links(new_article)  # use it to search again
    finally:  # regardless of how the execution ends, execute it!
        cur.close()
        conn.close()'''

    def insert_page_if_not_exists(url):
        cur.execute('SELECT * FROM pages WHERE url = %s', (url))
        if cur.rowcount == 0:
            cur.execute('INSERT INTO pages (url) VALUES (%s)', (url))
            conn.commit()
            return cur.lastrowid
        else:
            return cur.fetchone()[0]  # output the selected result
    def load_pages():
        cur.execute('SELECT * FROM pages')
        pages = [row[1] for row in cur.fetchall()]
        return pages
    def insert_link(from_page_id, to_page_id):
        cur.execute('SELECT * FROM links WHERE from_page_id = %s AND to_page_id = %s', (int(from_page_id),
                                                                                        int(to_page_id)))
        if cur.rowcount == 0:
            cur.execute('INSERT INTO links (from_page_id, to_page_id) VALUES (%s, %s)', (int(from_page_id),
                                                                                         int(to_page_id)))
            conn.commit()
    def get_links(page_url, recursion_level, pages):
        if recursion_level > 4:
            return
        page_id = insert_page_if_not_exists(page_url)
        html = urlopen('http://en.wikipedia.org{}'.format(page_url))
        bs = BeautifulSoup(html, 'html.parser')
        links = bs.find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))
        links = [link.attrs['href'] for link in links]
        for link in links:
            insert_link(page_id, insert_page_if_not_exists(link))
            if link not in pages:
                pages.append(link)
                get_links(link, recursion_level + 1, pages)  # enter into next recursion
    from random import shuffle
    '''conn = pymysql.connect(host='localhost', user='root', passwd='0618', db='MySQL', charset='utf8')
    cur = conn.cursor()
    cur.execute('USE wikipedia')
    get_links('/wiki/Kevin_Bacon', 0, load_pages())
    cur.close()
    conn.close()'''

    # send an Email
    import smtplib
    from email.mime.text import MIMEText

    mail_host = 'smtp.163.com'
    mail_user = 'endevise'
    mail_pass = '5891645'
    sender = 'endevise@163.com'
    receivers = ['endevise@163.com']

    def send_mail(subject, body):
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'endevise@163.com'
        msg['To'] = 'endevise@163.com'
        try:
            smtp_obj = smtplib.SMTP()
            smtp_obj.connect(mail_host, 25)
            smtp_obj.login(mail_user, mail_pass)
            smtp_obj.sendmail(sender, receivers, msg.as_string())
            print('success: send email')
        except smtplib.SMTPException:
            print('error: fail to send email')
        finally:
            smtp_obj.quit()



if __name__ == '__main__':
    main()
