def main():
    # multithread
    import _thread
    import time
    def print_time(thread_name, delay, iterations):
        start = int(time.time())
        for i in range(0, iterations):
            time.sleep(delay)
            seconds_elapsed = str(int(time.time()) - start)
            print("{} {}".format(seconds_elapsed, thread_name))
    '''try:
        _thread.start_new_thread(print_time, ('Fizz', 3, 33))
        _thread.start_new_thread(print_time, ('Buzz', 5, 20))
        _thread.start_new_thread(print_time, ('Counter', 1, 100))
    except:
        print ('Error: unable to start thread')
    while 1:
        pass'''

    # multithread with scraping
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    import re
    import random
    def get_links(thread_name, bs):
        print('Getting links in {}'.format(thread_name))
        return bs.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))
    def scrape_article(thread_name, path):  # function for thread
        html = urlopen('')
        time.sleep(5)
        bs = BeautifulSoup(html, 'html.parser')
        title = bs.find('h1').get_text()
        print('Scraping {} in thread {}'.format(title, thread_name))
        links = get_links(thread_name, bs)
        if len(links) > 0:
            new_article = links[random.randint(0, len(links) - 1)].attrs['href']
            print(new_article)
            scrape_article(thread_name, new_article)  # recursive scraping
    '''try:
        _thread.start_new_thread(scrape_article, ('Thread 1', '/wiki/Kevin_Bacon',))
        _thread.start_new_thread(scrape_article, ('Thread 2', '/wiki/Monty_Python',))
    except:
        print('Error: unable to start thread')
    while 1:
        pass'''

    # queue for communicating between threads
    from queue import Queue
    import pymysql
    def storage(queue):
        conn = pymysql.connect(host='localhost', user='root', passwd='0618', db='mysql', charset='utf8')
        cur = conn.cursor()
        cur.execute('USE wiki_threads')
        while 1:
            if not queue.empty():  # obtain item from queue
                article = queue.get()
                cur.execute('SELECT * FROM pages WHERE path = %s', (article["path"]))
                if cur.rowcount == 0:
                    print('storing article {}'.format(article['title']))
                    cur.execute('INSERT INTO pages (title, path) VALUES (%s, %s)', (article['title'], article['path']))
                    conn.commit()
                else:
                    print('article already exists: {}'.format(article['title']))
    visited = []  # make sure sites being visited only once
    def get_links(thread_name, bs):
        print('getting links in {}'.format(thread_name))
        links = bs.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))
        return [link for link in links if link not in visited]
    def scrape_article(thread_name, path, queue):
        visited.append(path)
        html = urlopen('http://en.wikipedia.org{}'.format(path))
        time.sleep(5)
        bs = BeautifulSoup(html, 'html.parser')
        title = bs.find('h1').get_text()
        print('added {} for storage in thread {}'.format(title, thread_name))
        queue.put({'title': title, 'path': path})  # put into queue for communication
        links = get_links(thread_name, bs)
        if len(links) > 0:
            new_article = links[random.randint(0, len(links) - 1)].attrs['href']
            scrape_article(thread_name, new_article, queue)
    queue = Queue()  # create the queue
    '''try:
        _thread.start_new_thread(scrape_article, ('Thread 1', '/wiki/Kevin_Bacon', queue,))
        _thread.start_new_thread(scrape_article, ('THread 2', '/wiki/Monty_Python', queue,))
        _thread.start_new_thread(storage, (queue,))  # a thread to store into database
    except:
        print('Error: unable to start threads')
    while 1:
        pass'''

    # threading
    import threading
    def print_time(thread_name, delay, iterations):
        start = int(time.time())
        for i in range(0, iterations):
            time.sleep(delay)
            seconds_elapsed = str(int(time.time()) - start)
            print('{} {}'.format(seconds_elapsed, thread_name))
    '''threading.Thread(target=print_time, args=('Fizz', 3, 33)).start()
    threading.Thread(target=print_time, args=('Buzz', 5, 20)).start()
    threading.Thread(target=print_time, args=('Counter', 1, 100)).start()'''
    def cralwer(url):
        data = threading.local()  # grant it a private memory
        data.visited = []  # local thread data
    '''threading.Thread(target=crawer, args=('http://brookings.edu')).start()'''
    t = threading.Thread(target=cralwer)
    '''t.start()
    while True:
        time.sleep(1)
        if not threading.Thread.is_alive():
            t = threading.Thread(target=crawer)
            t.start()  # for restart'''
    class Crawler(threading.Thread):   # inherit
        def __init__(self):
            threading.Thread.__init__(self)
            self.done = False
        def is_done(self):  # check for thread finish
            return self.done
        def run(self):
            time.sleep(5)
            self.done = True
            raise Exception('Something bad happened')
    t = Crawler()
    '''t.start()
    while True:
        time.sleep(1)
        if t.is_done():
            print('Done')
            break
        if not t.is_alive():
            t = cralwer()
            t.start()'''

    # processing, each as independent program
    from multiprocessing import Process
    def print_time(thread_name, delay, iterations):
        start = int(time.time())
        for i in range(0, iterations):
            time.sleep(delay)
            seconds_elapsed = str(int(time.time()) - start)
            print(thread_name if thread_name else seconds_elapsed)
    processes = []
    processes.append(Process(target=print_time, args=('Counter', 1, 100)))
    processes.append(Process(target=print_time, args=('Fizz', 3, 33)))
    processes.append(Process(target=print_time, args=('Buzz', 5, 20)))
    for p in processes:
        p.start()
    for p in processes:
        p.join()  # needed if want to execute process after all complete'''

    # multiprocesses crawling
    import os
    visited = []
    def get_links(bs):
        print('getting links in {}'.format(os.getpid()))
        links = bs.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))
        return [link for link in links if link not in visited]
    def scrape_article(path):
        visited.append(path)
        html = urlopen('http://en.wikipedia.org{}'.format(path))
        time.sleep(5)
        bs = BeautifulSoup(html, 'html.parser')
        title = bs.find('h1').get_text()
        print('Scraping {} in process {}'.format(title, os.getpid()))
        links = get_links(bs)
        if len(links) > 0:
            new_article = links[random.randint(0, len(links) - 1)].attrs['href']
            print(new_article)
            scrape_article(new_article)
    '''processes = []
    processes.append(Process(target=scrape_article, args=('/wiki/Kevin_Bacon')))
    processes.append(Process(target=scrape_article, args=('/wiki/Monty_Python')))
    for p in processes:
        p.start()'''

    # communicating between multiprocess
    def task_delegator(task_queue, urls_queue):  # as an inter-communicator
        visited = ['/wiki/Kevin_Bacon', '/wiki/Monty_Python']
        task_queue.put('/wiki/Kevin_Bacon')
        task_queue.put('/wiki/Monty_Python')
        while 1:
            if not urls_queue.empty():
                links = [link for link in urls_queue.get() if link not in visited]
                for link in links:
                    task_queue.put(link)
    def get_links(bs):
        print('getting links in {}'.format(os.getpid()))
        links = bs.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))
        return [link for link in links if link not in visited]

    def scrape_article(task_queue, urls_queue):
        while 1:
            while task_queue.empty():
                time.sleep(0.1)
            path = task_queue.get()
            html = urlopen('http://en.wikipedia.org{}'.format(path))
            time.sleep(5)
            bs = BeautifulSoup(html, 'html.parser')
            title = bs.find('h1').get_text()
            print('Scraping {} in process {}'.format(title, os.getpid()))
            links = get_links(bs)
            urls_queue.put(links)  # send to delegator for processing
    processes = []
    task_queue = Queue()
    urls_queue = Queue()
    processes.append(Process(target=task_delegator, args=(task_queue, urls_queue,)))
    processes.append(Process(target=scrape_article, args=(task_queue, urls_queue,)))
    processes.append(Process(target=scrape_article, args=(task_queue, urls_queue,)))
    for p in processes:
        p.start()


if __name__ == '__main__':
    main()
