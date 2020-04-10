def main():
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    from urllib.error import HTTPError, URLError

    # a task-specific scrapper
    try:
        html = urlopen('http://pythonscraping.com/pages/page1.html')
    except HTTPError as e:
        print(e)
    except URLError as e:
        pring('The server can\'t be found')
    else:
        bs = BeautifulSoup(html.read(), 'html.parser')  # using 'html.read()' will be reading text strings
        try:
            BadContent = bs.h1
        except AttributeError as e:
            print('Tag was not found')
        else:
            if BadContent == None:
                print('Tag was not found')
            else:
                print(BadContent)
    # a functino for scrapping
    def getTitle(url):
        try:
            html = urlopen(url)
        except HTTPError:
            return None
        try:
            bs = BeautifulSoup(html.read(), 'html.parser')
            title = bs.body.h1
        except AttributeError:
            return None
        return title
    title = getTitle('http://pythonscraping.com/pages/page1.html')
    if title == None:
        print('Title cound not be found')
    else:
        print(title)



if __name__ == '__main__':
    main()