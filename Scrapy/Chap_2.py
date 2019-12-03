def main():
    from urllib.request import  urlopen
    from bs4 import BeautifulSoup

    html = urlopen('http://www.pythonscraping.com/pages/page1.html')
    bs = BeautifulSoup(html.read(), 'html.parser')
    nameList = bs.findAll('span', {'class': 'green'})
    for name in nameList:
        print(name.get_text())

    html = urlopen('http://www.pythonscraping.com/pages/page3.html')
    bs = BeautifulSoup(html, 'html.parser')
    children = bs.find('table', {'id': 'giftList'}).children()  # differentiate between children and descendants; also next_siblings; parents
    for child in children:
        print(child)

    import re
    images = bs.find_all('img', {'src': re.compile('\.\.\./img/gifts/img.*\.jpg')})
    for image in images:
        print(image)



if __name__ == '__main__':
    main()