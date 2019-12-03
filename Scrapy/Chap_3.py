def main():
    from urllib.request import urlopen, urlparse
    from bs4 import BeautifulSoup
    import datetime
    import random
    import re
    from urllib.error import HTTPError


    random.seed(datetime.datetime.now())
    def getLink(url):
        html = urlopen('http://en.widipedia.org{}'.format(url))
        bs = BeautifulSoup(html, 'html.parser')
        return bs.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))

    try:
        links = getLink('/wiki/Kevin_Bacon')
    except HTTPError:
        pass
    else:
        while len(links) > 0:
            newArticle = links[random.randint(0, len(links)-1)].attrs['href']
            print(newArticle)
            links = getLink(newArticle)


    pages = set()  # to support unique elements
    def getLink(url):
        global pages
        html = urlopen('http://en.wikipedia.org{}'.format(url))
        bs = BeautifulSoup(html, 'html.parser')
        try:
            print(bs.h1.get_text())
            print(bs.find(id='mw-content-text').find_all('p')[0])
            print(bs.find(id='ca-edit').find('span').find('a').attrs['href'])
        except AttributeError:
            print('This page is missing something! Continuing.')
        for link in bs.find_all('a', href=re.compile('^(/wiki/)')):
            if 'href' in link.attrs:
                if link.attrs['href'] not in pages:  # make sure it's unique
                    newPage = link.attrs['href']
                    print('_' * 20)
                    print(newPage)
                    pages.add(newPage)
                    getLink(newPage)  # store and call another round
    try:
        getLink('')
    except HTTPError:
        pass

    class getLinks:
        def __init__(self, bs, url):
            self.bs = bs
            self.url = url
        def internal(self, bs, url):  # links start with a "/"
            includeUrl = '{}://{}'.format(urlparse(url).scheme, urlparse(url).netloc)
            internalLinks = []
            for link in bs.find_all('a', href=re.compile('^(/|.*'+includeUrl+')')):
                if link.attrs['href'] is not None:
                    if link.attrs['href'] not in internalLinks:
                        if link.attrs['href'].startswidth('/'):
                            internalLinks.append(includeUrl+link.attrs['href'])
                        else:
                            internalLinks.append(link.attrs['href'])
            return includeUrl
        def external(self, bs, url):  # links start with 'http', without current URL
            externalLinks = []
            for link in bs.find_all('a', href=re.compile('^(http|www)((?!'+url+').)*$')):
                if link.attrs['href'] is not None:
                    if link.attrs['href'] not in externalLinks:
                        externalLinks.append(link.attrs['href'])
            return externalLinks
        def randomExternal(self, page):
            html = urlopen(page)
            bs = BeautifulSoup(html, 'html.parser')
            externalLinks = self.external(bs, urlparse(page).netloc)
            if len(externalLinks) == 0:
                print('No external links, looking around the site for one')
                domain = '{}://{}'.format(urlparse(page).scheme, urlparse(page).netloc)
                internalLinks = self.internal(bs, domain)
                return self.randomExternal(internalLinks[random.randint(0, len(internalLinks) - 1)])
            else:
                return externalLinks[random.randint(0, len(externalLinks) - 1)]
        def onlyExternal(self, site):
            externalLink = self.randomExternal(site)
            print('Random external link is {}'.format(externalLink))
            self.onlyExternal(externalLink)
        allExtLinks = set()
        allIntLinks = set()
        def allExternal(self, site):
            html = urlopen(site)
            domain = '{}://{}'.format(urlparse(site).scheme, urlparse(site).netloc)
            bs = BeautifulSoup(html, 'html.parser')
            internalLinks = self.internal(bs, domain)
            externalLinks = self.external(bs, domain)

            for link in externalLinks:
                if link not in self.allExtLinks:
                    self.allExtLinks.add(link)
                    print(link)
            for link in internalLinks:
                if link not in self.allExtLinks:
                    self.allIntLinks.add(link)
                    self.allExternal(link)


if __name__ == '__main__':
    main()
