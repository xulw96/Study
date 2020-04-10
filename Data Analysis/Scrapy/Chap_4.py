def main():
    import requests
    from collections import namedtuple
    from bs4 import BeautifulSoup

    # base class for web content and website
    class Content:
        '''Common base class for all pages'''
        def __init__(self, url, title, body):
            self.url = url
            self.title = title
            self.body = body
        def print(self):
            '''flexible printing function controls output'''
            print('URL:{}'.format(self.url))
            print('Title:{}'.format(self.title))
            print('Body:{}'.format(self.body))
    class Website:
        def __init__(self, name, url, titleTag, bodyTag):
            self.name = name
            self.url = url
            self.titleTag = titleTag
            self.bodyTag = bodyTag

    # scrape NYTimes and Brookings
    def getPage(url):
        req = requests.get(url)
        return BeautifulSoup(req.text, 'html.parser')
    def scrapeNYTimes(url):
        bs = getPage(url)
        title = bs.find('h1').text
        lines = bs.find_all('p', {'class': 'story-content'})
        body = '\n'.join([line.text for line in lines])
        return Content(url, title, body)
    def scrapeBrookings(url):
        bs = getPage(url)
        title = bs.find('h1').text
        body = bs.find('div', {'class', 'post-body'}).text
        return Content(url, title, body)

    url = 'https://www.brookings.edu/blog/future-development/2018/01/26/' \
          'delivering-inclusive-urban-access-3-uncomfortable-truths/'
    content = scrapeBrookings(url)
    content.print()

    url = 'https://www.nytims.com/2018/01/25/opinion/sunday/silicon-valley-immortality.html'
    content = scrapeNYTimes()
    content.print()

    # deal with layouts
    class Crawler:
        def getPage(self, url):
            try:
                req = requests.get(url)
            except requests.exceptions.RequestException:
                return None
            return BeautifulSoup(req.text, 'html.parser')
        def safeGet(self, page, selector):
            '''Utility function used to get a content string from a BeautifulSoup
            object and a selector. Returns an empty string if no object is found
            for that selector'''
            selected = page.select(selector)  # 'select' works for CSS, 'find' does not
            if selected is not None and len(selected) > 0:
                return '\n'.join([element.get_text() for element in selected])
            return ''
        def parse(self, site, url):
            '''Extract content from a given url'''
            bs = self.getPage(url)
            if bs is not None:
                title = self.safeGet(bs, site.titleTag)
                body = self.safeGet(bs, site.bodyTag)
                if title != '' and body != '':
                    content = Content(url, title, body)
                    content.print()

    crawler = Crawler()
    SiteData = [
        ['O\'Reilly Media', 'http://oreilly.com', 'h1', 'section#product-description'],
        ['Reuters', 'http://reuters.com', 'h1', 'div.StandardArticleBody_body_1gnLA'],
        ['Brookings', 'http://www.brookings.edu', 'h1', 'div.post-body'],
        ['New York Times', 'http://nytimes.com', 'h1', 'p.story-content']]
    websites = []
    for row in SiteData:
        websites.append(Website(row[0], row[1], row[2], row[3]))
    crawler.parse(websites[0], 'http://shop.oreilly.com/produc/0636920028154.do')
    crawler.parse(websites[1], 'http://www.reuters.com/article/us-usa-epa-pruitt-idUSKBN19W2D0')
    crawler.parse(websites[2], 'https://www.brookings.edu/blog/techtank/2016/03/01/idea-to-retire-old-methods-of-policy-education/')
    crawler.parse(websites[3], 'https://www.nytimes.com/2018/01/28/business/energy-environment/oil-boom.html')

    # crawling through searching. appending '?= ***'
    class Content:
        '''common base class for all articles/pages'''
        def __init__(self, topic, url, title, body):
            self.topic = topic
            self.title = title
            self.body = body
            self.url = url
        def print(self):
            '''Flexible printing function controls output'''
            print('new article found for topic:{}'.format(self.topic))
            print('title:{}'.format(self.title))
            print('body:{}'.format(self.body))
            print('url:{}'.format(self.url))
    class Website:
        '''contains information about website structure'''
        def __init__(self, name, url, search_url, result_listing,
                     result_url, absolute_url, title_tag, body_tag):
            self.name = name
            self.url = url
            self.search_url = search_url
            self.result_listing = result_listing
            self.result_url = result_url
            self.absolute_url = absolute_url
            self.title_tag = title_tag
            self.body_tag = body_tag
    class Crawler:
        def get_page(self, url):
            try:
                req = requests.get(url)
            except requests.exceptions.RequestException:
                return None
            else:
                return BeautifulSoup(req.text, 'html.parser')
        def safe_get(self, page_obj, selector):
            obj = page_obj.select(selector)
            if obj is not None and len(obj) > 0:
                return obj[0].get_text()  # 'get_text()' for an html object
            return ''
        def search(self, topic, site):
            '''Searches a given website for a given topic and records all pages found'''
            bs = self.get_page(site.search_url + topic)
            search_results = bs.select(site.result_listing)
            for result in search_results:
                url = result.select(site.result_url)[0].attrs['href']
                # check whether relative or absolute URL
                if site.absolute_url:
                    bs = self.get_page(url)
                else:
                    bs = self.get_page(site.url + url)
                if bs is None:
                    print('something was wrong with that page. Skipping')
                    return
                title = self.safe_get(bs, site.title_tag)
                body = self.safe_get(bs, site.body_tag)
                if title != '' and body != '':
                    content = Content(topic, title, body, url)
                    content.print()

    crawler = Crawler()
    site_data = [
        ['O\'Reilly Media', 'http://oreilly.com', 'h1', 'section#product-description'],
        ['Reuters', 'http://reuters.com', 'h1', 'div.StandardArticleBody_body_1gnLA'],
        ['Brookings', 'http://www.brookings.edu', 'h1', 'div.post-body']]
    sites = []
    for row in site_data:
        sites.append(Website(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
    topics = ['python', 'data science']
    for topic in topics:
        print('getting into about:', topic)
        for target in sites:
            crawler.search(topic, target)

    # scrape by Rex on links
    class Website:
        def __init__(self, name, url, target_pattern, absolute_url, title_tag, body_tag):
            self.name = name
            self.url = url
            self.target_pattern = target_pattern
            self.absolute_url = absolute_url
            self.title_tag = title_tag
            self.body_tag = body_tag
    class Content:
        def __init__(self, url, title, body):
            self.url = url
            self.title = title
            self.body = body
        def print(self):
            print('url: {}'.format(self.url))
            print('title: {}'.format(self.title))
            print('body: {}'.format(self.body))
    import re
    class Crawler:
        def __init__(self, site):
            self.site = site
            self.visited = []  # class private attribute
        def get_page(self, url):
            try:
                req = requests.get(url)
            except requests.exceptions.RequestException:
                return None
            else:
                return BeautifulSoup(req.text, 'html.parser')
        def safe_get(self, page_obj, selector):
            selected = page_obj.select(selector)
            if selected is not None and len(selected) > 0:
                return '\n'.join(element.get_text for element in selected)
            return ''
        def parse(self, url):
            bs = self.get_page(url)
            if bs is not None:
                title = self.safe_get(bs, self.site.title_tag)
                body = self.safe_get(bs, self.site.body_tag)
                if title != '' and body != '':
                    content = Content(url, title, body)
                    content.print()
        def crawl(self):
            '''get pages from website home page'''
            bs = self.get_page(self.site.url)
            target_pages = bs.find_all('a', href=re.compile(self.site.target_pattern))
            for target_page in target_pages:
                target_page = target_page.attrs['href']
                if target_page not in self.visited:
                    self.visited.append(target_page)
                    if not self.site.absolute_url:
                        target_page = '{}{}'.format(self.site.url, target_page)
                    self.parse(target_page)
    reuters = Website('Reuters', 'https://www.reuters.com', '^(/article/)', False,'h1',
                      'div.StandardArticleBody_body_1gnLaA')
    crawler = Crawler(reuters)
    crawler.crawl()

    class Product(Website):  # a subclass to extend
        def __init__(self, name, url, title_tag, product_number, price):
            super().__init__(self, name, url, title_tag)
            self.product_number = product_number
            self.price_tag = price
    class Article(Website):
        def __init__(self, name, url, title_tag, body_tag, date_tag):
            super().__init__(self, name, url, title_tag)
            self.body_tag = body_tag
            self.date_tag = date_tag

if __name__ == '__main__':
    main()