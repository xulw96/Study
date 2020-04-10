def main():
    import scrapy

    # scrape a single page
    class ArticleSpider(scrapy.Spiders):
        name = 'article'
        def start_requests(self):
            urls = [
                'http://en.wikipedia.org/wiki/Python_%28programming_language%29',
                'https://en.wikipedia.org/wiki/Functional_programming',
                'https://en.wikipedia.org/wiki/Monty_python']
            return [scrapy.Request(url=url, callback=self.parse) for url in urls]  # self-defined callback method
        def parse(self, response):  # manage the response from the website
            url = response.url
            title = response.css('h1::text').extract_first()
            print('url is: {}'.format(url))
            print('title is: {}'.format(title))

    from scrapy.linkextractors import LinkExtractor
    from scrapy.spiders import CrawlSpider, Rule

    # scrape multiple pages with Rule
    class ArticleSpider(CrawlSpider):
        name = 'articles'
        allowed_domains = ['wikipedia.org']
        start_urls = ['https://en.wikipedia.org/wiki/Benevolent_dictator_for_life']
        rules = [Rule(LinkExtractor(allow=r'.*'), callback='parse_items', follow=True)]
        def parse_items(self, response):
            url = response.url
            title = response.css('h1::text').extract_first()  # ignore the childe tag within text
            text = response.xpath('//div[@id="mw-content-text"]//text()').extract()  # include child tag within text
            last_updated = response.css('li#footer-info-lastmod::text').extract_first()
            last_updated = last_updated.replace('This page was last edited on ', '')
            print('url is: {}'.format(url))
            print('title is: {}'.format(title))
            print('text is: {}'.format(text))
            print('last updated is: {}'.format(last_updated))

    # manage different kinds of response
    class ArticleSpider(CrawlSpider):
        name = 'articles'
        allowed_domains = ['wikipedia.org']
        start_urls = ['https://en.wikipedia.org/wiki/Benevolent_dictator_for_life']
        rules = [Rule(LinkExtractor(allow='^(/wiki/)((?!:).)*$'), callback='parse_items',
                      follow=True, cb_kwargs={'is_article': True}),  # this dict can be passed as argument into callback function
                 Rule(LinkExtractor(allow='.*'), callback='parse_items', cb_kwargs={'is_article': False})]
        def parse_items(self, response, is_article):
            print(response.url)
            title = response.css('h1::text').extract_first()
            if is_article:
                url = response.url
                text = response.xpath('//div[@id="mw-content-text"]//text()').extract()
                last_updated = response.css('li#footer-info-lastmod::text').extract_first()
                last_updated = last_updated.replace('This page was last edited on ', '')
                print('title is: {}'.format(title))
                print('url is: {}'.format(url))
                print('text is: {}'.format(text))
                print(last_updated)
            else:
                print('this is not an article: {}'.format(title))
    class Article(scrapy.Item):  # a subclass for response managing and processing
        url = scrapy.Field()
        title = scrapy.Field()
        text = scrapy.Field()
        last_updated = scrapy.Field()
    class ArticleSpider(CrawlSpider):
        name = 'articleItems'
        allowed_domains = ['wikipedia.org']
        start_urls = ['https://en.wikipedia.org/wiki/Benevolent_dictator_for_life']
        rules = [Rule(LinkExtractor(allow='(/wiki/)((?!:).)*$'), callback='parse_items', follow=True)]  # recursive scrapping
        def parse_items(self, response):
            article = Article()
            article['url'] = response.url
            article['title'] = response.css('h1::text').extract_first()
            article['text'] = response.xpath('//div[@id="mw-content-text"]//text()').extract()
            last_updated = response.css('li#footer-info-lastmod::text').extract_first()
            article['last_updated'] = last_updated.replace('This page was last edited on', '')
            return article


if __name__ == '__main__':
    main()