import unittest

# unittest
class TestAddition(unittest.TestCase):
    def setUp(self):  # run before each test
        print('setting up the test')
    def tearDown(self):
        print('tearing down the test')
    def test(self):
        total = 2 + 2
        self.assertEqual(4, total)  # check whether it's four as output

# testing frontend
from urllib.request import urlopen
from bs4 import BeautifulSoup
'''class TestWiki(unittest.TestCase):  # two test in one class
    bs = None
    def setUpClass():  # run at the start of class
        url = 'http://en.wikipedia.org/wiki/Monty_Python'
        html = urlopen(url)
        TestWiki.bs = BeautifulSoup(html, 'html.parser')
    def test_title(self):
        page_title = TestWiki.bs.find('h1').get_text()
        self.assertEqual('Monty Python', page_title)
    def test_content(self):
        content = TestWiki.bs.find('div', {'id': 'mw-content-text'})
        self.assertIsNotNone(content)'''
# repeatetively test
import re
import random
from urllib.parse import unquote
class TestWikipedia(unittest.TestCase):
    def test_page(self):
        self.url = 'http://en.wikipedia.org/wiki/Monty_Python'
        for i in range(1, 10):  # test first 10 pages
            html = urlopen(self.url)
            self.bs = BeautifulSoup(html, 'html.parser')
            titles = self.title_match()
            self.assertEqual(titles[0], titles[1])
            self.assertTrue(self.content_exists())
            self.url = self.get_next_link()
        print('Done')
    def title_match(self):
        page_title = self.bs.find('h1').get_text()
        url_title = self.url[(self.url.index('/wiki/')+6):]
        url_title = url_title.replace('_', ' ')
        url_title = unquote(url_title)
        return [page_title.lower(), url_title.lower()]
    def content_exists(self):
        content = self.bs.find('div', {'id': 'mw-content-text'})
        if content is not None:
            return True
        return False
    def get_next_link(self):
        links = self.bs.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))
        random_link = random.SystemRandom().choice(links)  # make a random choice from links.
        return 'https://wikipedia.org{}'.format(random_link.attrs['href'])
def main():
    # selenium for test
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    path = '/Users/aa/PycharmProjects/CS/Scrapy/chromedriver'
    driver = webdriver.Chrome(executable_path=path, options=chrome_options)
    '''driver.get('http://en.wikipedia.org/wiki/Monty_Python')
    assert 'Monty Python' in driver.title  # use 'assert' statement to test
    driver.close()'''

    # actionchain
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver import ActionChains
    from selenium.webdriver.remote.webelement import WebElement
    driver.get('http://pythonscraping.com/pages/files/form.html')
    first_name_field = driver.find_element_by_name('firstname')
    last_name_field = driver.find_element_by_name('lastname')

    actions = ActionChains(driver).click(first_name_field).send_keys('Ryan').click(last_name_field).send_keys('Mitchell').send_keys(Keys.RETURN)
    actions.perform()
    print(driver.find_element_by_tag_name('body').text)
    driver.close()

    hrome_options = Options()
    chrome_options.add_argument('--headless')
    path = '/Users/aa/PycharmProjects/CS/Scrapy/chromedriver'
    driver = webdriver.Chrome(executable_path=path, options=chrome_options)
    driver.get('http://www.pythonscraping.com')
    driver.get_screenshot_as_file('pythonscraping.png')
    driver.close()

    hrome_options = Options()
    chrome_options.add_argument('--headless')
    path = '/Users/aa/PycharmProjects/CS/Scrapy/chromedriver'
    driver = webdriver.Chrome(executable_path=path, options=chrome_options)
    driver.get('http://pythonscraping.com/pages/javascript/draggableDemo.html')
    print(driver.find_element_by_id('message').text)
    element = driver.find_elements_by_id('draggable')
    target = driver.find_element_by_id('div2')
    actions = ActionChains(driver)
    actions.drag_and_drop(element, target).perform()
    print(driver.find_element_by_id('message').text)

if __name__ == '__main__':
    main()
'''if __name__ == '__main__':
    unittest.main()'''
