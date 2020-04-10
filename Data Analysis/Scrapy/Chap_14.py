def main():
    # request headers
    import requests
    from bs4 import BeautifulSoup
    session = requests.Session()
    headers = {'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5)'
                              'AppleWebKit 537.36 (KHTML, like Gecko) Chrome',
               'Accept': 'text/html,application/xhtml+xml,application/xml;'
                         'q=0.9,image/webp,*/*;q=0.8'}
    '''url = 'https://www.watismybrowser.com/developers/what-http-headers-is-my-browser-sending'
    req = session.get(url, headers=headers)  # change the header with the request'''

    # handle cookies
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(executable_path='./chromedriver', options=chrome_options)
    '''driver.get('http://pythonscraping.com')
    driver.implicitly_wait(1)
    cookies = driver.get_cookies()
    print(cookies)'''

    '''driver2 = webdriver.Chrome(executable_path='./chromedriver', options=chrome_options)
    driver2.get('http://pythonscraping.com')  # have to load before add; for selenium to know which domain cookie belongs to
    driver2.delete_all_cookies()
    for cookie in cookies:
        driver2.add_cookie(cookie)
    driver2.get('http://pythonscraping.com')
    driver2.implicitly_wait(1)
    print(driver2.get_cookies())'''

    # hidden fields
    from selenium.webdriver.remote.webelement import WebElement
    driver.get('http://pythonscraping.com/pages/itsatrap.html')
    links = driver.find_elements_by_tag_name('a')
    for link in links:
        if not link.is_displayed():  # check whether this field can be shown in browser
            print('The link {} is a trap'.format(link.get_attribute('href')))
    fields = driver.find_elements_by_tag_name('input')
    for field in fields:
        if not field.is_displayed():
            print('Do not change value of {}'.format(field.get_attribute('name')))  # a hidden form


if __name__ == '__main__':
    main()