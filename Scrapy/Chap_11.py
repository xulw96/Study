def main():
    from selenium import webdriver
    import time
    from selenium.webdriver.chrome.options import Options

    # webdriver to scrap JS
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # set to use headless chrome
    path = '/Users/aa/PycharmProjects/CS/Scrapy/chromedriver'
    '''driver = webdriver.Chrome(executable_path=path, options=chrome_options)
    driver.get('http://pythonscraping.com/pages/javascript/ajaxDemo.html')
    time.sleep(3)
    print(driver.find_element_by_id('content').text)
    driver.close()
    page_source = driver.page_source  # can be turned to BeautifulSoup'''

    # implicit wait
    from selenium.webdriver.common.by import By  # a locator is greater than selector
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    '''driver = webdriver.Chrome(executable_path=path, options=chrome_options)
    driver.get('http://pythonscraping.com/pages/javascript/ajaxDemo.html')
    try:
        element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'loadedButton')))  # condituioned waiting
    finally:
        print(driver.find_element_by_id('content').text)
        driver.close()'''

    # handle redirects
    from selenium.webdriver.remote.webelement import WebElement  # check the DOM
    from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
    def wait_for_load(driver):
        element = driver.find_element_by_tag_name('html')
        count = 0
        while True:
            count += 1
            if count > 20:
                print('time out and returning')
                return
            time.sleep(0.5)
            try:
                element = driver.find_element_by_tag_name('html')
            except StaleElementReferenceException:
                return
    driver = webdriver.Chrome(executable_path=path, options=chrome_options)
    driver.get('http://pythonscraping.com/pages/javascript/redirectDemo1.html')
    wait_for_load(driver)
    print(driver.page_source)

    driver.get('http://pythonscraping.com/pages/javascript/redirectDemo1.html')
    try:
        body_element = WebDriverWait(driver, 15).until(EC.presence_of_element_located
                                                       (By.XPATH, '//body[contains(text(),"This is the page you are looking for!")]'))
        print(body_element.text)
    except TimeoutException:
        print('did not find the element')


if __name__ == '__main__':
    main()