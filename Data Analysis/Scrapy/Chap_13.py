def main():
    from PIL import Image
    import pytesseract
    from pytesseract import Output

    # pytesseract
    image = Image.open('./test.png')
    print(pytesseract.image_to_string(image))
    box_file = pytesseract.image_to_boxes(image)
    print(box_file)
    all_output = pytesseract.image_to_data(image)
    print(all_output)
    output_dict = pytesseract.image_to_data(image, output_type=Output.DICT)
    output_bytes = pytesseract.image_to_data(image, output_type=Output.BYTES)
    print(output_bytes)
    print(output_dict)

    # threshold filter
    def clean_file(file, new_file):
        image = Image.open(file)
        image = image.point(lambda x: 0 if x < 143 else 255) # set a threshold value for image
        image.save(new_file)
        return image

    # automatically adjust
    import numpy as np
    def clean_file(file_path, threshold):
        image = Image.open(file_path)
        image = image.point(lambda x: 0 if x < threshold else 255)
        return image
    def get_confidence(image):
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        text = data['text']
        confidences = []
        numChars = []
        for i in range(len(text)):
            if int(data['conf'][i]) > -1:
                confidences.append(data['conf'][i])
                numChars.append(len(text[i]))
        return np.average(confidences, weights=numChars), sum(numChars)
    file_path = './test.png'
    start, step, end = 80, 5, 200
    '''for threshold in range(start, end, step):
        image = clean_file(file_path, threshold)
        scores = get_confidence(image)
        print('threshold: ' + str(threshold) + ', confidence: ' +
              str(scores[0]) + ' numChars ' + str(scores[1]))'''

    # scrape text from images
    import time
    from urllib.request import urlretrieve
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    import subprocess
    def get_image_text(url):
        urlretrieve(url, 'page.jpg')
        p = subprocess.Popen(['tesseract', 'page.jpg', 'page'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        f = open('page.txt', 'r')
        print(f.read())
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    '''driver = webdriver.Chrome(executable_path='./chromedriver', options=chrome_options)
    driver.get('https://www.amazon.com/Death-Ivan-Ilyich-Nikolayevich-Tolstoy/dp/1427027277')
    time.sleep(2)
    driver.find_element_by_id('imgBlkFront').click()  # click the preview button
    imageList = []
    time.sleep(5)  # wait for the page to load
    while 'pointer' in driver.find_element_by_id('sitbReaderRightPageTurner').get_attribute('style'):
        driver.find_element_by_id('sitbReaderRightPageTurner').click()  # turn through pages
        time.sleep(2)
        pages = driver.find_elements_by_xpath('//div[@class=\'pageImage\']/div/img')
        if not len(pages):
            print('No pages found')
        for page in pages:
            image = page.get_attribute('src')
            print('Found image: {}'.format(image))
            if image not in imageList:
                imageList.append(image)
                get_image_text(image)
    driver.quit()'''

    # CAPTCHA
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    from PIL import ImageOps
    import requests
    def clean_image(image_path):
        image = Image.open(image_path)
        image = image.point(lambda x: 0 if x < 143 else 255)
        border_image = ImageOps.expand(image, border=20, fill='white')
        border_image.save(image_path)
    html = urlopen('http://www.pythonscraping.com/humans-only')
    bs = BeautifulSoup(html, 'html.parser')
    image_location = bs.find('img', {'title': 'Image CAPTCHA'})['src']
    form_build_id = bs.find('input', {'name': 'form_build_id'})['value']
    captcha_sid = bs.find('input', {'name': 'captcha_sid'})['value']
    captcha_token = bs.find('input', {'name': 'captcha_token'})['value']
    captcha_url = 'http://pythonscraping.com' + image_location

    urlretrieve(captcha_url, 'captcha.jpg')
    clean_image('captcha.jpg')
    p = subprocess.Popen(['tesseract', 'captcha.jpg', 'captcha'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    f = open('captcha.txt', 'r')
    captcha_response = f.read().replace(' ', '').replace('\n', '')
    print('Captcha solution attempt: ' + captcha_response)

    if len(captcha_response) == 5:
        params = {'captcha_token': captcha_token, 'captcha_sid': captcha_sid,
                  'form_id': 'comment_node_page_form', 'form_build_id': form_build_id,
                  'captcha_response': captcha_response, 'name': 'Ryan Mitchell',
                  'subject': 'I come to seek the Grail',
                  'comment_body[und][0][value]': '...and I am definitely not a bot'}
        r = requests.post('http://www.pythonscraping.com/comment/reply/10', data=params)
        response_obj = BeautifulSoup(r.text, 'html.parser')
        if response_obj.find('div', {'class': 'messages'}) is not None:
            print(response_obj.find('div', {'class': 'messages'}).get_text())
    else:
        print('There was a problem reading the CAPTCHA correctly')



if __name__ == '__main__':
    main()