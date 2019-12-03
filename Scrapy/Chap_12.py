def main():
    # json
    import json
    from urllib.request import urlopen

    def get_country(ip):
        response = urlopen('http://freegeoip.net/json'+ip).read().decode('utf-8')
        response = json.load(response)
        return response.get('country_code')[0]  # get for the key; '[]' for the index
    print(get_country('50.78.253.58'))

    import random
    import re
    import datetime
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    def get_links(url):
        html = urlopen('http://en.wikipedia.org{}'.format(url))
        bs = BeautifulSoup(html, 'html.parser')
        return bs.find('div', {'id': 'bodyContent'}).find_all('a', href=re.compile('^(/wiki/)((?!:).)*$'))
    def get_ip(url):
        url = url.replace('/wiki/', '')
        history_url = 'http://en.wikipedia.org/w/index.php?title={}&action=history'.format(url)
        print('history url is: {}'.format(history_url))
        html = urlopen(history_url)
        bs = BeautifulSoup(html, 'html.parser')
        ip_address = bs.find_all('a', {'class': 'mw-anouserlink'})
        address_list = set()
        for ip in ip_address:
            address_list.add(ip.get_text())
        return address_list
    links = get_links('/wiki/python_(programming_language)')
    while (len(links) > 0):
        for link in links:
            print('-' * 20)
            history_ips = get_ip(link.attrs['href'])
            for history_ip in history_ips:
                print(history_ip)
        new_link = links[random.randint(0, len(links) - 1)].attrs['href']
        links = get_links(new_link)



if __name__ == '__main__':
    main()