def main():
    import requests
    params = {'firstname': 'Ryan', 'lastname': 'Mitchell'}
    r = requests.post('http://pythonscraping.com/pages/processing.php', data=params)
    print(r.text)

    params = {'email_addr': 'ryan.e.mitchell@gmail.com'}
    r = requests.post('http://post.oreilly.com/client/o/oreilly/forms/quicksignup.cgi', data=params)
    print(r.text)

    '''files = {'uploadFile': open('files/python.png', 'rb')}  # dict mapping a Python File object
    r = requests.post('http://pythonscraping.com/pages/processing2.php', files=files)
    print(r.text)'''

    params = {'username': 'Ryan', 'password': 'password'}
    r = requests.post('http://pythonscraping.com/pages/cookies/welcom.php', data=params)
    print(r.cookies.get_dict())  # retrieve cookies from the result, as authentification
    r = requests.get('http://pythonscraping.com/pages/cookies/profile.php', cookies=r.cookies)
    print(r.text)

    session = requests.Session()
    params = {'username': 'username', 'password': 'password'}
    s = session.post('http://pythonscraping.com/pages/cookies/welcom.php', params)
    s = session.get('http://pythonscraping.com/pages/cookies/profile.php')  # cookies are automactically included

    from requests.auth import HTTPBasicAuth
    auth = HTTPBasicAuth('ryan', 'password')
    r = requests.post('http://pythonscraping.com/pages/auth/login.php', auth=auth)
    print(r.text)






if __name__ == '__main__':
    main()