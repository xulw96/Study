def main():
    # Tor
    import socks
    import socket
    from urllib.request import urlopen
    socks.set_default_proxy(socks.SOCKS5, 'localhost', 9150)
    socket.socket = socks.socksocket
    print(urlopen('http://icanhazip.com').read())




if __name__ == '__main__':
    main()