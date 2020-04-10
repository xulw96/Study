def main():
    # text spinner with a thread
    import threading
    import itertools
    import time
    import sys
    class Signal:  # used to terminate thread
        go = True
    def spin(msg, signal):  # run in a seperate thread
        write, flush = sys.stdout.write, sys.stdout.flush
        for char in itertools.cycle('|/-\\'):  # an infinite loop
            status = char + ' ' + msg
            write(msg)
            flush()
            write('\x08' * len(status))  # '/x08' for backspace character
            time.sleep(0.1)
            if not signal.go:
                break  # exit the loop
        write(' ' * len(status) + '\x08' * len(status))  # clear the line and move cursor back
    def slow_function():  # pretend to be some costly function
        time.sleep(3)  # block thread but release GIL
        return 42
    def supervisor():  # 2nd thread: display thread, run computation, kill thread
        signal = Signal()
        spinner = threading.Thread(target=spin, args=('thinking!', signal))
        print('spinner  object:', spinner)
        spinner.start()  # start this thread
        result = slow_function()
        signal.go = False  # terminate the for loop
        spinner.join()  # wait until thread finish
        return result
    print('Answer:', supervisor())  # start the function

    # text spinner with a coroutine
    import asyncio
    @asyncio.coroutine  # decorate coroutines
    def spin(msg):  # no need to input a shut-down argument
        write, flush = sys.stdout.write, sys.stdout.flush
        for char in itertools.cycle('|/-\\'):
            status = char + ' ' + msg
            write(status)
            flush()
            write('\x08' * len(status))
            try:
                yield from asyncio.sleep(0.1)  # not use time.sleep(0.1), in case event loop being blocked.
            except asyncio.CancelledError:  # the cancellation is requested
                break
        write(' ' * len(status) + '\x08' * len(status))
    @asyncio.coroutine
    def slow_function():
        yield from asyncio.sleep(3)  # let the event loop proceed when this coroutine pretend I/O
        return 42
    @asyncio.coroutine
    def supervisor():
        spinner = asyncio.ensure_future(spin('thinking!'))  # schedule the 'spin' coroutine to run
        print('spinner object:', spinner)
        result = yield from slow_function()
        spinner.cancel()  # cancel the coroutine
        return result
    loop = asyncio.get_event_loop()  # a reference to the event loop
    result = loop.run_until_complete(supervisor())
    loop.close()
    print('Answer:', result)
    # asynchronous download
    import aiohttp
    @asyncio.coroutine
    def get_flag(cc):
        url = '{}/{cc}/{cc}.gif'.format(BASE_URL, cc=cc.lower())
        resp = yield from aiohttp.request("GET", url)  # using 'yield from', this operation won't stop the line. Only suspend this delegating coroutine
        image = yield from resp.read()
        return image
    @asyncio.coroutine
    def download_one(cc):
        image = yield from get_flag(cc)
        show(cc)
        save_flag(image, cc.lower() + '.gif')
        return cc
    def download_many(cc_list):
        loop = asyncio.get_event_loop()  # event_loop is not directly implemented, but only obtain a reference to it
        to_do = [download_one(cc) for cc in sorted(cc_list)]
        wait_coro = asyncio.wait(to_do)  # not a blocking function, only a coroutine waiting for all coroutines to be finished
        res, _ = loop.run_until_complete(wait_coro)  # here the script will be blocked, and *** drive the coroutine ***, the second item hold unfinished coroutines
        loop.close()
        return len(res)
    # Error handling in asyncio. Every function should be not blocking. We use only one thread and can't bear that
    import collections
    from aiohttp import web
    import tqdm
    DEFAULT_CONCUR_REQ = 5
    MAX_CONCUR_REQ = 1000
    class FetechError(Exception):  # wrap excetions for error reporting
        def __init__(self, country_code):
            self.country_code = country_code
    @asyncio.coroutine
    def get_flag(BASE_URL, cc):
        url = '{}/{cc}/{cc}.gif'.format(BASE_URL, cc=cc.lower())
        resp = yield from aiohttp.request('GET', url)
        if resp.status == 200:
            image = yield from resp.read()
            return image
        elif resp.status == 494:
            raise web.HTTPNotFound()
        else:
            raise aiohttp.HttpProcessingError(code=resp.status, message=resp.reason, headers=resp.headers)
    @asyncio.coroutine
    def download_one(cc, BASE_URL, semaphore, verbose):
        try:
            with (yield from semaphore):  # semaphore is used to limit concurent requests. As a context manager the whole system is not blocked
                image = yield from get_flag(BASE_URL, cc)
        except web.HTTPNotFound:
            status = HTTPStatus.not_found
            msg = 'not found'
        except Exception as exc:
            raise FetechError(cc) from exc  # the 'raise X from Y' syntax chained exceptions
        else:
            save_flag(image, cc.lower() + '.gif')
            status = HTTPStatus.ok
            msg = 'OK'
        if verbose and msg:
            print(cc, msg)
        return Result(status, cc)
    @asyncio.coroutine
    def downloader_coro(cc_list, BASE_URL, verbose, concur_req):  # this is a coroutine, not a plain function
        counter = collections.Counter()
        semaphore = asyncio.Semaphore(concur_req)  # set limits to concurrents
        to_do = [download_one(cc, BASE_URL, semaphore, verbose) for cc in sorted(cc_list)]
        to_do_iter = asyncio.as_completed(to_do)  # an iterator that return futures when done
        if not verbose:
            to_do_iter = tqdm.tqdm(to_do_iter, total=len(cc_list))
        for future in to_do_iter:  # to handle the errors
            try:
                res = yield from future  # retrieve result of an asncio.Future
            except FetechError as exc:
                country_code = exc.country_code
                try:
                    error_msg = exc.__cause__.args[0]  # message from the originial Exception
                except IndexError:
                    error_msg = exc.__cause__.__class__.__name__  # use name of chained exception class when original exception not found
                if verbose and error_msg:
                    msg = '*** Error for {}: {}'
                    print(msg.format(country_code, error_msg))
                status = HTTPStatus.error
            else:
                status = res.status
            counter[status] += 1
        return counter
    def download_many(cc_list, BASE_URL, verbose, concur_req):
        loop = asyncio.get_event_loop()
        coro = downloader_coro(cc_list, BASE_URL, verbose, concur_req)
        counts = loop.run_until_complete(coro)  # instantiate the coroutine and pass to event loop
        looop.close()  # shut down event loop and return counts
    # improve save_flag to avoid blocking thread during I/O
    @asyncio.coroutine
    def download_one(cc, BASE_URL, semaphore, verbose):
        try:
            with(yield from semaphore):
                image = yield from get_flag(BASE_URL, cc)
        except web.HTTPNotFound:
            status = HTTPStatus.not_found
            msg = 'not found'
        except Exception as exc:
            raise FetchError(cc) from exc
        else:
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, save_flag, image, cc.lower() + '.gif')  # calling threadpool executor
            status = HTTPStatus.ok
            msg = 'ok'
        if verbose and msg:
            print(cc, msg)
    return Results(status, cc)

    # two requests per flag
    def http_get(url):
        res = yield from aiohttp.request(('GET', url))
        if res.status == 200:
            ctype = res.headers.get('Content-type', '').lower()
            if 'json' in ctype or url.endswith('json'):
                data = yield from res.json()  # return a dict for json type
            else:
                data  = yield from res.read()  # return the original bytes
            return data
        elif res.status == 404:
            raise web.HTTPNotFound()
        else:
            raise aiohttp.errors.HttpProcessingError(code=res.status, message=res.reason, headers=res.headers)
        @asyncio.coroutine
        def get_country(base_url, cc):
            url = '{}/{cc}/metadata.json'.format(base_url, cc=cc.lower())
            metadata = yield from http_get(url)  # receive a python dict holding JSON contents
            return metadata['country']
        @asyncio.coroutine
        def get_flag(base_url, cc):
            url = '{}/{cc}/{cc}.gif'.format(base_url, cc=cc.lower())
            return (yield from http_get(url))  # in case 'return yield from' return a SyntaxError
        @asyncio.coroutine
        def download_one(cc, base_url, semaphore, verbose):
            try:
                with (yield from semaphore):
                    image = yield from get_flag(base_url, cc)
                with (yield from semaphore):
                    country = yield from get_country(base_url, cc)
            except web.HTTPNotFound:
                status = HTTPStatus.not_found
                msg = 'not found'
            except Exception as exc:
                raise FetchError(cc) from exc
            else:
                country = country.replace(' ', '_')
                filename = '{}-{}.gif'.format(country, cc)
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, save_flag, image, filename)
                status = HTTPStatus.ok
                msg = 'ok'
            if verbose and msg:
                print(cc, msg)
            return Result(status, cc)
    # a TCP server
    import sys
    import pickle
    class UnicodeNameIndex:
        def __init__(self, chars=None):
            self.load(chars)
        def load(self, chars=None):
            self.index = None
            if chars is None:
                try:
                    with open(INDEX_NAME, 'rb') as fp:
                        self.index = pickle.load(fp)
                except OSError:
                    pass
            if self.index is None:
                self.build_index(chars)
            if len(self.index) > MINIMUM_SAVE_LEN:
                try:
                    self.save()
                except OSError as exc:
                    warnings.warn('Could not save {!r}: {}'
                                  .format(INDEX_NAME, exc))
        def save(self):
            with open(INDEX_NAME, 'wb') as fp:
                pickle.dump(self.index, fp)
        def build_index(self, chars=None):
            if chars is None:
                chars = (chr(i) for i in range(32, sys.maxunicode))
            index = {}
            for char in chars:
                try:
                    name = unicodedata.name(char)
                except ValueError:
                    continue
                if name.startswith(CJK_UNI_PREFIX):
                    name = CJK_UNI_PREFIX
                elif name.startswith(CJK_CMP_PREFIX):
                    name = CJK_CMP_PREFIX

                for word in tokenize(name):
                    index.setdefault(word, set()).add(char)
            self.index = index
        def word_rank(self, top=None):
            res = [(len(self.index[key]), key) for key in self.index]
            res.sort(key=lambda item: (-item[0], item[1]))
            if top is not None:
                res = res[:top]
            return res
        def word_report(self, top=None):
            for postings, key in self.word_rank(top):
                print('{:5} {}'.format(postings, key))
        def find_chars(self, query, start=0, stop=None):
            stop = sys.maxsize if stop is None else stop
            result_sets = []
            for word in tokenize(query):
                chars = self.index.get(word)
                if chars is None:  # shorcut: no such word
                    result_sets = []
                    break
                result_sets.append(chars)
            if not result_sets:
                return QueryResult(0, ())
            result = functools.reduce(set.intersection, result_sets)
            result = sorted(result)  # must sort to support start, stop
            result_iter = itertools.islice(result, start, stop)
            return QueryResult(len(result),
                               (char for char in result_iter))
        def describe(self, char):
            code_str = 'U+{:04X}'.format(ord(char))
            name = unicodedata.name(char)
            return CharDescription(code_str, char, name)
        def find_descriptions(self, query, start=0, stop=None):
            for char in self.find_chars(query, start, stop).items:
                yield self.describe(char)
        def get_descriptions(self, chars):
            for char in chars:
                yield self.describe(char)
        def describe_str(self, char):
            return '{:7}\t{}\t{}'.format(*self.describe(char))
        def find_description_strs(self, query, start=0, stop=None):
            for char in self.find_chars(query, start, stop).items:
                yield self.describe_str(char)
        @staticmethod  # not an instance method due to concurrency
        def status(query, counter):
            if counter == 0:
                msg = 'No match'
            elif counter == 1:
                msg = '1 match'
            else:
                msg = '{} matches'.format(counter)
            return '{} for {!r}'.format(msg, query)
    CRLF = b'\r\n'
    PROMPT = b'?>'
    index = UnicodeNameIndex()
    @asyncio.coroutie
    def handle_queries(reader, writer):  # handle multiple queries from each client
        while True:
            writer.write(PROMT)  # this is a plain function
            yield from writer.drain()  # flushes the writer buffer
            data  = yield from reader.readline()  # return bytes
            try:
                query = data.decode().strip()
            except UnicodeDecodeError:
                query = '\x00'  # pretend to be blank
            client = writer.get_extra_info('peername')  # return the remote adress
            print('Received from {}: {!r}'.format(client, query))
            if query:
                if ord(query[:1] < 32):  # a control or null charater is received
                    break
                lines = list(index.find_description_strs(query))
                if lines:
                    writer.writelines(line.encode() + CRLF for line in lines)  # a generator expression
                writer.write(index.status(query, len(lines)).encode() + CRLF)
                yield from writer.drain()
                print('Sent {} results'.format(len(lines)))
        print('Close the clinet socket')
        writer.close()
    def server_start(address='127.0.0.1', port=2323):
        port = int(port)
        loop = asyncio.get_event_loop()
        server_coro = asyncio.start_server(handle_queries, address, port, loop=loop)  # return asyncio.Server, an TCP socket server
        server = loop.run_until_complete(server_coro)
        host = server.sockets[0].getsockname()
        print('Serving on {}. Hit CTRL-C to stop.'.format(host))
        try:
            loop.run_forever()  # block the line until killed
        except KeyboardInterrupt:
            pass
        print('Server shutting down')
        server.close()
        loop.run_until_complete(server.wait_closed())  # .wait_closed() returns a Future.
        loop.close()
    # a HTTP server
    @asyncio.coroutine
    def init(loop, address, port):
        app = web.Application(loop=loop)
        app.router.add_route('GET', '/', home)
        handler = app.make_handler()  # return a handler to handle HTTP request
        server = yield from loop.create_server(handler, address, port)
        return server.sockets[0].getsockname()
    def main(address='127.0.0.1', port=8888):
        port = int(port)
        loop = asyncio.get_event_loop()
        host = loop.run_until_complete(init(loop, address, port))  # run init to start the server
        print('Serving on {}. HIt CTRL=C to stop.'.format(host))
        try:
            loop.run_forever()  # main will be block here
        except KeyboardInterrupt:
            pass
        print('Server shutting down.')
        loop.close()
    def home(request):
        query = request.GET.get('query', '').strip()  # strip blanks
        print('Query: {!r}'.format(query))
        if query:
            descriptions = list(index.find_descriptions(query))
            res = '\n'.join(ROW_TPL.format(**vars(descr)) for descr in descriptions)
            msg = index.status(query, len(descriptions))
        else:
            descriptions = []
            res = ''
            msg = 'Enter words describing character.'
        html = template.format(query=query, results=res, message=msg)
        print('sending {} results'.format(len(descriptions)))
        return web.Response(content_type=CONTENT_TYPE, text=html)


if __name__ == "__main__":
    main()