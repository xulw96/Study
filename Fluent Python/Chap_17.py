import os
import time
import sys
import requests

def main():
    # sequentially downloading national flags
    POP20_CC = ('CN IN US ID BR PK NG BD RU JP'
                'MX PH VN ET EG DE IR TR CD FR').split()
    BASE_URL = 'http://flupy.org/data/flags'
    DEST_DIR = './Downloads/'
    def save_flag(img, filename):
        path = os.path.join(DEST_DIR, filename)
        with open(path, 'wb') as fp:
            fp.write(img)
    def get_flag(cc):
        url = '{}/{cc}/{cc}.gif'.format(BASE_URL, cc=cc.lower())
        resp = requests.get(url)
        return resp.content
    def show(text):  # to see downloading process
        print(text, end='')
        sys.stdout.flush()
    def download_many(cc_list):  # key function for sequential downloading
        for cc in sorted(cc_list):
            image = get_flag(cc)
            show(cc)
            save_flag(image, cc.lower() + '.gif')
        return len(cc_list)
    def report():  # report elapsed time for whole downloading
        t0 = time.time()
        count = download_many(POP20_CC)
        elapsed = time.time() - t0
        msg = '\n{} flags downloaded in {:.2f}s'
        print(msg.format(count, elapsed))
    # concurrent downloading
    from concurrent import futures
    MAX_WORKERS = 20  # define max threads
    def download_one(cc):  # function to download a single image
        image = get_flag(cc)
        show(cc)
        save_flag(image, cc.lower() + '.gif')
        return cc
    def download_many(cc_list):
        workers = min(MAX_WORKERS, len(cc_list))
        with futures.ThreadPoolExecutor(workers) as executor:
            res = executor.map(download_one, sorted(cc_list))  # calling 'download_one' from multiple threads
        return len(list(res))  # use list to handle  exception when aroused: calling 'next()'
    # inspect future
    def download_many(cc_list):
        cc_list = cc_list[:5]
        with futures.ThreadPoolExecutor(max_workers=3) as executor:
            to_do = []
            for cc in sorted(cc_list):
                future = executor.submit(download_one, cc)  # .submit() schedule the callable and return a 'future'
                to_do.append(future)
                msg = 'Scheduled for {}: {}'
                print(msg.format(cc, future))
            results = []
            for future in futures.as_completed(to_do):  # .as_completed() yield future when completed
                res = future.result()
                msg = '{} result: {}'
                print(msg.format(future, res))
                results.append(res)
        return len(results)
    # executor.map
    from time import sleep, strftime
    from concurrent import futures
    def display(*args):
        print(strftime('[%H:%M:%S]', end=''))
        print(*args)
    def loiter(n):
        msg = '{}loiter({}): doing nothing for {}s...'
        display(msg.format('\t'*n, n, n))  # '\t' for a tab
        sleep(n)  # time.sleep will release GIL, and loiter(1) may precede loiter(0), as python runs another script
        msg = '{}loiter({}): done.'
        display(msg.format('\t'*n, n, n))
        return n * 10  # a 'return' to collect results
    def map_test():
        display('Script starting')
        executor = futures.ThreadPoolExecutor(max_workers=3)
        results = executor.map(loiter, range(5))  # .map() returns a generator and will not block the line
        display('results:', results)  # will immediately show. Results will be in same order as calls
        for i, result in enumerate(results):  # this 'for loop' will call next() on .result() and will block the line until next item is ready
            display('results {}: {}'.format(i, result))
    # TQDM
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        time.sleep(0.01)
    # Error handling
    from collections import namedtuple
    from enum import Enum
    HTTPStatus = Enum('status', 'ok not_found error')
    Result = namedtuple('Results', 'status cc')
    def get_flag(BASE_URL, cc):
        url = '{}/{cc}/{cc}.gif'.format(BASE_URL, cc=cc.lower())
        resp = requests.get(url)
        if resp.status_code != 200:
            resp.raise_for_status()  # only to raise an exception, not handle error
        return resp.content
    def download_one(cc, BASE_URL, verbose=False):
        try:
            image = get_flat(BASE_URL, cc)
        except requests.exceptions.HTTPError as exc:
            res = exc.response
            if res.status_code == 404:
                status = HTTPStatus.not_found  # set local_status
                msg = 'not found'
            else:
                raise  # raise HTTPError, and propagate other exceptions
        else:
            save_flag(image, cc.lower() + '.gif')
            status = HTTPStatus.ok
            msg = 'ok'
        if verbose:
            print(cc, msg)
        return Result(status, cc)
    def download_many(cc_list, BASE_URL, verbose, max_req):
        counter = collections.Counter()  # handle the download outcome
        cc_iter = sorted(cc_list)
        if not verbose:
            cc_iter = tqdm(cc_iter)
        for cc in cc_iter:
            try:
                res = download_one(cc, BASE_URL, verbose)
            except requests.exceptions.HTTPError as exc:  # handle those not handled before. e,g: 'download_one()'
                error_msg = 'HTTP error {res.status_code} - {res.reason}'
                error_msg = error_msg.format(res=exc.response)
            except requests.exceptions.ConnectionError as exc:  # handle network-related error
                error_msg = 'Connection error'
            else:
                error_msg = ''
                status = res.status  # no exception raised, then retrieve status from the Result(status, cc)
            if error_msg:
                status = HTTPStatus.error  # set local status with coming error
            counter[status] += 1
            if verbose and error_msg:
                print('*** Error for {}: {}'.format(cc, error_msg))
        return counter  # return the result for the main() function to show
    # futures.as_completed
    DEFAULT_CONCUR_REQ = 30
    MAX_CONCUR_REQ = 1000
    def download_many(cc_list, BASE_URL, verbose, concur_req):  # concur_req will be computed by main() function for the min
        counter = collections.Counter()
        with futures.ThreadPoolExecutor(max_workers=concur_req) as executor:
            to_do_map = {}  # use dict to map each Future instance
            for cc in sorted(cc_list):  # the result_order depend on HTTP response instead of submit()
                future = executor.submit(download_one, cc, BASE_URL, verbose)  # input the callable and arguments received
                to_do_map[future] = cc  # store into the dict
            done_iter = future.as_completed(to_do_map)  # an iterator over futures when completed
            if not verbose:
                done_iter = tqdm(done_iter, total=len(cc_list))  # grant expected number of items for computing work remaining time
            for future in done_iter:  # iterate over the finished futures
                try:
                    res = future.result()  # as .as_completed only returns finished futures, this call will not block the line
                except requests.exceptions.HTTPError as exc:    # the following is the same as above sequential version of 'download_many()'
                    error_msg = 'HTTP error {res.status_code} - {res.reason}'
                    error_msg = error_msg.format(res=exc.response)
                except requests.exceptions.ConnectionError as exc:
                    error_msg = 'Connection error'
                else:
                    error_msg = ''
                    status = res.status
                if error_msg:
                    status = HTTPStatus.error
                counter[status] += 1
                if verbose and error_msg:
                    cc = to_do_map[future]
                    print('*** Error for {}: {}'.format(cc, error_msg))
        return counter



if __name__ == "__main__":
    main()