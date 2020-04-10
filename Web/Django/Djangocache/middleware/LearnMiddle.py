import random
import time

from django.core.cache import cache
from django.http import HttpResponse
from django.shortcuts import redirect
from django.urls import reverse
from django.utils.deprecation import MiddlewareMixin


class HelloMiddle(MiddlewareMixin):

    def process_request(self, request):
        ip = request.META.get('REMOTE_ADDR')

        # if request.path == '/app/getphone/':
        #     if ip == '127.0.0.1':
        #         if random.randrange(100) > 20:
        #             return HttpResponse('success')
        #
        # if request.path == '/app/getticket/':
        #     if ip.startswith('10.0.122.7'):
        #         return HttpResponse('fail')
        #
        # if request.path == '/app/search/':
        #     result = cache.get('ip')
        #     if result:
        #         return HttpResponse('wait 10s')
        #     cache.set('ip', ip, timeout=10)

        # default to [] if get returns none

        black_list = cache.get('black', [])
        if ip in black_list:
            return HttpResponse('in black list')

        requests = cache.get(ip, [])
        while requests and time.time() - requests[-1] > 60:
            requests.pop()

        requests.insert(0, time.time())
        cache.set(ip, requests, timeout=60)

        if len(requests) > 30:
            black_list.append(ip)
            cache.set('black', black_list, timeout=60*60*24)
            return HttpResponse('scrape stopped')

        if len(requests) > 10:
            return HttpResponse('request too often')

    def process_exception(self, request, exception):
        print(request, exception)
        return redirect(reverse('app:index'))


class TwoMiddle(MiddlewareMixin):
    def process_request(self, request):
        print('two middleware')