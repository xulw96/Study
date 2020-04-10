    # encoding = 'utf_8'
def main():
    # encode and decode
    s = 'café'
    print("length for 'café':", len(s))
    b = s.encode('utf8')  # convert string into bytes
    print("bytes for 'café':", b)  # '\x00' for other byte value. here 'é' takes two bytes, hence the '\xc3\xa9'
    print("length for the bytes:", len(b))
    a = b.decode('utf8')
    print("decode the bytes:", a)
    # bytes and bytearray
    cafe = bytes('café', encoding='utf_8')  # call function to get bytes
    cafe_arr = bytearray(cafe)
    print(cafe, ' ', cafe[0], ' ', cafe[:1])  # each item in bytes is an integer; but slices get bytes
    print(cafe_arr, ' ', cafe_arr[0], ' ', cafe_arr[:1])  # slices get bytearray
    bytes_hex = bytes.fromhex('31 4B CE A9')  # build binary sequence by parsing pairs
    print(bytes_hex)
    import array
    numbers = array.array('h', [-2, -1, 0, 1, 2])  # 'h' for short integer(16 bits)
    octets = bytes(numbers)  # build binary sequence from buffer-like object
    print(octets)
    # memoryview
    """import struct
    fmt = '<3s3sHH'  # 3s for 3 bytes, H for 16-bit integers
    with open('filter.gif', 'rb') as fp:
        img = memoryview(fp.read())  # gain acccess shared memory
    header = img[:10]  # extract the bytest from memory
    header_bytes = bytes(header)
    print(header_bytes)
    struct_unpack = struct.unpack(fmt, header)  # unpack bytes into tuple
    print(struct_unpack)
    del header
    del img  # delete to release the memory"""
    # encoder/decoder
    for codec in ['latin_1', 'utf_8', 'utf_16']:
        print(codec, 'El Niño'.encode(codec), sep='\t')  # note the use of "sep='\t'" here.
    # UnicodeEncodeError
    city = 'São Paulo'
    '''print(city.encode('cp437'))'''
    print(city.encode('cp437', errors='ignore'))  # ignore the error
    print(city.encode('cp437', errors='replace'))  # replace with '?'
    print(city.encode('cp437', errors='xmlcharrefreplace'))  # replace with XML entity
    # UnicodeDecodeError
    octets = b'Montr\xe9al'
    print(octets.decode('cp1252'))
    print(octets.decode('iso8859_7'))  # misinterpreted
    '''print(octets.decode('utf_8'))'''
    print(octets.decode('utf_8', errors='replace'))  # replace with '�'
    # handle text file
    fp = open('cafe.txt', 'w', encoding='utf_8')  # read in text mode
    print(fp)  # return a TextIOWrapper object
    print(fp.write('café'))
    fp.close()
    import os
    print(os.stat('cafe.txt').st_size)  # return the bytes being holded
    fp2 = open('cafe.txt', encoding='cp1252')
    print(fp2.encoding)  # TextIOWrapper obecjt has an encoding attribute
    print(fp2.read())  # wrong encoding type
    fp4 = open('cafe.txt', 'rb')  # read in binary mode
    print(fp4)  # return a BufferedReader object
    print(fp4.read())
    # encoding defaults
    import sys
    import locale
    my_file = open('dummy', 'w')
    expressions = """locale.getpreferredencoding()
            type(my_file)
            my_file.encoding
            sys.stdout.isatty()  
            sys.stdout.encoding
            sys.stdin.isatty()
            sys.stdin.encoding
            sys.stderr.isatty()
            sys.stderr.encoding
            sys.getdefaultencoding()
            sys.getfilesystemencoding()"""
    for expression in expressions.split():
        value = eval(expression)
        print(expression.rjust(30), '->', repr(value))
    # nomarlizing Unicode for comparison
    s1 = 'café'
    s2 = 'cafe\u0301'  # canonical equivalent
    print(s1, s2)
    print(len(s1), len(s2))
    print(s1 == s2)
    from unicodedata import normalize
    print(len(normalize('NFC', s1)), len(normalize('NFC', s2)))  # 'NFC' for shortest, 'NFD' for base
    print(normalize('NFC', s1) == normalize('NFC', s2))  # 'NFKC' would be for compatibility mode
    # utility functions for text matching
    s1 = 'café'
    s2 = 'cafe\u0301'
    s3 = 'Straße'
    s4 = 'strasse'
    def nfc_equal(str1, str2):
        return normalize('NFC', str1) == normalize('NFC', str2)
    def fold_equal(str1, str2):
        return (normalize('NFC', str1).casefold() == normalize('NFC', str2).casefold())
    print("NFC, case sensitive", nfc_equal(s1, s2))
    print("NFC, case sensitive", nfc_equal(s3, s4))
    print("NFC, case folding", fold_equal(s1, s2))
    print("NFC, case folding", fold_equal(s3, s4))
    # remove combining marks
    import unicodedata
    import string
    def shave_marks(txt):
        '''remove all diacritic marks'''
        norm_txt = unicodedata.normalize('NFD', txt)
        shaved = ''.join(c for c in norm_txt if not unicodedata.combining(c))  # 'join' to concantenate strings
        return unicodedata.normalize('NFC', shaved) # recompose characters
    a = 'deutschland'
    shave_marks(a)
    print(a)
    # remove combining marks only for those from Latin characters
    def shave_marks_latin(txt):
        '''remove all diacritic marks form latin base character'''
        norm_txt = unicodedata.normalize('NFD', txt)
        latin_base = False
        keepers= []
        for c in norm_txt:
            if unicodedata.combining(c) and latin_base:
                continue  # ignore on latin base char
            keepers.append(c)
            if not unicodedata.combining(c):
                latin_base = c in string.ascii_letters  # new base char if not combining char
        shaved = ''.join(keepers)
        return unicodedata.normalize('NFC', shaved)
    a = 'öä'
    shave_marks_latin(a)
    print(a)
    # replace using mapping table
    map = str.maketrans({'ß': 'ss'})
    def dewinize(txt):
        '''using mapping table to replace char'''
        return txt.translate(map)
    a = 'ß'
    dewinize(a)
    a = a.replace('ß', 'ss')
    print(a)
    # Unicode database
    import unicodedata
    import re
    re_digit = re.compile(r'\d')
    sample = '1\xbc\xb2\u0969\u136b\u216b\u2466\u2480\u3285'
    for char in sample:
        print('U+%04x' % ord(char),  # code point in U+0000 format
            char.center(6),  # character centralized in a string of length 6
            're_dig' if re_digit.match(char) else '-',
            'isdig' if char.isdigit() else '-',
            'isnum' if char.isnumeric() else '-',
            format(unicodedata.numeric(char), '5.2f'),  # numeric format with width 5 and decimal places 2
            unicodedata.name(char),
            sep='\t')
    # str VS bytes on re module
    import re
    re_numbers_str = re.compile(r'\d+')
    re_words_str = re.compile(r'\w+')
    re_numbers_bytes = re.compile(rb'\d+')
    re_words_bytes = re.compile(rb'\w+')
    text_str = ("Ramanujan saw \u0be7\u0bed\u0be8\u0bef"
                " as 1729 = 1^3 + 12^3 = 9^3 + 10^3.")
    text_bytes = text_str.encode('utf_8')
    print('Text', repr(text_str), sep='\n')
    print('Numbers')
    print('  str  :', re_numbers_str.findall(text_str))
    print('  bytes:', re_numbers_bytes.findall(text_bytes))
    print('Words')
    print('  str  :', re_words_str.findall(text_str))
    print('  bytes:', re_words_bytes.findall(text_bytes))
    # str VS bytes on os module
    print(os.listdir('.'))
    print(os.listdir(b'.'))
    name_str = os.listdir('.').decode('ascii', 'surrogateescape')  # 'surrogateescape' to get rid of nonencodable bytes
    name_str.encode('ascii', 'surrogateescape')


if __name__ == "__main__":
    main()