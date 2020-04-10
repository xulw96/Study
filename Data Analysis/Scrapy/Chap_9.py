def main():
    # summarize data
    import re
    from urllib.request import urlopen
    from bs4 import BeautifulSoup
    import string
    from collections import Counter
    def clean_sentence(sentence):
        sentence = sentence.split(' ')
        sentence = [word.strip(string.punctuation + string.whitespace) for word in sentence]
        sentence = [word for word in sentence if len(word) > 1 or (word.lower() == 'a' or word.lower() == 'i')]
        return sentence
    def clean_input(content):
        content = content.upper()
        content = re.sub('\n', ' ', content)
        content = bytes(content, 'UTF-8')
        content = content.decode('ascii', 'ignore')
        sentences = content.split('. ')
        return [clean_sentence(sentence) for sentence in sentences]
    def get_n_grams_from_sentece(content, n):
        output = []
        for i in range(len(content) - n + 1):
            output.append(content[i:i+n])
        return output
    def get_n_grams(content, n):
        content = clean_input(content)
        n_grams = Counter()
        for sentence in content:
            new_n_grams = [' '.join(n_gram) for n_gram in get_n_grams_from_sentece(sentence, 2)]
            n_grams.update(new_n_grams)
        return n_grams
    '''content = str(urlopen('http://pythonscraping.com/files/inaugurationspeech.txt').read(), 'utf-8')
    n_grams = get_n_grams(content, 2)
    print(n_grams)'''

    # markov chain. Nested dict
    from random import randint
    def word_list_sum(word_list):  # to enable weighted randint
        sum = 0
        for word, value in word_list.items():  # iterate the key and value of the dict
            sum += value
        return sum
    def retrieve_random_word(word_list):
        rand_index = randint(1, word_list_sum(word_list))  # gain a weighted random value from the last function
        for word, value in word_list.items():
            rand_indedx -= value
            if rand_index <= 0:
                return word
    def build_word_dict(text):
        text = text.replace('\n', ' ')
        text = text.replace('"', '')  # remove citation
        punctuation = [',', '.', ';', ':']
        for symbol in punctuation:
            text = text.replace(symbol, ' {} '.format(symbol))  # make sure puntuation as word
        words = text.split(' ')
        words = [word for word in words if word != '']  # filter empty word
        word_dict = {}
        for i in range(1, len(words)):
            if words[i-1] not in word_dict:
                word_dict[words[i-1]] = {}  # build nested dict for this word
            if words[i] not in word_dict[words[i-1]]:
                word_dict[words[i-1]][words[i]] = 0
            word_dict[words[i-1]][words[i]] += 1
        return word_dict
    text = str(urlopen('http://pythonscraping.com/files/inaugurationSpeech.txt'))
    word_dict = build_word_dict(text)

    length = 100
    '''chain = ['I']
    for i in range(0, length):
        new_word = retrieve_random_word(word_dict[chain[-1]])
        chain.append((new_word))
    print(' '.join(chain))'''

    # breath-first search
    import pymysql
    conn = pymysql.connect(host='localhost', user='root', passwd='0618', db='mysql', charset='utf8')
    cur = conn.cursor()
    cur.execute('USE wikipedia')
    def get_url(page_id):  # get url from database
        cur.execute('SELECT url FROM pages WHERE id = %s', int(page_id))
        return cur.fetchone()[0]
    def get_links(from_page_id):
        cur.execute('SELECT to_page_id FROM links WHERE from_page_id = %s', int(from_page_id))
        if cur.rowcount == 0:
            return []
        return [x[0] for x in cur.fetchall()]
    def search_breadth(target_page_id, paths=[[1]]):
        new_paths = []
        for path in paths:
            links = get_links(path[-1])
            for link in links:
                if link == target_page_id:
                    return path + [link]
                else:
                    new_paths.append(path + [link])  # return a longer list for next search
        return search_breadth(target_page_id, new_paths)  # a recursion for searching
    '''nodes = get_links(1)
    target_page_id = 28624
    page_ids = search_breadth(target_page_id)
    for page_id in page_ids:
        print(get_url(page_id))'''

    # NLT
    from nltk import word_tokenize
    from nltk import Text
    from nltk import pos_tag
    text = word_tokenize('the dust was thick so he had to dust')
    print(pos_tag(text))



if __name__ == '__main__':
    main()