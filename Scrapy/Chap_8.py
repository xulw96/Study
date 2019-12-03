def main():
    # n-gram
    from urllib.request import urlopen
    from bs4 import BeautifulSoup

    def get_n_gram(content, n):
        content = content.split(' ')
        output = []
        for i in range(len(content) - n + 1):
            output.append(content[i:i+n])
        return output
    '''html = urlopen('http://en.wikipedia.org/wiki/python_(programming_language)')
    bs = BeautifulSoup(html, 'html.parser')
    content = bs.find('div', {'id': 'mw-content-text'}).get_text()
    n_gram = get_n_gram(content, 2)
    print(n_gram)
    print('2-grams count is: {}'.format(str(len(n_gram))))'''

    # regex for refining
    import re
    def get_n_gram(content, n):
        content = re.sub('\n|[[\d+\]]', ' ', content)  # remove citation [123], newline
        content = bytes(content, 'UTF-8')
        content = content.decode('ascii', 'ignore')  # code and decode to filter bad one
        content = content.split(' ')
        content = [word for word in content if word != '']
        output = []
        for i in range(len(content) - n + 1):
            output.append(content[i:i+n])
        return output

    import string
    def clean_sentence(sentence):
        sentence = sentence.split(' ')
        sentence = [word.strip(string.punctuation + string.whitespace for word in sentence)]
        sentence = [word for word in sentence if len(word) > 1 or (word.lower() == 'a' or
                                                                   word.lower() == 'i')]
        return sentence
    def clean_input(content):
        content = re.sub('\n|[[\d+\]]', ' ', content)
        content = bytes(content, 'UTF-8')
        content = content.decode('ascii', 'ignore')
        sentences = content.split('. ')  # split based on periods
        return [clean_sentence(sentence) for sentence in sentences]
    def get_n_grams_from_sentence(content, n):
        output = []
        for i in range(len(content) - n + 1):
            output.append(content[i:i+n])
        return output
    def get_n_grams(content, n):
        content = clean_input(content)
        n_grams = []
        for sentence in content:
            n_grams.extend(get_n_grams_from_sentence(sentence, n))
        return n_grams

    # data normalization
    from collections import Counter
    def get_n_grams(content, n):
        content = clean_input(content)
        n_grams = Counter()  # a dict to count hashable objects
        for sentence in content:
            new_n_grams = [' '.join(n_gram) for n_gram in get_n_grams_from_sentence(sentence, 2)]
            n_grams.update(new_n_grams)  # update the dict with counter
        return n_grams


if __name__ == '__main__':
    main()