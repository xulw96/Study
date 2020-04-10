def main():
    from urllib.request import urlopen
    from io import StringIO
    import csv

    # read a csv_file online
    '''data = urlopen('http://pythonscraping.com/files/MontyPythonAlbums.csv').read().decode('ascii', 'ignore')  # specify the decoding method
    data_file = StringIO(data)  # wrao the text into an object
    csv_reader = csv.reader(data_file)  # in a list object, including first line
    dict_reader = csv.DictReader(data_file)  # in a dict object, first line as attibutes
    print(dict_reader.fieldnames)'''

    # read PDF
    from pdfminer.pdfinterp import PDFResourceManager, process_pdf
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from io import StringIO
    from io import open
    def read_pdf(file):
        resource = PDFResourceManager()
        string = StringIO()
        device = TextConverter(resource, string, laparams=LAParams())
        process_pdf(resource, device, file)
        device.close()

        content = string.getvalue()
        string.close()
        return content
    '''file = urlopen('http://pythonscraping.com/pages/warandpeace/chapter1.pdf')
    # file = open('./chapter1.pdf', 'rb')
    output = read_pdf(file)
    print(output)'''

    # read a Word file
    from zipfile import ZipFile
    from urllib.request import urlopen
    from io import BytesIO
    from bs4 import BeautifulSoup
    def read_word(file):
        file = BytesIO(file)
        document = ZipFile(file)
        xml_content = document.read('word/document.xml')

        word_obj = BeautifulSoup(xml_content.decode('utf-8'), 'xml')
        text = word_obj.find_all('w:t')  # text is contained in the 'w:t' tags
        for element in text:
            style = element.parent.parent.find('w:pStyle')
            if style is not None and style['w:val'] == 'Title':
                print('Title is: {}'.format(element.text))
            else:
                print(element.text)
    file = urlopen('http://pythonscraping.com/pages/AWordDocument.docx').read()
    read_word(file)




if __name__ == '__main__':
    main()