import bs4 as bs
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import nltk
import re

class Book(object):
    curdir = os.path.dirname(os.path.realpath(__file__)) + os.sep + ".." + os.sep + "gap-html" + os.sep
    stemmer = SnowballStemmer("english")
    english_stopwords = stopwords.words('english')
    nltk.download("stopwords")

    def __init__(self, _id,  subdir, pages):
        self.id = _id
        self.meta = {}
        print("starting book {}".format(self.id))
        self.pages= pages
        self.word_list = []
        self.dir = subdir + os.sep
        self.word_list = self.create_raw_word_list()
        self.word_list = self.remove_stop_words(self.word_list)
        print("Finished preprocessing on book{}".format(_id))

    def create_raw_word_list(self):
        word_list = []
        for page in self.pages:
            with open(self.dir + page, 'r') as f:
                webpage = f.read()
            soup = bs.BeautifulSoup(webpage, "html.parser")

            if page=="meta.html":
                author_node = soup.find(id="author")
                self.meta["author"] = author_node.text
                title_node = soup.find(id="title")
                self.meta["title"] = title_node.text
                published_node = soup.find(id="published")
                self.meta["published"] = published_node.text
            else:
                for node in soup.findAll('span'):
                    for word in node.text.split():
                        if re.search('[a-zA-Z]', word):
                            word_list.append(word.lower())
        return word_list

    def remove_stop_words(self, text_list):
        word_list = [word for word in text_list if word not in self.english_stopwords]
        return word_list

    def stem_words(self,text_list):
        stems = [self.stemmer.stem(t) for t in text_list]
        return stems

    def unique_words(self):
        return Counter(self.world_list)
