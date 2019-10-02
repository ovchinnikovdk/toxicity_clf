from nltk.tokenize import WordPunctTokenizer, RegexpTokenizer
import nltk
from nltk.stem import SnowballStemmer
import re
from nltk.corpus import stopwords

# Use for downloading NLTK corpora
# nltk.download()


class TextPreprocess:
    def __init__(self, preprocess_func=None):
        if preprocess_func is not None:
            self.process = preprocess_func
        else:
            self.process = self._common_process
            self.tokenizer = RegexpTokenizer(r'\w+')
            self.stop_words = set(stopwords.words('english'))

    def _common_process(self, text):
        text = re.sub("[^a-zA-Z]", " ", text)
        tokens = self.tokenizer.tokenize(text.lower())
        tokens = list(filter(lambda x: len(x) < 20 and x not in self.stop_words, tokens))
        stemmer = SnowballStemmer(language='english')
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    def __call__(self, text):
        return self.process(text)
