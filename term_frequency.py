from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

def _do_counter(text):
    print('Starting thread...')
    vectorizer = CountVectorizer(lowercase=True, ngram_range = (5,5))
    analyzer = vectorizer.build_analyzer()
    _c =  Counter(analyzer(text))
    print('Thread finished!')
    return _c