import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from debate_cleaning import *
from nltk.stem.wordnet import WordNetLemmatizer
from matplotlib import pyplot as plt


def build_text_vectorizer(contents, use_tfidf=True, use_stemmer=False, max_features=None):

    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    tokenizer = RegexpTokenizer(r"[\w']+")
    stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
    # stem = WordNetLemmatizer().lemmatize
    stop_set = set(stopwords.words('english'))
    stop_set = stop_set.union({'go','way','say','back','sure','would','think','see','first','second','one','said','go','put',
    'get','going','think','get','done','having','has','need','want','us','got','well','crosstalk'})

    # Closure over the tokenizer et al.
    def tokenize(text):
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token not in stop_set]
        return stems

    vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=max_features)
    vectorizer_model.fit(contents)
    vocabulary = np.array(vectorizer_model.get_feature_names())

    # Closure over the vectorizer_model's transform method.
    def vectorizer(X):
        return vectorizer_model.transform(X).toarray()

    return vectorizer, vocabulary

def hand_label_topics(H, vocabulary):

    hand_labels = []
    for i, row in enumerate(H):
        top_five = np.argsort(row)[::-1][:20]
        print('topic', i)
        print('-->', ' '.join(vocabulary[top_five]))
        label = input('please label this topic: ')
        hand_labels.append(label)
        print()
    return hand_labels

def analyze_article(article_index, contents, W, hand_labels):
    '''
    Print an analysis of a single NYT articles, including the article text
    and a summary of which topics it represents. The topics are identified
    via the hand-labels which were assigned by the user.
    '''
    print(contents.iloc[article_index])
    dictionary_results = {}
    speaker = df.iloc[article_index]['speaker']
    probs = softmax(W[article_index], temperature=0.01)
    for prob, label in zip(probs, hand_labels):
        dictionary_results[label] = prob
        
    return dictionary_results

def softmax(v, temperature=1.0):
    '''
    A heuristic to convert arbitrary positive values into probabilities.
    See: https://en.wikipedia.org/wiki/Softmax_function
    '''
    expv = np.exp(v / temperature)
    s = np.sum(expv)
    return expv / s

if __name__ == '__main__':

    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
    df = unusable_rows(df)
    df = get_candidates(df)

    contents = df['speech']
    vectorizer, vocabulary = build_text_vectorizer(contents,
                                use_tfidf=True,
                                use_stemmer=False,
                                max_features=5000)
    X = vectorizer(contents)

    nmf = NMF(n_components=14, max_iter=100, alpha=0.0)

    W = nmf.fit_transform(X)
    H = nmf.components_

    rand_articles = np.random.choice(list(range(len(W))), 15)
    hand_labels = hand_label_topics(H, vocabulary)

    dictionary = []

    for i in range(len(W)):
         dictionary_results = analyze_article(i, contents, W, hand_labels)
         print(dictionary_results)
         break
