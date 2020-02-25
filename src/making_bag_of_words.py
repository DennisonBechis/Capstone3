import pandas as pd
import numpy as np
from nltk import *
from nltk.corpus import stopwords
from debate_cleaning import *
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')

    df = unusable_rows(df)
    df = get_candidates(df)

    X = df['speech'].to_numpy()
    y = df['speaker'].to_numpy()

    X_train, y_train = train_test_split(df, test_size = .2, stratify=df['speaker'].values)


    # # 1. Create a set of documents.
    # documents = [''.join(article).lower() for article in X_train]
    #
    # cv = CountVectorizer(stop_words='english')
    # vectorized = cv.fit_transform(documents)
    #
    # tfidf = TfidfVectorizer(stop_words='english')
    # tfidfed = tfidf.fit_transform(documents)
    #
    # classifier = MultinomialNB()
    # classifier.fit(tfidfed, y_train)
    #
    # predictions = classifier.predict()
