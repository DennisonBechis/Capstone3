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

if __name__ == "__main__":
    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
    df = unusable_rows(df)
    df = get_candidates(df)
    coll = df['speech'].to_numpy()
    print(df.info())
    print(len(coll))
    # 1. Create a set of documents.
    documents = [''.join(article).lower() for article in coll]

    cv = CountVectorizer(stop_words='english')
    vectorized = cv.fit_transform(documents)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidfed = tfidf.fit_transform(documents)
