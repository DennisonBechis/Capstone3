import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from debate_cleaning import *
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def find_centroids(cluster_matrix):
    for x in cluster_matrix:
        pass


def build_text_vectorizer(contents, use_tfidf=True, use_stemmer=False, max_features=None):

    Vectorizer = TfidfVectorizer if use_tfidf else CountVectorizer
    tokenizer = RegexpTokenizer(r"[\w']+")
    stem = PorterStemmer().stem if use_stemmer else (lambda x: x)
    stop_set = set(stopwords.words('english'))
    stop_set = stop_set.union({'go','way','say','back','sure','would','think','see','first','second','one','said','go','put',
    'get','going','think','get','done','having','has','need','want','us','got','well','crosstalk','they', 'whatd','?','whatd they say?'})

    # Closure over the tokenizer et al.
    def tokenize(text):
        tokens = tokenizer.tokenize(text)
        stems = [stem(token) for token in tokens if token not in stop_set]
        return stems

    vectorizer_model = Vectorizer(tokenizer=tokenize, max_features=max_features)
    vectorizer_model.fit_transform(contents)
    return_model = vectorizer_model.fit_transform(contents)
    vocabulary = np.array(vectorizer_model.get_feature_names())

    # Closure over the vectorizer_model's transform method.
    def vectorizer(X):
        return vectorizer_model.transform(X).toarray()

    return vectorizer, vocabulary, return_model


if __name__ == "__main__":
    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
    df1 = pd.read_csv("/Users/bechis/DSI/repo/capstone3/data/debate_transcripts_nevada.csv", encoding= 'unicode_escape')
    df = pd.concat([df, df1])
    df = unusable_rows(df)
    df = get_candidates(df)

    vectorizer, features, X = build_text_vectorizer(df['speech'], use_tfidf=True, use_stemmer=True, max_features=None)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['speech'])
    features = vectorizer.get_feature_names()
    kmeans = KMeans(n_clusters=13)
    kmeans.fit(X)

    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print("\n3) top features (words) for each cluster:")
    for num, centroid in enumerate(top_centroids):
        print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))

    print("\n5) random sample of titles in each cluster")
    assigned_cluster = kmeans.transform(X).argmin(axis=1)

    distances = pairwise.euclidean_distances(X, kmeans.cluster_centers_)
    # print(distances.shape)
    # a = np.argsort(distances[0])
    # print(np.argsort(distances[0]))

    for x in range(len(distances[0,:])):
        closest_to_0_cluster = np.argsort(distances[:,x])
        # print(closest_to_0_cluster)
        print("topic {}:".format(x))
        for speech in range(len(df['speech'].iloc[closest_to_0_cluster[0:10]])):
            print(df['speaker'].iloc[closest_to_0_cluster[speech]])
            print(df['speech'].iloc[closest_to_0_cluster[speech]])
            print('\n')
        print('\n')
        print(df['speaker'].iloc[closest_to_0_cluster[0:10]])
        print('\n')

    # # print(distances)
    # # print(len(distances))
    # return_list = []
    #
    # for x in distances:
    #     return_list.append(np.average(x))
    #
    # print(return_list)
    #
    # sorted_index = np.argsort(return_list)[-11:-1]
    # print(sorted_index)
    # print(sorted_index[0])
    #
    # print(df['speech'][sorted_index[0]])
    # print(assigned_cluster)
