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


def find_centroids(cluster_matrix):
    for x in cluster_matrix:
        pass


if __name__ == "__main__":
    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
    df = unusable_rows(df)
    df = get_candidates(df)


    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['speech'])
    features = vectorizer.get_feature_names()
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(X)

    top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
    print(kmeans.cluster_centers_)
    print("\n3) top features (words) for each cluster:")
    for num, centroid in enumerate(top_centroids):
        print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))

    print("\n5) random sample of titles in each cluster")
    assigned_cluster = kmeans.transform(X).argmin(axis=1)

    distances = pairwise.euclidean_distances(X, kmeans.cluster_centers_)
    # print(distances.shape)
    # a = np.argsort(distances[0])
    # print(np.argsort(distances[0]))

    for x in distances[0]:
        closest_to_0_cluster = np.argsort(distances[:,x])
        # print(closest_to_0_cluster)
        print(df['speech'].to_numpy()[closest_to_0_cluster[0:10]], df['speaker'].iloc[closest_to_0_cluster[0:10]])

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
