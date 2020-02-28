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
from sklearn.metrics import silhouette_score

df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
df = unusable_rows(df)
df = get_candidates(df)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['speech'])
features = vectorizer.get_feature_names()
kmeans = KMeans(n_clusters=7)
kmeans.fit(X)

top_centroids = kmeans.cluster_centers_.argsort()[:,-1:-11:-1]
print("\n3) top features (words) for each cluster:")
for num, centroid in enumerate(top_centroids):
    print("%d: %s" % (num, ", ".join(features[i] for i in centroid)))


print("\n5) random sample of titles in each cluster")
assigned_cluster = kmeans.transform(X).argmin(axis=1)
candidates_list = {}
print(silhouette_score(X, assigned_cluster))

for times_run in range(200):
    for i in range(kmeans.n_clusters):
        if i not in candidates_list.keys():
            candidates_list[i] = {}
        else:
            candidates_list[i] = candidates_list[i]

        cluster = np.arange(0, X.shape[0])[assigned_cluster==i]
        sample_articles = np.random.choice(cluster, 10, replace=False)

        # print("cluster %d:" % i)
        for article in sample_articles:

            # print("    %s" % df.iloc[article]['speaker'])

            if df.iloc[article]['speaker'] not in candidates_list[i].keys():
                candidates_list[i][df.iloc[article]['speaker']] = 1/2000
            else:
                candidates_list[i][df.iloc[article]['speaker']] += 1/2000

# for x in candidates_list.keys():
#     print('Cluster {}'.format(x))
#     print('Bernie:' + ' ' + str(round(candidates_list[x]['Bernie Sanders'],2)*100) + '%')
#     print('Elizabeth:' + ' ' + str(round(candidates_list[x]['Elizabeth Warren'],2)*100) + '%')
#     print('Joe:' + ' ' + str(round(candidates_list[x]['Joe Biden'],2)*100) + '%')
#     print('Pete:' + ' ' + str(round(candidates_list[x]['Pete Buttigieg'],2)*100) + '%')
#     print('Amy Klobuchar:' + ' ' + str(round(candidates_list[x]['Amy Klobuchar'],2)*100) + '%')
#     print('\n')



# # reduce the features to 2D
# pca = PCA(n_components=2, random_state=0)
# reduced_features = pca.fit_transform(X.toarray())
# # reduce the cluster centers to 2D
# reduced_cluster_centers = pca.transform(kmeans.cluster_centers_)
# plt.scatter(reduced_features[:,0], reduced_features[:,1], c=kmeans.predict(X), s=4)
# plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:,1], marker='x', s=150, c='b')
# plt.show()
