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

plt.style.use('classic')

df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
df = unusable_rows(df)
df = get_candidates(df)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['speech'])
features = vectorizer.get_feature_names()

x_list = [p for p in range(2,15)]
y_list = []

for x in range(2,15):
    kmeans = KMeans(n_clusters=x)
    kmeans.fit(X)

    assigned_cluster = kmeans.transform(X).argmin(axis=1)
    m = silhouette_score(X, assigned_cluster)

    y_list.append(m)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1,1,1)

ax.plot(x_list, y_list)
ax.set_xlabel('k')
ax.set_ylabel('Silhouette score')
ax.set_title('Plot of Silhouette Scores')
name = 'Silhouette_plot'
plt.savefig(name +'.png', transparent = True)
plt.show()
