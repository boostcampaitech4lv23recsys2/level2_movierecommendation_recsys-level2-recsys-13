import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

ratings = pd.read_csv('/opt/ml/input/data/data2/interactions.csv')
genre = pd.read_csv('/opt/ml/input/data/data2/genres.csv')

genre2idx = {v: k for k, v in enumerate(genre['genre'].unique())}
idx2genre = {k: v for k, v in enumerate(genre['genre'].unique())}

movie_genres_list = genre.groupby('item')['genre'].apply(list)
user_movies_list = ratings.groupby('user')['item'].apply(list)

user_genre = []
for u, movies in tqdm(user_movies_list.iteritems(), total=len(user_movies_list)):
    user_genre_vecs = np.zeros((len(movies), genre['genre'].nunique())) 
    for i, movie in enumerate(movies): 
        movie_genres = movie_genres_list[movie] 
        user_genre_vecs[i, [genre2idx[movie_genre] for movie_genre in movie_genres]] = 1. 
    user_genre_vec = np.sum(user_genre_vecs, axis=0)
    user_genre.append([u] + user_genre_vec.tolist())

df = pd.DataFrame(user_genre, columns=['user'] + [idx2genre[i] for i in range(genre['genre'].nunique())])
df_nouser = df.drop(['user'], axis=1)
df_normalized = df_nouser.div(df_nouser.sum(axis=1), axis=0)

pca = PCA(n_components=2)
printcipalComponents = pca.fit_transform(df_normalized)
principalDf = pd.DataFrame(data=printcipalComponents, columns = ['pca1', 'pca2'])

kmeans_cluster = KMeans(n_clusters=4, random_state=42)
kmeans_cluster.fit(df_normalized)
cluster = pd.DataFrame(kmeans_cluster.labels_)

data = pd.concat([principalDf, cluster], axis=1)
data.columns = ['pca1', 'pca2', 'cluster']

user_id = df[['user']]
df_final = pd.concat([user_id, cluster], axis=1)
df_final.columns = ['user', 'cluster']

df_final.to_csv('genre_clustering.csv', index=False)
    