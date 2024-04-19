import os
from tqdm import tqdm

import gensim.downloader as api
from gensim.models import KeyedVectors

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

print("Loading model...")
if os.path.exists("word2vec.model"):
    model = KeyedVectors.load("word2vec.model")
else:
    model = api.load('fasttext-wiki-news-subwords-300')
    model.save("word2vec.model")

# word_vector = model['apple']
# print(f"Vector for the word 'apple': {word_vector}")

# similarity_score = model.similarity('apple', 'orange')
# print(f"Similarity score between 'apple' and 'orange': {similarity_score}")

word_list = ['cal', 'gal', 'in', 'oz', 'aim', 'intend', 'mean', 'plan', 'girls', 'rule', 'grate', 'fleece', 'gutter', 'parachute', 'curb', 'manhole']

print("Calculating similarities...")
similarity_matrix = {}
for i, word1 in enumerate(tqdm(word_list, desc="Calculating similarities")):
    for j, word2 in enumerate(word_list):
        if i < j:  # To avoid repeating comparisons and comparing a word with itself
            similarity_score = model.similarity(word1, word2)
            similarity_matrix[f"{word1}-{word2}"] = similarity_score
            print(f"Similarity score between '{word1}' and '{word2}': {similarity_score}")


# Assuming `similarity_matrix` is filled as per your code snippet
# First, we need to convert it to a distance matrix
distance_matrix = np.zeros((len(word_list), len(word_list)))

for key, value in similarity_matrix.items():
    word1, word2 = key.split('-')
    index1, index2 = word_list.index(word1), word_list.index(word2)
    distance = 1 - value  # Convert similarity to distance
    distance_matrix[index1][index2] = distance
    distance_matrix[index2][index1] = distance

# Since the diagonal is zero (distance from word to itself), we don't need to change it

# Perform hierarchical clustering
Z = linkage(squareform(distance_matrix), method='average')

# Plot dendrogram to help decide the number of clusters
plt.figure(figsize=(10, 8))
dendrogram(Z, labels=word_list, leaf_rotation=90)
plt.title("Word Clustering Dendrogram")
plt.xlabel("Words")
plt.ylabel("Distance")
plt.show()

# Assuming we determine the optimal cluster count (let's say we pick 4 clusters)
max_clusters = 4
clusters = fcluster(Z, max_clusters, criterion='maxclust')

# Map words to their clusters
clustered_words = {}
for i, word in enumerate(word_list):
    cluster_id = clusters[i]
    if cluster_id not in clustered_words:
        clustered_words[cluster_id] = []
    clustered_words[cluster_id].append(word)

# Print the clusters
for cid, words in clustered_words.items():
    print(f"Cluster {cid}: {words}")