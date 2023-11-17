import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Read in the sentences from a text file
def clusteringData(sentences) :
    # Vectorize the sentences using TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(X)
    result=[]
    # Get the cluster labels and the corresponding sentences
    labels = kmeans.labels_
    clusters = [[] for _ in range(kmeans.n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(sentences[i])
        result.append(label)


    #print(clusters)
    # Print out the sentences in each cluster
    for i, cluster in enumerate(clusters):
        print(f'Cluster {i}:')
        for sentence in cluster:
            print(f'- {sentence}')
        print('\n')


    return(result)