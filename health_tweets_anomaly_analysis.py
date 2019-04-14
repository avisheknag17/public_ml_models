from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import scikitplot.metrics as m_plot

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator
from sklearn.metrics.cluster.unsupervised import silhouette_samples, silhouette_score

from gensim.parsing.preprocessing import preprocess_string
from sklearn import utils
from tqdm import tqdm
from pyod.models.lof import LOF

from nltk.cluster import KMeansClusterer, cosine_distance

import multiprocessing
import numpy as np

from matplotlib import pyplot as plt

# Author Avishek Nag

class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1

    def fit(self, x, y=None):
        tagged_x = [TaggedDocument(preprocess_string(item), [index]) for index, item in enumerate(x)]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            model.train(utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, x):
        arr = np.array([self._model.infer_vector(preprocess_string(item))
                                     for index, item in enumerate(x)])
        return arr


class LOFDetectionTransformer(BaseEstimator):

    def __init__(self):
        self._model = None

    def fit(self, x, y=None):
        self._model = LOF(metric='cosine')
        self._model.fit(x)
        return self

    def transform(self, x):
        return self._model.decision_scores_


class OptimalKMeansTextsClusterTransformer(BaseEstimator):

    def __init__(self, min_k, max_k):
        self.min_k = min_k
        self.max_k = max_k

    def fit(self, x, y=None):
        return self

    def _silhouette_score_with_k_(self, vectors, k):
        clusterer = KMeansClusterer(num_means=k, distance=cosine_distance, repeats=3)
        cluster_labels = clusterer.cluster(vectors=vectors, assign_clusters=True, trace=False)
        silhouette_score_k = silhouette_score(X=vectors, labels=cluster_labels, metric='cosine')
        return k, silhouette_score_k

    def _determine_k_for_max_silhouette_score_(self, process_responses):
        max_silhoutte_score = -100.0
        optimal_k = 2
        for index, process_response in enumerate(process_responses):
            current_k, silhouette_score_k = process_response.get()
            print('Silhoutte Score: ', silhouette_score_k, ' for k', current_k)
            if silhouette_score_k > max_silhoutte_score:
                max_silhoutte_score = silhouette_score_k
                optimal_k = current_k

        return optimal_k

    def transform(self, x):
        range_of_k = [x for x in range(self.min_k, self.max_k)]
        clusterer_pool = multiprocessing.Pool(processes=len(range_of_k))
        clusterer_process_responses = []
        for k in range_of_k:
            clusterer_process_responses.append(clusterer_pool.apply_async(self._silhouette_score_with_k_, args=(x, k,)))

        optimal_k = self._determine_k_for_max_silhouette_score_(process_responses=clusterer_process_responses)
        clusterer_pool.close()
        print("Optimal k: ", optimal_k)
        optimal_clusterer = KMeansClusterer(num_means=optimal_k, distance=cosine_distance, repeats=3)
        optimal_cluster_labels = optimal_clusterer.cluster(vectors=x, assign_clusters=True, trace=False)
        return x, optimal_cluster_labels


def _read_all_health_tweets():

    all_tweets = {}
    file = open('../../data/Health-Tweets/nytimeshealth.txt', 'r')
    lines = file.readlines()
    for index, line in enumerate(lines):
        parts = line.split(sep='|', maxsplit=2)
        tweet = "".join(parts[2:len(parts)])
        all_tweets[index] = tweet

    file.close()
    return all_tweets


def analyze_tweets_pca(n_pca_components):
    tweets_dict = _read_all_health_tweets()
    tweets = tweets_dict.values()
    doc2vectors = Pipeline(steps=[('doc2vec', Doc2VecTransformer())]).fit(tweets).transform(tweets)
    pca = PCA(n_components=n_pca_components)
    pca_vectors = pca.fit_transform(doc2vectors)
    print('All Principal Components ..')
    print(pca_vectors)
    for index, var in enumerate(pca.explained_variance_ratio_):
        print("Explained Variance ratio by Principal Component ", (index+1), " : ", var)


def plot_tweets_k_means_clusters_with_anomalies(pca_vectors, cluster_labels, pca_vectors_anomalies):
    pca_vectors_anomalies_x = []
    pca_vectors_anomalies_y = []

    for pca_vectors_elem in pca_vectors_anomalies:
        pca_vectors_anomalies_x.append(pca_vectors_elem[1])
        pca_vectors_anomalies_y.append(pca_vectors_elem[0])

    plt.xlabel('Principal Component 2')
    plt.ylabel('Principal Component 1')
    plt.title('Kmeans Cluster of Tweets')

    plt.scatter(x=pca_vectors[:, 1], y=pca_vectors[:, 0], c=cluster_labels)
    plt.scatter(x=pca_vectors_anomalies_x, y=pca_vectors_anomalies_y, marker='^')
    plt.show()


def determine_anomaly_tweets_k_means(top_n):
    tweets_dict = _read_all_health_tweets()
    tweets = tweets_dict.values()
    pl = Pipeline(steps=[('doc2vec', Doc2VecTransformer()),
                         ('pca', PCA(n_components=2)),
                         ('kmeans', OptimalKMeansTextsClusterTransformer(min_k=2, max_k=5))])
    pl.fit(tweets)
    pca_vectors, cluster_labels = pl.transform(tweets)
    silhouette_values = silhouette_samples(X=pca_vectors, labels=cluster_labels, metric='cosine')
    tweet_index_silhouette_scores = []
    absolute_silhouette_scores_tweet_index = []

    for index, sh_score in enumerate(silhouette_values):
        absolute_silhouette_scores_tweet_index.append((abs(sh_score), index))
        tweet_index_silhouette_scores.append((index, sh_score))

    sorted_scores = sorted(absolute_silhouette_scores_tweet_index, key=sort_key)

    top_n_silhouette_scores = []
    pca_vectors_anomalies = []
    print("Top ", top_n, " anomalies")
    for i in range(top_n):
        abs_sh_score, index = sorted_scores[i]
        index_1, sh_score = tweet_index_silhouette_scores[index]
        top_n_silhouette_scores.append((index, sh_score))
        print(tweets_dict[index])
        print('PCA vector', pca_vectors[index])
        pca_vectors_anomalies.append(pca_vectors[index])
        print('Silhouette Score: ', sh_score)
        print("..................")

    plot_tweets_k_means_clusters_with_anomalies(pca_vectors=pca_vectors, pca_vectors_anomalies=pca_vectors_anomalies,
                                                cluster_labels=cluster_labels)
    plot_scatter_silhouette_scores(top_n_silhouette_scores=top_n_silhouette_scores,
                                   tweets_dict=tweets_dict,
                                   silhouette_score_per_tweet=tweet_index_silhouette_scores)


def sort_key(t):
    return t[0]


def plot_scatter_silhouette_scores(top_n_silhouette_scores, tweets_dict, silhouette_score_per_tweet):
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle('Silhouette Scores vs Tweets')
    sub_plot_scatter_silhouette_scores(ax=ax1, top_n_silhouette_scores=top_n_silhouette_scores,
                                       tweets_dict=tweets_dict,
                                       silhouette_score_per_tweet=silhouette_score_per_tweet,
                                       with_annotation=False)

    sub_plot_scatter_silhouette_scores(ax=ax2, top_n_silhouette_scores=top_n_silhouette_scores,
                                       tweets_dict=tweets_dict,
                                       silhouette_score_per_tweet=silhouette_score_per_tweet,
                                       with_annotation=True)
    plt.show()


def sub_plot_scatter_silhouette_scores(ax,top_n_silhouette_scores, tweets_dict, silhouette_score_per_tweet, with_annotation):
    ax.set(xlabel='Tweet Index', ylabel='Silhouette Score')
    ax.scatter(*zip(*silhouette_score_per_tweet))
    ax.scatter(*zip(*top_n_silhouette_scores), edgecolors='red')

    if with_annotation:
        for (index, score) in top_n_silhouette_scores:
            ax.annotate(tweets_dict[index], xy=(index, score), xycoords='data')


def determine_anomaly_tweets_lof(top_n):
    tweets_dict = _read_all_health_tweets()
    tweets = tweets_dict.values()
    pl = Pipeline(steps=[('doc2vec', Doc2VecTransformer()),
                         ('pca', PCA(n_components=2)),
                         ('lof', LOFDetectionTransformer())])
    pl.fit(tweets)
    scores = pl.transform(tweets)
    tweet_index_decision_scores = []
    decision_scores_tweet_index = []

    for index, score in enumerate(scores):
        decision_scores_tweet_index.append((score, index))
        tweet_index_decision_scores.append((index, score))

    sorted_scores = sorted(decision_scores_tweet_index, key=sort_key, reverse=True)

    top_n_tweet_index_decision_scores = []
    print("Top ", top_n, " anomalies")
    for i in range(top_n):
        score, index = sorted_scores[i]
        top_n_tweet_index_decision_scores.append((index, score))
        print(tweets_dict[index])
        print('Decision Score: ', score)
        print("..................")

    plot_scatter_lof(tweets_dict=tweets_dict, tweet_index_decision_scores=tweet_index_decision_scores,
                     top_n_tweet_index_decision_scores=top_n_tweet_index_decision_scores)


def plot_scatter_lof(tweets_dict, tweet_index_decision_scores, top_n_tweet_index_decision_scores):
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    fig.suptitle('Decision scores vs Tweets')

    sub_plot_scatter_lof(ax=ax1, tweets_dict=tweets_dict,
                         tweet_index_decision_scores=tweet_index_decision_scores,
                         top_n_tweet_index_decision_scores=top_n_tweet_index_decision_scores, with_annotation=False)
    sub_plot_scatter_lof(ax=ax2, tweets_dict=tweets_dict,
                         tweet_index_decision_scores=tweet_index_decision_scores,
                         top_n_tweet_index_decision_scores=top_n_tweet_index_decision_scores, with_annotation=True)
    plt.show()


def sub_plot_scatter_lof(ax, tweets_dict, tweet_index_decision_scores, top_n_tweet_index_decision_scores,
                     with_annotation=True):
    ax.set(xlabel='Tweet Index', ylabel='Decision Score')
    ax.scatter(*zip(*tweet_index_decision_scores))
    ax.scatter(*zip(*top_n_tweet_index_decision_scores), edgecolors='red')

    if with_annotation:
        for (index, score) in top_n_tweet_index_decision_scores:
            ax.annotate(tweets_dict[index], xy=(index, score), xycoords='data')

