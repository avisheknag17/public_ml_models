import pandas as pd
import pathlib as pl
import multiprocessing
import numpy as np
import sklearn.metrics as metrics

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import preprocess_string
from sklearn import utils
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV

class Doc2VecTransformer(BaseEstimator):

    def __init__(self, vector_size=100, learning_rate=0.02, epochs=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count()

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(preprocess_string(row['reviews_content']), [index]) for index, row in df_x.iterrows()]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)

        for epoch in range(self.epochs):
            model.train(utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha

        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(np.array([self._model.infer_vector(preprocess_string(row['reviews_content']))
                                     for index, row in df_x.iterrows()]))


def _read_all_reviews_():
    all_reviews = []

    for p in pl.Path('../data/txt_sentoken/pos').iterdir():
        file = open(p, 'r')
        all_reviews.append({'reviews_content': file.read(), 'category': 'positive'})
        file.close()

    for p in pl.Path('../data/txt_sentoken/neg').iterdir():
        file = open(p, 'r')
        all_reviews.append({'reviews_content': file.read(), 'category': 'negative'})
        file.close()

    all_reviews_df = pd.DataFrame(all_reviews)
    return all_reviews_df


def train_and_build_model():
    all_reviews_df = _read_all_reviews_()
    train_x_df, test_x_df, train_y_df, test_y_df = train_test_split(all_reviews_df[['reviews_content']], all_reviews_df[['category']])

    pl = Pipeline(steps=[('doc2vec', Doc2VecTransformer(vector_size=220)),
                         ('pca', PCA(n_components=100)),
                         ('logistic', LogisticRegression())
                         ])
    pl.fit(train_x_df[['reviews_content']], train_y_df[['category']])
    predictions_y = pl.predict(test_x_df[['reviews_content']])
    print('Accuracy: ', metrics.accuracy_score(y_true=test_y_df[['category']], y_pred=predictions_y))


def train_long_range_grid_search():
    all_reviews_df = _read_all_reviews_()
    train_x_df, test_x_df, train_y_df, test_y_df = train_test_split(all_reviews_df[['reviews_content']], all_reviews_df[['category']])

    pl = Pipeline(steps=[('doc2vec', Doc2VecTransformer()),
                         ('pca', PCA()),
                         ('logistic', LogisticRegression())
                         ])

    param_grid = {
        'doc2vec__vector_size': [x for x in range(100, 250)],
        'pca__n_components': [x for x in range(1, 50)]
    }
    gs_cv = GridSearchCV(estimator=pl, param_grid=param_grid, cv=5, n_jobs=-1,
                         scoring="accuracy")
    gs_cv.fit(train_x_df[['reviews_content']], train_y_df[['category']])

    print("Best parameter (CV score=%0.3f):" % gs_cv.best_score_)
    print(gs_cv.best_params_)
    predictions_y = gs_cv.predict(test_x_df[['reviews_content']])
    print('Accuracy: ', metrics.accuracy_score(y_true=test_y_df[['category']], y_pred=predictions_y))


def train_short_range_grid_search():
    all_reviews_df = _read_all_reviews_()
    train_x_df, test_x_df, train_y_df, test_y_df = train_test_split(all_reviews_df[['reviews_content']], all_reviews_df[['category']])

    pl = Pipeline(steps=[('doc2vec', Doc2VecTransformer()),
                         ('pca', PCA()),
                         ('logistic', LogisticRegression())
                         ])

    param_grid = {
        'doc2vec__vector_size': [200, 220, 250],
        'pca__n_components': [50, 75, 100]
    }
    gs_cv = GridSearchCV(estimator=pl, param_grid=param_grid, cv=3, n_jobs=-1,
                         scoring="accuracy")
    gs_cv.fit(train_x_df[['reviews_content']], train_y_df[['category']])

    print("Best parameter (CV score=%0.3f):" % gs_cv.best_score_)
    print(gs_cv.best_params_)
    predictions_y = gs_cv.predict(test_x_df[['reviews_content']])
    print('Accuracy: ', metrics.accuracy_score(y_true=test_y_df[['category']], y_pred=predictions_y))


def main():
    train_and_build_model()


if __name__== "__main__" :
    main()


