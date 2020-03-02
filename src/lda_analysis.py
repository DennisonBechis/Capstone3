import re
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import GridSearchCV, train_test_split
from debate_cleaning import *
from sklearn.feature_extraction.text import TfidfVectorizer

pd.options.display.max_columns = 40

stop_words = ENGLISH_STOP_WORDS.union({'redirects', 'mentions', 'locations','ve','know','don','way','think','going','just',
                                        'said','got','like','need','say','ll','america','want','sure','make','come','right',
                                        'let','did','look', 'actually','lot','does','people','fact','time','president',
                                        'country','united','states','american','trump'})

class TopicModeler(object):

    def __init__(self, model, vectorizer, distance_func=cosine_distances):
        self.model = model
        self.text = None
        self.titles = None
        self.vectorizer = vectorizer
        self.feature_names = None
        self.doc_probs = None
        self.distance_func = distance_func
        self.word_vec = None

    def set_feature_names(self):

        self.feature_names = np.array(self.vectorizer.get_feature_names())

    def vectorize(self):

        print('Vectorizing...')
        return self.vectorizer.fit_transform(self.text)

    def fit(self, text, titles=None):

        self.text = text
        self.word_vec = self.vectorize()
        print('Fitting...')
        self.model.fit(self.word_vec)
        self.set_feature_names()
        self.titles = titles

    def top_topic_features(self, num_features=10):

        sorted_topics = self.model.components_.argsort(axis=1)[:, ::-1][:, :num_features]

        return self.feature_names[sorted_topics]

    def predict_proba(self, text):

        if type(text) == str:
            text = [text]

        vec_text = self.vectorizer.transform(text)
        return self.model.transform(vec_text)

    def sort_by_distance(self, doc_index, doc_probs):

        distances = self.distance_func(doc_probs[doc_index, np.newaxis], doc_probs)
        return distances.argsort()

    def find_closest_document_titles(self, sorted_indices, num_documents=10):

        name_array = self.titles.iloc[sorted_indices.ravel()][:num_documents]
        return {name_array.iloc[0]: name_array.iloc[1:].tolist()}

    def find_article_idx(self, article_title):

        return self.titles[self.titles == article_title].index[0]

    def top_closest_articles(self, article_title, num_documents=10):

        doc_index = self.find_article_idx(article_title)
        doc_probs = self.predict_proba(self.text)
        article_similarities = self.sort_by_distance(doc_index, doc_probs)
        return self.find_closest_document_titles(article_similarities, num_documents)

    def grid_search_lda(self, params):

        X_train, X_test = train_test_split(self.word_vec, test_size=0.25)
        lda_cv = GridSearchCV(self.model, param_grid=params, n_jobs=-1,  verbose=1)
        lda_cv.fit(X_train)

        print(pd.DataFrame.from_dict(lda_cv.cv_results_))
        print('Test Score:', lda_cv.score(X_test))
        return lda_cv

    def load_model(self, filepath):

        self.model = joblib.load(filepath)

    def load_vectorizer(self, filepath):

        self.tf_vectorizer = joblib.load(filepath)

    def save_model(self, filepath):

        joblib.dump(self.model, filepath)

    def save_vectorizer(self, filepath):

        joblib.dump(self.vectorizer, filepath)

def main():
    df = pd.read_csv("/Users/bechis/dsi/repo/Capstone3/data/debate_transcripts.csv", encoding= 'unicode_escape')
    df = unusable_rows(df)
    df = get_candidates(df)
    df['still_running'] = df.apply(lambda row: assign_dropped(row[2]), axis=1)

    lda = LatentDirichletAllocation(n_components=10, learning_offset=50., verbose=1,
                                    doc_topic_prior=0.7, topic_word_prior=0.7, n_jobs=-1, learning_method='online')

    tf_vectorizer = CountVectorizer(max_df=0.85, min_df=2, max_features=1000, stop_words=stop_words)

    td_idf = TfidfVectorizer(stop_words = stop_words)

    tm = TopicModeler(lda, td_idf)
    tm.fit(df.speech, titles=df.speaker)

if __name__ == '__main__':
    main()
