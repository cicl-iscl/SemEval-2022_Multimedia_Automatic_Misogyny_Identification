import numpy as np
import pandas as pd
import pickle
import h5py
import mahotas
import cv2

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.ensemble import StackingClassifier, VotingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC


class ImageLoader(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass

    def fd_hu_moments(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def fd_haralick(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        return haralick

    def fd_histogram(self, image, mask=None):
        bins = 8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def transform(self, X):
        fixed_size = (500, 500)
        global_features = []

        for index, row in X.iterrows():
            image_path = "TRAINING/" + row["file_name"]
            image = cv2.imread(image_path)
            image = cv2.resize(image, fixed_size)

            fv_hu_moments = self.fd_hu_moments(image)
            fv_haralick = self.fd_haralick(image)
            fv_histogram = self.fd_histogram(image)

            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            global_features.append(global_feature)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_features = scaler.fit_transform(global_features)
        print("finished transforming")

        return np.array(rescaled_features)

    def fit(self, X, y=None):
        return self


class ColumnExtractor(TransformerMixin, BaseEstimator):
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X.iloc[:, self.cols]

    def fit(self, X, y=None):
        return self


if __name__ == "__main__":
    # _________________________________________________________
    # Read data
    filepath = "data.csv"
    data = pd.read_csv(filepath, sep=';')
    X = data[["Text Transcription", "maleCount", "femaleCount",
              "avgMaleAge", "avgFemaleAge", "nudity", "file_name"]]
    y = data["misogynous"]

    # _________________________________________________________
    # fit model with Text Transcription only
    pipe1 = Pipeline([
        ("tt_col", ColumnExtractor(cols=0)),
         ("cv", TfidfVectorizer()),
         ("mnb", MultinomialNB()),
    ])

    pipe2 = Pipeline([
        ("other", ColumnExtractor(cols=0)),
        ("cv", TfidfVectorizer()),
        ("mnb", GradientBoostingClassifier()),
    ])

    # fit model with age, gender, nudity
    pipe3 = Pipeline([
        ("other", ColumnExtractor(cols=range(1, 5))),
        ("mnb", MultinomialNB()),
    ])

    pipe4 = Pipeline([
        ("img", ImageLoader()),
        ("mnb", RandomForestClassifier(5000))
    ])

    # GridSearch Parameters
    grid_params = {
        "clf1__cv__ngram_range": [(1,2), (2,2), (1,1), (2,3), (1,3)],
        "clf1__cv__analyzer": ['word', 'char'],
        "clf1__cv__max_features": [None, 5000, 10000],
        "clf1__mnb__alpha": [0.5, 1.0, 3.0],
        "clf1__mnb__fit_prior": [True, False],

        "clf2__cv__ngram_range": [(1,2), (2,2), (1,1), (2,3), (1,3)],
        "clf2__cv__analyzer": ['word', 'char'],
        "clf2__cv__max_features": [None, 5000, 10000],
        "clf2__mnb__n_estimators": [100, 1000, 5000, 7000],

        "clf4__mnb__n_estimators": [100, 1000, 5000, 7000]
    }


    ensemble = GridSearchCV(estimator=VotingClassifier(estimators=[('clf1', pipe1), ('clf2', pipe2), ('clf3', pipe3), ('clf4', pipe4)]),
                            param_grid=grid_params, scoring='f1_macro', verbose=2, n_jobs=5, error_score='raise')
    ensemble.fit(X, y)

    pickle.dump(ensemble, open("taskA.sav", 'wb'))
