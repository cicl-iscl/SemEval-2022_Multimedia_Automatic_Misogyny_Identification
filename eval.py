import pickle
import pandas as pd
import numpy as np
import h5py
import cv2
import mahotas
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
            image_path = "test/" + row["file_name"]
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
    # Read data
    filepath = "test_data.csv"
    data = pd.read_csv(filepath, sep=';')
    X = data[["Text Transcription", "maleCount", "femaleCount",
              "avgMaleAge", "avgFemaleAge", "nudity", "sentiment", "file_name"]]
    # _________________________________________________________

    model = pickle.load(open("taskA12.sav", 'rb'))
    print(model)
    #print(model.best_params_)
    pred = model.predict(X)
    print(pred)
    with open("answer.txt", 'w', encoding="utf-8") as f:
        for filename, label in zip(data["file_name"], pred):
            print(filename, label)
            f.write(f"{filename}\t{label}\n")
