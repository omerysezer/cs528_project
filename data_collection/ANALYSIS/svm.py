import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

NUM_ITERATIONS = 15

def get_data_files(data_directory):
    files = [
        os.path.join(data_directory, entry)
        for entry in os.listdir(data_directory)
        if os.path.isfile(os.path.join(data_directory, entry))
    ]

    features = {}
    for file in files:
        df = pd.read_csv(file, sep=",")

        data = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values.astype(np.float32)


        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

        label = os.path.basename(file).split("_")[0]
        if label not in features:
            features[label] = []
        features[label].append(data.flatten())

    return features

def split_test_train_data(features, train_percentage):
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    for label in features:
        random.shuffle(features[label])
        labels = [label for _ in range(len(features[label]))]
        num_train = int(round(len(features[label]) * train_percentage))
        train_features += features[label][:num_train]
        train_labels += labels[:num_train]
        test_features += features[label][num_train:]
        test_labels += labels[num_train:]

    return train_features, train_labels, test_features, test_labels

def train_svm(X_train, Y_train):
    svm_classifier = SVC(kernel="rbf")

    svm_classifier.fit(X_train, Y_train)

    return svm_classifier

def evaluate_svm(svm_classifier: SVC, X_test, Y_test):
    Y_pred = svm_classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred, labels=svm_classifier.classes_)
    return accuracy, conf_matrix

def save_svm(svm_classifier, file_path):
    with open(file_path, "wb") as file:
        joblib.dump(svm_classifier, file)

if __name__ == "__main__":        
    data = get_data_files("data")
    num_classes = len(data.keys())

    X_train, Y_train, X_test, Y_test = split_test_train_data(data, 0.7)
    model = train_svm(X_train, Y_train)

    accuracy, conf_matrix = evaluate_svm(model, X_test, Y_test)

    save_svm(model, "motion_classifier.mdl")
