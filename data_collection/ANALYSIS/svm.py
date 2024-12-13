import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
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

        # Keep only accelerometer and gyroscope signals
        data = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].values.astype(np.float32)

        # Normalize data
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

        # Populate lists with normalized data and labels
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
    # Create the SVM classifier
    svm_classifier = SVC(kernel="rbf")

    # Train the classifier
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


def train_and_save_svm(data):
    train_data, train_labels, a, b = split_test_train_data(data, 1)
    model = SVC(C=10, gamma=0.01, kernel="rbf", class_weight='balanced')
    model.fit(train_data, train_labels)
    
    # model = KNeighborsClassifier(n_neighbors=5)
    # model.fit(train_data, train_labels)
    with open("motion_classifier.pkl", "wb") as file:
        joblib.dump(model, file)


def load_svm_from_file(file_path="./motion_classifier.mdl"):
    with open(file_path, "rb") as file:
        return joblib.load(file)

if __name__ == "__main__":
    # load_and_evaluate()
    data = get_data_files("data")
    train_and_save_svm(data)
    exit()
    num_classes = len(data.keys())

    X_train, Y_train, X_test, Y_test = split_test_train_data(data, 0.7)
    model = train_svm(X_train, Y_train)

    accuracy, conf_matrix = evaluate_svm(model, X_test, Y_test)

    save_svm(model, "motion_classifier.mdl")
