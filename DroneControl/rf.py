#random forest
from sklearn.ensemble import RandomForestClassifier
import os
import random
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


import matplotlib.pyplot as plt

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


def plot_confusion_matrix(conf_matrix, classes, title='Confusion Matrix'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()



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

def train_random_forest(X_train, Y_train, n_estimators=100):
    
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators)

    rf_classifier.fit(X_train, Y_train)

    return rf_classifier

def evaluate_random_forest(rf_classifier: RandomForestClassifier, X_test, Y_test):
    Y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred, labels=rf_classifier.classes_)
    return accuracy, conf_matrix

def save_random_forest(rf_classifier, file_path):
    with open(file_path, "wb") as file:
        joblib.dump(rf_classifier, file)

if __name__ == "__main__":        
    data = get_data_files("data")
    num_classes = len(data.keys())

    X_train, Y_train, X_test, Y_test = split_test_train_data(data, 0.7)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = train_random_forest(X_train, Y_train)

    accuracy, conf_matrix = evaluate_random_forest(model, X_test, Y_test)

    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    plot_confusion_matrix(conf_matrix, model.classes_)

    save_random_forest(model, "motion_classifier_rf.mdl")
