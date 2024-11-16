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


def get_train_test_data(data_directory, train_percentage=0.7):
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


def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    # Create the SVM classifier
    svm_classifier = SVC(kernel="rbf")

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    # Perform prediction on the test set
    y_pred = svm_classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM accuracy: {accuracy:.3%}")

    # Plot the confusion matrix
    labels = sorted(list(set(y_train + y_test)))
    ticks = [i + 0.5 for i in range(len(labels))]
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues")
    plt.title("train")
    plt.xlabel("pred")
    plt.ylabel("actual")
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.show()


train_and_evaluate_svm(*get_train_test_data("data", 0.7))
