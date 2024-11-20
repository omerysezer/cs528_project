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
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    return accuracy, conf_matrix

def train_and_evaluate_svm_forest(data, num_models=5):
    models = []
    train_x, train_y, test_x, test_y = split_test_train_data(data, 0.7)
    subset_size = int(0.7*len(train_x))
    for _ in range(num_models):    
        indices = random.sample(range(len(train_x)), subset_size)
        model_x = [train_x[index] for index in indices]
        model_y = [train_y[index] for index in indices]
        model = SVC(kernel="rbf")
        model.fit(model_x, model_y)
        models.append(model)
    
    y_pred = []
    for instance, label in zip(test_x, test_y):
        counts = {}
        for model in models:
            label = model.predict(instance.reshape(1, -1))[0]
            counts[label] = counts.get(label, 0) + 1
        max_count = max(counts.values())
        candidates = [word for word, count in counts.items() if count == max_count]
        chosen_label = random.choice(candidates)
        y_pred.append(chosen_label)
    
    accuracy = accuracy_score(test_y, y_pred)
    print(f"SVM accuracy: {accuracy:.3%}")

    # Plot the confusion matrix
    labels = sorted(list(set(train_y + test_y)))
    conf_matrix = confusion_matrix(test_y, y_pred, labels=labels)
    return accuracy, conf_matrix

if __name__ == "__main__":        
    data = get_data_files("data")
    num_classes = len(data.keys())
    avg_accuracy = 0
    avg_conf_matrix = np.zeros(shape=(len(data.keys()), len(data.keys())))
    labels = list(data.keys())

    avg_accuracy, avg_conf_matrix = train_and_evaluate_svm_forest(data, 10)
    # exit()

    # for _ in range(NUM_ITERATIONS):
    #     split_data = split_test_train_data(data, 0.7)
    #     acc, conf_matrix = train_and_evaluate_svm(*split_data)  
    #     avg_accuracy += acc
    #     avg_conf_matrix += conf_matrix
    
    # avg_conf_matrix /= NUM_ITERATIONS
    # avg_accuracy /= NUM_ITERATIONS
    print(f"Avg accuracy over {NUM_ITERATIONS} runs = {100*avg_accuracy:.3f}%")
    ticks = [i + 0.5 for i in range(len(labels))]
    sns.heatmap(avg_conf_matrix, annot=True, cmap="Blues")
    plt.title("train")
    plt.xlabel("pred")
    plt.ylabel("actual")
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.show()

        