from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

def main():
    # LOAD
    iris = load_iris()

    # SPLIT
    # training features, testing features, training labels, testing(obscured) labels
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25)

    # SCALE
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # TRAIN
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train)

    # TEST
    predictions = mlp.predict(X_test)
    print(predictions)

    # ASSESS
    accuracy = mlp.score(X_test, y_test)
    print('mean accuracy:', accuracy)

    # ASSESS - CONFUSION MATRIX
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(1, 1)
    m = ax.matshow(cm)
    fig.colorbar(m)
    ax.xaxis.set_label_position('top')
    labels = list(iris['target_names'])
    ax.set(title='Confusion Matrix',
           xlabel='OUTPUT', ylabel='INPUT',
           xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels)
    plt.tight_layout()
    plt.show()

    # ASSESS - Classification report
    class_rep = classification_report(y_test,
                                      predictions)
    print('classification report:\n', class_rep)

if __name__ == '__main__':
    main()