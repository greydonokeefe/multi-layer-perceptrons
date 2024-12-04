from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def main():

    # LOAD
    data = load_breast_cancer()
    # SPLIT
    X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                        data.target, test_size=0.25)
    # TRAIN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    # TEST
    predictions = knn.predict(X_test)
    print('predictions:', predictions)

    # ROC curve
    probs = knn.predict_proba(X_test)
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(1, 1)
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.plot([0, 1], [0, 1], 'r--')  # random guess line
    ax.set(ylabel='True Positive Rate (TPR)', xlabel='False Positive Rate(FPR)', title='Receiver Operating Characteristic (ROC) curve', xlim=[0, 1.1], ylim=[0, 1])
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()