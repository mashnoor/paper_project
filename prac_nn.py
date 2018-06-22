import pickle
import time

from numpy import arange
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

def run():

    mnist = pickle.load(open('mnist.pkl', 'rb'), encoding='utf-8')

    n_train = 60000
    n_test = 10000


    train_idx = arange(0, n_train)
    test_idx = arange(n_train + 1, n_train + n_test)

    X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
    X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]


    # Apply a learning algorithm
    print("Applying a learning algorithm...")
    clf = MLPClassifier(solver='sgd', alpha=1e-5, activation='logistic',
                        hidden_layer_sizes=(30, 30, 30))
    clf.fit(X_train, y_train)

    # Make a prediction
    print("Making predictions...")
    y_pred = clf.predict(X_test)


    # Evaluate the prediction
    print("Evaluating results...")
    print("Precision: \t", metrics.precision_score(y_test, y_pred, average=None))
    print("Recall: \t", metrics.recall_score(y_test, y_pred, average=None))
    print("F1 score: \t", metrics.f1_score(y_test, y_pred, average=None))
    print("Mean accuracy: \t", clf.score(X_test, y_test))
    print("Confusion Matrix: \t", metrics.confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    start_time = time.time()
    results = run()
    end_time = time.time()
    print("Overall running time:", end_time - start_time)
