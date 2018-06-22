import pickle
import random
import tempfile
import time

from numpy import arange
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def run():
    test_data_home = tempfile.mkdtemp()
    #mnist = fetch_mldata('MNIST Original', data_home=test_data_home)
    #pickle.dump(mnist, open('mnist.pkl', 'wb'))
    mnist = pickle.load(open('mnist.pkl', 'rb'), encoding='utf-8')
    #mnist = pickle._Unpickler(open('mnist.pkl', 'r'))
    #mnist.encoding = 'latin1'
    # mnist.data, mnist.target = shuffle(mnist.data, mnist.target)
    # print mnist.data.shape
    # Trunk the data
    n_train = 60000
    n_test = 10000

    # Define training and testing sets
    indices = arange(len(mnist.data))
    random.seed(0)
    # train_idx = random.sample(indices, n_train)
    # test_idx = random.sample(indices, n_test)
    train_idx = arange(0, n_train)
    test_idx = arange(n_train + 1, n_train + n_test)

    X_train, y_train = mnist.data[train_idx], mnist.target[train_idx]
    X_test, y_test = mnist.data[test_idx], mnist.target[test_idx]


    # Apply a learning algorithm
    print("Applying a learning algorithm...")
    clf = RandomForestClassifier(n_estimators=10, n_jobs=2)
    clf.fit(X_train, y_train)

    # Make a prediction
    print("Making predictions...")
    y_pred = clf.predict(X_test)

    # print y_pred

    # Evaluate the prediction
    print("Evaluating results...")
    print("Precision: \t", metrics.precision_score(y_test, y_pred, average=None))
    print("Recall: \t", metrics.recall_score(y_test, y_pred, average=None))
    print("F1 score: \t", metrics.f1_score(y_test, y_pred, average=None))
    print("Mean accuracy: \t", clf.score(X_test, y_test))


if __name__ == "__main__":
    start_time = time.time()
    results = run()
    end_time = time.time()
    print("Overall running time:", end_time - start_time)
