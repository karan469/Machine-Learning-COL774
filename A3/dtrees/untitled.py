"""Implementation of the CART algorithm to train decision tree classifiers."""
import numpy as np
import pandas as pd
from xclib.data import data_utils
import matplotlib.pylab as plt
from random import seed
from random import randrange
import math

# paths
data_folder_path = './data/'
train_x_path = data_folder_path + 'small_x.txt'
train_y_path = data_folder_path + 'small_y.txt'
train_size = (64713, 482)
# train_size = (14,4)
test_x_path = data_folder_path + 'small_x.txt'
test_y_path = data_folder_path + 'small_y.txt'
test_size = (21571, 482)
# test_size = (14,4)
val_x_path = data_folder_path + 'valid_x.txt'
val_y_path = data_folder_path + 'valid_y.txt'

def load_y(path, size):
    y = np.ones(size)
    f = open(path)
    cnt = 0
    for x in f:
        y[cnt] = int(x)
        cnt += 1
    return y

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = int(classes[i - 1])
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] <= thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


if __name__ == "__main__":
    import sys
    from sklearn.datasets import load_iris

    train_x = data_utils.read_sparse_file(train_x_path)
    train_y = load_y(train_y_path, train_x.shape[0])

    test_x = data_utils.read_sparse_file(test_x_path)
    test_y = load_y(test_y_path, test_x.shape[0])

    i = 0
    train_data = np.zeros(shape=train_size)
    for x in train_x.toarray():
        train_data[i] = x
        i += 1
    i = 0
    test_data = np.zeros(shape=test_size)
    for x in test_x.toarray():
        test_data[i] = x
        i += 1
    print('halfway')
#     dataset = load_iris()
    X, y = test_data, test_y  # pylint: disable=no-member
    clf = DecisionTreeClassifier(max_depth=20)
    clf.fit(X, y)
    
#     print(clf.predict(test_data))
    
    i = 0
    cnt = 0
    for x in clf.predict(test_data):
        if(x==test_y[i]):
            cnt += 1
        i+=1
    print(cnt/i)
