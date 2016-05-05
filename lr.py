#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class LogisticRegress:

    def __init__(self, alpha = 0.1, bias = 1.0, num_it = 500):
        self.alpha = alpha
        self.b_ = bias
        self.num_it = num_it

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))

    def SGA(self, X, y):
        n_samples, n_features = X.shape
        self.w_ = np.ones(n_features + 1)

        for i in xrange(self.num_it):
            index = range(n_samples)
            for j in range(n_samples):
                self.alpha = 4 / (1.0 + i + j) + 0.01
                rand_idx = int(np.random.uniform(0, len(index)))

                z = self.sigmoid(np.sum(X[rand_idx] * self.w_[0:-1]) + self.w_[-1] * self.b_)
                error = y[rand_idx] - z
                self.w_[0:-1] = self.w_[0:-1] + self.alpha * error * X[rand_idx]
                self.w_[-1] = self.w_[-1] + self.alpha * error * self.b_

                del(index[rand_idx])

        return self.w_

    def fit(self, X, y):
        self.SGA(X, y)
        return self

    def predict(self, x):
        p = self.sigmoid(np.sum(x * self.w_[0:-1] + self.w_[-1] *self.b_))
        if p > 0.5:
            return 1
        return 0

def test():
    data = np.loadtxt('testSet.txt')
    X = data[:, 0:-1]
    y = data[:, -1]
    
    clf = LogisticRegress(num_it = 150)
    clf = clf.fit(X, y)

    correct = 0;
    for i in range(X.shape[0]):
        result = clf.predict(X[i])
        if result == y[i]:
            correct += 1

    print("%d%%" % correct)

if __name__ == '__main__':
    test()
