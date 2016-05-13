#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import copy
import numpy as np
import matplotlib.pyplot as plt

class GMM:

    def __init__(self, sigma, mu1, mu2, K, N):
        self.sigma = sigma
        self.mu1 = mu1
        self.mu2 = mu2

        self.K = K
        self.N = N

        self.X = np.zeros((1, N))
        self.mu = np.random.random(2)
        self.expectations = np.zeros((N, K))

        for i in xrange(0, N):
            if np.random.random > 0.5:
                self.X[0, i] = np.random.normal() * sigma + mu1
            else:
                self.x[0:i] = np.random.normal() * sigma + mu2

    def e_step(self):
        for i in xrange(0, self.N):
            Denom = 0
            for j in xrange(0, self.K):
                Denom += math.exp((-1 / (2 * (float(self.sigma**2)))) * 
                        (float(self.X[0, i] - self.mu[j]))**2)
            for j in xrange(0, self.K):
                Numer = math.exp((-1 / (2 * (float(self.sigma**2)))) * (float(self.X[0, i] - self.mu[j]))**2)

    def m_step(self):
        for j in xrange(0, self.K):
            Numer = 0
            Denom = 0
            for i in xrange(0, self.N):
                Numer += self.expectations[i, j] * self.X[0, i]
                Denom += self.expectations[i, j]
            self.mu[j] = Numer / Denom

    def run(self, num_iterations = 1000, Epsilon = 0.0001):
        for i in range(num_iterations):
            old_mu = copy.deepcopy(self.mu)
            self.e_step()
            self.m_step()
            if sum(abs(self.mu - old_mu)) < Epsilon:
                break;

if __name__ == '__main__':
    clf = GMM(6, 40, 20, 2, 1000)
    clf.run()
    plt.hist(X[0, :], 50)
    plt.show()
