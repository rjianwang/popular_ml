import numpy as np

class SimpleLinearRegression:

    def fit(self, X, y):
        n_samples = X.shape[0]

        a = 0.0
        b = 0.0
        for i in range(0, n_samples):
            a += (X[i] - np.mean(X)) * (y[i] - np.mean(y))
            b += (X[i] - np.mean(X)) ** 2

        self.b1_ = a / float(b)
        self.b0_ = np.mean(y) - self.b1_ * np.mean(X)
        return self

    def predict(self, x):
        return self.b1_ * x + self.b0_

def test():
    X = np.array([1, 3, 2, 1, 3])
    y = np.array([14, 24, 18, 17, 27])

    clf = SimpleLinearRegression()
    clf = clf.fit(X, y)
    print clf.predict(1)
    print clf.b1_, clf.b0_

if __name__ == '__main__':
    test()
