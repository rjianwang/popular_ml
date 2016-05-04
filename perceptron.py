import numpy as np

class Perceptron(object):
    def __init__(self, eta = 0.01, epochs = 50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros (1 + X.shape[1])

        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update

        return self

    def predict(self, x):
        res = np.dot(x, self.w_[1:]) + self.w_[0]
        return np.where(res >= 0.0, 1, -1)

def test():
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    x = [1, 0]

    clf = Perceptron(eta = 1)
    clf = clf.train(X, y)
    print "class:   ", clf.predict(x)
    print "weights: ", clf.w_[1:]
    print "bias:    ", clf.w_[0]

if __name__ == '__main__':
    test()
