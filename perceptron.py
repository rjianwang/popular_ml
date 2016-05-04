import numpy as np

class Perceptron(object):
    def __init__(self, eta = 0.01, epochs = 50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros (1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

        return self

    def new_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        return np.where(self.new_input(x) >= 0.0, 1, -1)

def test():
    X = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])
    x = [1, 0]

    clf = Perceptron()
    clf = clf.train(X, y)
    print clf.predict(x)
    print clf.errors_

if __name__ == '__main__':
    test()
