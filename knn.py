import numpy as np
import operator

class kNN:
    def kNNClassify(self, x, X, y, k = 3):
        n_samples, n_features = X.shape

        # calculate euclidean distances
        diff = np.tile(x, (n_samples, 1)) - X
        squaredDiff = diff**2
        squaredDist = np.sum(squaredDiff, axis = 1)
        distance = squaredDist**0.5

        sortedDistIndices = np.argsort(distance)

        classCount = {}
        for i in range(k):
            voteLabel = y[sortedDistIndices[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

        sortedClassCount = sorted(classCount.iteritems(),
                key = operator.itemgetter(1), reverse = True)

        return sortedClassCount[0][0]



def test():
    knn = kNN() # create KNN object

    # create datasets
    X = np.array([[1, 1], [1, 2], [2, 2], [6, 4], [6, 5], [6, 6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    x = np.array([[1, 1]])
    k = 3

    print knn.kNNClassify(x, X, y, k)

if __name__ == '__main__':
    test()
