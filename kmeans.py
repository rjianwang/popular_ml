import numpy as np

# Euclidean distance
def dist(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

# k-means cluster algorithm
def kmeans(X, k = 3, max_it = None):
    n_samples, n_features = X.shape

    # initialize k random centroids
    centroids = X[np.random.randint(n_samples, size = k), :]

    # iterate while cluster changed
    cluster = np.mat(np.zeros((n_samples, 2)))
    iterations = 0
    changed = True
    while changed:
        iterations += 1
        changed = False

        # stop if iteration times is great than max_it
        if max_it != None and iterations > max_it:
            break

        # for each sample
        for i in xrange(n_samples):
            min_dist = 100000.0
            index = 0
            for j in range(k):
                distance = dist(centroids[j, :], X[i, :])
                if distance < min_dist:
                    min_dist = distance
                    index = j

            if cluster[i, 0] != index:
                changed = True
                cluster[i, :] = index, min_dist
        
        # update centroids
        for j in range(k):
            pointsInCluster = X[np.nonzero(cluster[:, 0].A == j)[0]]
            centroids[j, :] = np.mean(pointsInCluster, axis = 0)

    return cluster, centroids

def test():
    X = np.array([[1, 1], [1, 2], [4, 3], [4, 5]])
    k = 2
    cluster, centroids = kmeans(X, k)
    print cluster
    print centroids

if __name__ == '__main__':
    test()
