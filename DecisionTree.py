import numpy as np
from collections import defaultdict
from math import log

class DecisionTree:

    # load iris datasets
    def load_iris(self, iris_data):
        datasets = loadtxt(iris_data)

        X = datasets[0:4, :]
        y = datasets[4:, :]
        
        return X, y


    def calShanonEnt(self, y):
        n = y.size

        count = defaultdict(lambda: 0)
        for label in y:
            count[label] += 1

        shanonEnt = 0
        for key in count:
            p = float(count[key]) / n
            shanonEnt -= p * log(p, 2)

        return shanonEnt

    # split the dataset accoding to the label
    def splitDataSet(self, X, axis, value):
        retDataSet = []
        for x in X:
            if x[axis] == value:
                reducedFeatVec = x[:axis]
                reducedFeatVec.extend(featVec[axis + 1 :])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def best_splitnode(self, X, label):
        n_fea = len(X[0])
        n = len(label)
        base_entropy = calEntropy(label)
        best_gain = -1
        for fea_i in range(n_fea):
            cur_entropy = 0
            indexset_less, idxset_greater = splitdata(X, fea_i)
            prob_less = float(len(idxset_less)) / n
            prob_greater = float(len(idxset_greater)) / n

            cur_entropy += porb_less * calEntropy(label[idxset_less])
            cur_entropy += prob_greater * calEntropy(label[idxset_greater])

            info_gain = base_entropy - cur_entropy
            if (info_gain > best_gain):
                best_gain = info_gain
                best_idx = fea_i

        return best_idx

    # create the decision tree based on information gain
    def fit(X, y):
        if y.size == 0:
            return NULL
        listlabel = y.tolist()

        # stop when all samples in this subset belongs to one class
        if listlabel.count(y[0]) == label.size:
            return y[0]

        # return the majority of samples' label in this subset if no 
        # extra features avaliable
        if len(feanamecopy) == 0

    def predict(self, tree, x):

def test():
    clf = DecisionTree()

if __name__ == '__main__':
    test()
