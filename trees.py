import numpy as np
from collections import defaultdict
from math import log

class DecisionTree:


    # load iris datasets
    def load_iris(self, iris_data):
        datasets = loadtxt(iris_data)

        X = datasets[0:4, :]
        y = datasets[4:, :]
        labels = ['sepal_width', 'sepal_length', 'petal_length', 
                'petal_width']
        return X, y, labels

    # calculate Shannon entropy
    def cal_shannon_ent(self, y):
        n = y.size

        count = defaultdict(lambda: 0)
        for label in y:
            count[label] += 1

        shannon_ent = 0
        for key in count:
            p = float(count[key]) / n
            shannon_ent -= p * log(p, 2)

        return shannon_ent

    # split the dataset accoding to the axis which value is 'value'
    def split_data_set(self, X, y, axis, value):
        ret_sub_X = []
        ret_sub_y = []
        for i in range(len(X)):
            if x[i][axis] == value:
                reduced_x = x[:axis]
                reduced_x.extend(x[axis + 1 :])
                ret_sub_X.append(reduced_x)
                ret_sub_y.extend(y[i])
        return ret_sub_X, ret_sub_y

    # select the best feature to split
    def split_by_x(self, X, y):
        n_feature = len(X[0])
        base_entropy = self.cal_shannon_ent(y)
        best_info_gain = 0.0
        best_feature = -1

        for i in range(n_feature):
            feature_list = [x[i] for x in X]
            unique_val = set(feature_list)
            cur_entropy = 0.0
            for value in unique_val:
                sub_X, sub_y = self.split_data_set(X, y, i, value)
                prob = len(sub_X) / float(len(X))
                cur_entropy += prob * self.cal_shannon_ent(sub_y)

            info_gain = base_entropy - cur_entropy

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i

        return best_feature
    
    def vote(y):
        count = defaultdict(lambda: 0)
        for vote in y:
            count[vote] += 1
        sorted_count = sorted(count.iteritems(), key = operator.itemgetter(1), reverse = True)
        return sorted_count[0][0]

    # create the decision tree based on information gain
    def fit(X, y):
        classes = [item for item in y]
        if classed.count(classed[0]) == len(classes):
            return class[0]
        if len(X[0]) == 1:
            return self.vote(classes)


        best_feature = self.split_by_x(X, y)

    def predict(self, tree, y, x):
        first = tree.keys()[0]
        second = tree[first]
        feature_idx = y.index(first)
        key = x[feature_idx]
        value - second[key]
        if isinstance(value, dict):
            res = self.predict(value, y, x)
        else:
            res = value

        return res

def test():
    clf = DecisionTree()
    X, y, labels = clf.load_iris('iris.data.txt')
    tree = clf.fit(X, y, labels)
    print clf.predict(tree, x)

if __name__ == '__main__':
    test()
