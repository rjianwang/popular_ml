import numpy as np
from collections import defaultdict
from math import log

class DecisionTree:

    # load iris datasets
    def load_iris(self):
        dataset = np.loadtxt('iris.data.txt', delimiter = ',', usecols = (0, 1, 2, 3), dtype = float)
        target = np.loadtxt('iris.data.txt', delimiter = ',', usecols = (range(4, 5)), dtype = str)
        X = dataset.tolist()
        y = target.tolist()
        labels = ['sepal_width', 'sepal_length', 'petal_length', 'petal_width']
        labels_copy = labels[:]
        return X, y, labels

    # calculate Shannon entropy
    def cal_shannon_ent(self, y):
        n = len(y)

        count = defaultdict(lambda: 0)
        for label in y:
            count[label] += 1

        shannon_ent = 0
        for key in count:
            p = float(count[key]) / n
            shannon_ent -= p * log(p, 2)

        return shannon_ent

    # split the dataset accoding to the axis which value is 'value'
    def split_dataset(self, X, y, axis, value):
        ret_sub_X = []
        ret_sub_y = []
        for i in range(len(X)):
            if X[i][axis] == value:
                reduced_x = X[i][:axis]
                reduced_x.extend(X[i][axis + 1 :])
                ret_sub_X.append(reduced_x)
                ret_sub_y.append(y[i])
        return ret_sub_X, ret_sub_y

    # select the best feature to split
    def chooseBestFeatureToSplit(self, X, y):
        n_feature = len(X[0])
        base_entropy = self.cal_shannon_ent(y)
        best_info_gain = 0.0
        best_feature = -1

        for i in range(n_feature):
            feature_list = [x[i] for x in X]
            unique_values = set(feature_list)
            cur_entropy = 0.0
            for value in unique_values:
                sub_X, sub_y = self.split_dataset(X, y, i, value)
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
    def fit(self, X, y, labels):
        if y.count(y[0]) == len(y):
            return y[0]
        if len(X[0]) == 1:
            return self.vote(y)

        best_feature = self.chooseBestFeatureToSplit(X, y)
        best_feature_label = labels[best_feature]
        
        tree = {best_feature_label: {}}
        del(labels[best_feature])

        feature = [item[best_feature] for item in X]
        unique_values = set(feature)

        for value in unique_values:
            sub_labels = labels[:]
            sub_X, sub_y = self.split_dataset(X, y, best_feature, value)
            tree[best_feature_label][value] = self.fit(sub_X,  sub_y, sub_labels)

        return tree

    # classify sample x accoding the tree
    def predict(self, tree, labels, x):
        first = tree.keys()[0]
        second = tree[first]
        feature_idx = labels.index(first)
        
        for key in second.keys():
            if x[feature_idx] == key:
                if type(second[key]).__name__ == 'dict':
                    class_label = self.predict(second[key], labels, x)
                else:
                    class_label = second[key]
    
        return class_label

def test():
    clf = DecisionTree()

    X, y, labels = clf.load_iris()
    x = [5.9, 3.0, 5.1, 1.8]

    tree = clf.fit(X, y, labels)
    print clf.predict(tree, labels, x)

if __name__ == '__main__':
    test()
