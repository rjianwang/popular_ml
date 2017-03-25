#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(datafile):
	return np.array(pd.read_csv(datafile, sep = "\t", header = -1)).astype(np.float)

def pca(X, k):
	"""
	STEP:
	1. 对数据进行归一化
	2. 计算归一化后的数据集的协方差矩阵
	3. 计算协方差矩阵的特征值和特征向量
	4. 保留最重要的k个特征(k小于矩阵维数n)及其特征向量
	5. 将m * n 的数据乘以k个n维的特征向量(n * k)，得到降维后的数据
	"""
	
	m, n = np.shape(X)
	if k > n:
		print("ERROR: k must lower than feature number")
		return

	# 归一化 
	avg = np.mean(X, axis = 0)
	X = X - avg

	# 协方差矩阵
	covX = np.cov(X, rowvar = 0)

	# 协方差矩阵的特征值及特征向量
	eigValue, eigVector = np.linalg.eig(np.mat(covX))

	# 对特征值从大到小排序
	sortedIndex = np.argsort(-eigValue)

	# 选择k个最重要的特征
	T = eigVector[sortedIndex[:k]]

	# 得到降维后的数据
	_X = np.dot(X, T.T)
	recX = (_X * T) + avg

	return _X, recX

def plotBestFit(data1, data2):
	dataArr1 = np.array(data1)
	dataArr2 = np.array(data2)

	m = np.shape(dataArr1)[0]

	axis_x1 = []
	axis_y1 = []
	axis_x2 = []
	axis_y2 = []

	for i in range(m):
		axis_x1.append(dataArr1[i, 0])
		axis_y1.append(dataArr1[i, 1])
		axis_x2.append(dataArr2[i, 0])
		axis_y2.append(dataArr2[i, 1])

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(axis_x1, axis_y1, s = 50, c = 'red', marker = 's')
	ax.scatter(axis_x2, axis_y2, s = 50, c = 'blue')

	plt.xlabel('x1')
	plt.ylabel('x2')
	plt.savefig('outfile.png')
	plt.show()


def main():
	datafile = "data.txt"

	X = load_data(datafile)
	K = 2

	return pca(X, k)

if __name__ == '__main__':
	datafile = "data.txt"
	X = load_data(datafile)
	k = 2;

	X1, X2 = pca(X, k)

	plotBestFit(X1, X2)
