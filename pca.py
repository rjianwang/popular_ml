#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def pca(X, crate):
	
	mValue = np.mean(X, axis = 0)
	X = X - mValue

	C = np.conv(X, rowvar = 0)

	eigValue, eigVector = np.linalg.eig(np.mat(C))

	sumEigValue = np.sum(eigValue)
	sortedEigValue = np.sort(eigValue)[::-1]
	
	for i in range(sortedEigValue.size):
		j = i + 1
		rate = np.sum(eigValue[0, j]) / sumEigValue
		if rate > crate:
			break

	indexVec = np.argsort(-eigValue)
	nLargestsIndex = indexVec[:j]
	T = eigVector[:, nLargestsIndex]

	newX = np.dot(X, T)

	return newX, T, mValue
