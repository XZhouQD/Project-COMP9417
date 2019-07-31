#!/usr/bin/python3

# UNSW COMP9417 Group Project
# Topic 3.4 Recommender system using collaborative filtering
# Group members:
# XIMING FAN z5092028
# JIEXIN ZHOU z5199357
# XIAOWEI ZHOU z5108173

# Environment:
# Python 3.7.3
# pandas 0.25.0
# numpy 1.17.0
# matplotlib 3.1.1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import math

def meanSquaredError(test, predict):
	#mean squared error of prediction and test values
	return np.average((test-predict)**2)

def predict_all(simMatrix, train, test):
	# aggregate prediction matrix
	# similarity * ratingMatrix / similarity sum
	# for prediction, use training set of ratingMatrix
	predict = simMatrix.dot(train) / np.array([np.abs(simMatrix).sum(axis=1)]).T
	mse = meanSquaredError(test[test.nonzero()].flatten(), predict[test.nonzero()].flatten())
	print("mse of predict based on all is: ", str(mse))
	return mse


if __name__ == '__main__':
	# Part 1
	# Load and reformat data
	# Default data position: data/ratings.csv
	dataFile = 'data/ratings.csv'

	if len(sys.argv) > 1:
		# try command line input file as data resource
		dataFile = sys.argv[1]
		if not os.path.isfile(dataFile):
			exit("Input file name Error: Check your argument or use default data file.")

	header = ["userID", "movieID", "rating", "timestamp"]
	dataFrame = pd.read_csv(dataFile, skiprows=1, sep=',', names=header)
	print("data read success")

	# Part 2
	# Construct rating matrix
	# User x MovieItem = Rating
	# ID starts from 1, index starts from 0
	nUsers = max(dataFrame.userID)
	nMovies = max(dataFrame.movieID)
	ratingMatrix = np.zeros((nUsers, nMovies))
	for row in dataFrame.itertuples():
		ratingMatrix[row[1]-1,row[2]-1] = row[3]
	entryCount = dataFrame.shape[0]
	print("rating matrix established")

	# Part 3
	# train-test split, we make 10% of testing for each user
	# randomly choose from user's rating, move them to testing
	train = ratingMatrix.copy()
	test = np.zeros((nUsers, nMovies))

	#nItemsTest = math.floor(0.1*float(entryCount)/nUsers)
	for userID in range(nUsers):
		userRatedMovies = ratingMatrix[userID,:].nonzero()[0];
		nItemsTest = math.floor(0.1*len(userRatedMovies))
		rMovieID = np.random.choice(userRatedMovies, size=nItemsTest, replace=False) #no-replacement selection
		test[userID,rMovieID] = ratingMatrix[userID,rMovieID]
		train[userID,rMovieID] = float(0)
	print("train-test split success")

	# Part 4
	# calculate similarity matrix based on training set
	# use cosine similarity function from the lecture notes
	sim = np.dot(train, train.T) + 1e-6 #self dot product, non-zero result
	norms = np.array([np.sqrt(np.diagonal(sim))]) #diagonal values
	simMatrix = sim/(norms*norms.T)
	print("similarity matrix established")

	mse_all = predict_all(simMatrix, train, test)
	print("predict based on all users")
