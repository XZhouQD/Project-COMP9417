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
	print("mse of predict based on all is: " + str(mse))
	return mse

def predict_k(k, simMatrix, train, test):
	predict = np.zeros(test.shape)
	percent = -10
	for userID in range(simMatrix.shape[0]):
		if userID % (math.ceil(simMatrix.shape[0]/10)) == 0:
			percent += 10
			print(percent, "% finished")
		#select top k except current userID
		#since argsort gives inverse order, we use negative indexing
		#dispose himself : do not include index -1
		top_k = [np.argsort(simMatrix[:,userID], axis=-1, kind='quicksort')[-2:-k-2:-1]]
		for movieID in range(train.shape[1]):
			# similarity of curr user * ratingMatrix of curr item of top k / similarity of curr user sum
			# it is simply one value
			predict[userID, movieID] = simMatrix[userID,:][tuple(top_k)].dot(train[:,movieID][tuple(top_k)]) / np.sum(simMatrix[userID,:][tuple(top_k)])
	mse = meanSquaredError(test[test.nonzero()].flatten(), predict[test.nonzero()].flatten())
	print("mse of predict based on " + str(k) + " is: " + str(mse))
	return mse

def recommand_k(userID, rank, k, simMatrix, ratingMatrix):
	predict = np.zeros((simMatrix.shape))
	top_k = [np.argsort(simMatrix[:,userID], axis=-1, kind='quicksort')[-2:-k-2:-1]]
	for movieID in range(train.shape[1]):
		if ratingMatrix[userID, movieID] == 0: #not marked by the user
			predict[userID,movieID] = simMatrix[userID,:][tuple(top_k)].dot(ratingMatrix[:,movieID][tuple(top_k)]) / np.sum(simMatrix[userID,:][tuple(top_k)])
	return [movieID for movieID in np.argsort(predict[userID,:])[-rank:]]


if __name__ == '__main__':
	# Part 1
	# Load and reformat data
	# Default data position: data/ratings.csv
	#dataFile = 'data/ratings.csv'
	dataFile = 'data-1m/ratings.dat'
	if len(sys.argv) > 1:
		# try command line input file as data resource
		dataFile = sys.argv[1]
		if not os.path.isfile(dataFile):
			exit("Input file name Error: Check your argument or use default data file.")

	header = ["userID", "movieID", "rating", "timestamp"]
	dataFrame = pd.read_csv(dataFile, skiprows=1, sep='::', names=header, engine='python')
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
		train[userID,rMovieID] = 0
	print("train-test split success")

	# Part 4
	# calculate similarity matrix based on training set
	# use cosine similarity function from the lecture notes
	sim = np.dot(train, train.T) + 1e-6 #self dot product, non-zero result
	norms = np.array([np.sqrt(np.diagonal(sim))]) #diagonal values
	simMatrix = sim/(norms*norms.T)
	print("similarity matrix established")

	recSim = np.dot(ratingMatrix, ratingMatrix.T) + 1e-6
	norms = np.array([np.sort(np.diagonal(recSim))])
	recSimMatrix = recSim/(norms*norms.T)

	print(recommand_k(15-1, 10, 50, recSimMatrix, ratingMatrix))

	print("predict based on all users")
	mse_all = predict_all(simMatrix, train, test)

	k_list = [5]
	mse_list = []
	for k in k_list:
		print("predict based on top " + str(k) + " user")
		mse_list.append(predict_k(k, simMatrix, train, test))
