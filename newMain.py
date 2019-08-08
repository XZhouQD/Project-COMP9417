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
	#Mean squared error of prediction and test values.
	return np.average((test-predict)**2)

def predict_all(simMatrix, train, test):
	# Aggregate prediction matrix.
	# similarity * ratingMatrix / similarity sum
	# For prediction, use training set of ratingMatrix.
	predict = simMatrix.dot(train) / np.array([np.abs(simMatrix).sum(axis=1)]).T
	mse = meanSquaredError(test[test.nonzero()].flatten(), predict[test.nonzero()].flatten())
	print("Mse of predict based on all is: " + str(mse) + ".")
	return mse

def predict_k(k, simMatrix, train, test):
	predict = np.zeros(test.shape)
	percent = -10
	for userID in range(simMatrix.shape[0]):
		if userID % (math.ceil(simMatrix.shape[0]/10)) == 0:
			percent += 10
			print(percent, "% finished.")
		#select top k except current userID
		#since argsort gives inverse order, we use negative indexing
		#dispose himself : do not include index -1
		top_k = [np.argsort(simMatrix[:,userID], axis=-1, kind='quicksort')[-2:-k-2:-1]]
		for movieID in range(train.shape[1]):
			# similarity of curr user * ratingMatrix of curr item of top k / similarity of curr user sum
			# It is simply one value.
			predict[userID, movieID] = simMatrix[userID,:][tuple(top_k)].dot(train[:,movieID][tuple(top_k)]) / np.sum(simMatrix[userID,:][tuple(top_k)])
	mse = meanSquaredError(test[test.nonzero()].flatten(), predict[test.nonzero()].flatten())
	print("Mse of predict based on " + str(k) + " is: " + str(mse) + ".")
	return mse

def recommand_k(userID, rank, k, simMatrix, ratingMatrix):
	# UserID starts at 1 while matrix index starts at 0.
	userID -= 1
	predict = np.zeros((simMatrix.shape))
	top_k = [np.argsort(simMatrix[:,userID], axis=-1, kind='quicksort')[-2:-k-2:-1]]
	for movieID in range(train.shape[1]):
		if ratingMatrix[userID, movieID] == 0: #not marked by the user
			predict[userID,movieID] = simMatrix[userID,:][tuple(top_k)].dot(ratingMatrix[:,movieID][tuple(top_k)]) / np.sum(simMatrix[userID,:][tuple(top_k)])
	return [movieID for movieID in np.argsort(predict[userID,:])[-(rank + 5):]]


def movieID_matrix_correction(nMovies):
	# The movieIDs in the movie.dat are not continuous so the matrix indexes and movieIDs are not consistent.
	# So we need to add NA values to fill up the matrix.
    dataFile = 'data-1m\movies.dat'
    header = ["MovieID", "MovieName", "Classification"]
    dataFrame = pd.read_csv(dataFile, skiprows=0, sep='::', names=header, engine='python')
    counter = 0
    while True:
        counter += 1
        if counter != dataFrame.iloc[counter - 1]["MovieID"]:
            # Create empty line to fill up the space.
            insert_row = pd.DataFrame([[counter, "NA", "NA"]], columns=header)
			# We separate the matrix into upper and lower matrix and add the NA values in it.
            upper_matrix = dataFrame[:counter - 1]
            lower_matrix = dataFrame[counter - 1:]
			# And then concatenate them.
            dataFrame = pd.concat(
                [upper_matrix, insert_row, lower_matrix], ignore_index=True)
        if counter == nMovies:
            break
    return dataFrame

def movie_id_to_name(top10_id, matrix):
	# Get movieID and return movie name & class. 
	# top10_id is in reversed order.
	movie_name_list = []
	counter = 0
	top10_id.reverse()
	for i in top10_id:
		if i == 0:
			continue
		if counter == 10:
			break
		temp = np.array(matrix[matrix["MovieID"] == i])
		if temp[0][1] != "NA":
			movie_name_list.append(np.array(matrix[matrix["MovieID"] == i]))
			counter += 1
	movie_name_list = np.array(movie_name_list)
	return movie_name_list

if __name__ == '__main__':
	# Part 1
	# Load and reformat data.
	# Default data position: data/ratings.csv
	# dataFile = 'data/ratings.csv'
	dataFile = 'data-1m/ratings.dat'
	seperater = '::'
	print("Initializing...")
	if len(sys.argv) > 2:
		# try command line input file as data resource
		dataFile = sys.argv[1]
		seperater = sys.argv[2]
		if not os.path.isfile(dataFile):
			exit("Input file name Error: Check your argument or use default data file.")
	header = ["userID", "movieID", "rating", "timestamp"]
	dataFrame = pd.read_csv(dataFile, skiprows=1, sep=seperater, names=header, engine='python')
	print("...Data read success.")
	np.random.seed(9417) # constant random seed

	# Part 2
	# Construct rating matrix.
	# User x MovieItem = Rating
	# ID starts from 1, index starts from 0
	nUsers = max(dataFrame.userID)
	nMovies = max(dataFrame.movieID)
	ratingMatrix = np.zeros((nUsers, nMovies))
	for row in dataFrame.itertuples():
		ratingMatrix[row[1]-1,row[2]-1] = row[3]
	entryCount = dataFrame.shape[0]
	print("...Rating matrix established.")

	# Part 3
	# Train-test split, we make 10% of testing for each user.
	# Randomly choose from user's rating, move them to testing.
	train = ratingMatrix.copy()
	test = np.zeros((nUsers, nMovies))
	#nItemsTest = math.floor(0.1*float(entryCount)/nUsers)
	for userID in range(nUsers):
		userRatedMovies = ratingMatrix[userID,:].nonzero()[0];
		nItemsTest = math.floor(0.1*len(userRatedMovies))
		rMovieID = np.random.choice(userRatedMovies, size=nItemsTest, replace=False) #no-replacement selection
		test[userID,rMovieID] = ratingMatrix[userID,rMovieID]
		train[userID,rMovieID] = 0
	print("...Train-test split success.")

	# Part 4
	# Calculate similarity matrix based on training set.
	# Use cosine similarity function from the lecture notes.
	sim = np.dot(train, train.T) + 1e-6 #self dot product, non-zero result
	norms = np.array([np.sqrt(np.diagonal(sim))]) #diagonal values
	simMatrix = sim/(norms*norms.T)
	print("...Similarity matrix established.")

	# Part 5
	# Calculate the mse based on all users.
	# Calculate the mse based on different percentage of users.
	# To find the minimum mse, use that number of users to proceed Part 6.
	print("Run MSE calculation based on all users? (Y/N) (Might take up few seconds.)")
	run_confirm = sys.stdin.readline()
	if run_confirm == "Y\n" or run_confirm == "y\n":
		mse_all = predict_all(simMatrix, train, test)
		mse_list = []
		k_list = []
		print("Run MSE calculation based on different percentage of users? (Y/N) (Might take up few minutes!!!)")
		run_confirm = sys.stdin.readline()
		if run_confirm == "Y\n" or run_confirm == "y\n":
			k_list = [5, 10, 20, 50, 100]
			for k in k_list:
				print("Predict based on top " + str(k) + " user.")
				mse_list.append(predict_k(k, simMatrix, train, test))
		mse_list.append(mse_all)
		k_list.append(nUsers)
		# Part 5.5
		# Visualise top-k prediction mse
		yPos = np.arange(len(k_list))
		plt.bar(yPos, mse_list, align='center')
		plt.xticks(yPos, k_list)
		plt.ylabel('Mean Squared Error')
		plt.title('MSE with top-k prediction')
		plt.show()

	# Part 6
	# Get top 10 movies predicted with highest rankings and haven't seen before for input userID.
	print("Input UserID you want to predict.")
	recSim = np.dot(ratingMatrix, ratingMatrix.T) + 1e-6
	norms = np.array([np.sort(np.diagonal(recSim))])
	recSimMatrix = recSim/(norms*norms.T)
	while True:
		try:
			target_user = int(sys.stdin.readline())
			if target_user < 1 or target_user > nUsers:
				exit(f"Invalid UserID. UserID should be an integer within [1, {nUsers}].")
		except:
			exit(f"Invalid UserID. UserID should be an integer within [1, {nUsers}].")
		print(f"UserID: {target_user}")
		top10_id = recommand_k(target_user, 10, 20, recSimMatrix, ratingMatrix)
		movie_matrix = movieID_matrix_correction(nMovies)
		print(f"{target_user}'s top 10 predicted movies are:")
		print(movie_id_to_name(top10_id, movie_matrix))
		print()
		print("Input next UserID you want to predict, or other characters to exit.")
