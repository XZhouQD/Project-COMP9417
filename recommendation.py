import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math

def movie_id_to_name(top10_id):
	dataFile = 'data-1m/movies.dat'
	header = ["MovieID", "MovieName", "Classification"]
	dataFrame = pd.read_csv(dataFile, skiprows=1, sep='::', names=header, engine='python')
	movie_name_list = []
	for i in top10_id:
		movie_name_list.append(np.array(dataFrame[dataFrame["MovieID"] == i]))
	movie_name_list = np.array(movie_name_list)
	print(movie_name_list)
	return 

def meanSquaredError(test, predict):
	#mean squared error of prediction and test values
	return np.average((test-predict)**2)

def predict_k_list(k, simMatrix, train, test):
	predict = np.zeros((test.shape))
	percent = -10
	for userID in range(simMatrix.shape[0]):
		if userID % (math.ceil(simMatrix.shape[0]/10)) == 0:
			percent += 10
			print(percent, "% finished.")
		#select top k except current userID
		#since argsort gives inverse order, we use negative indexing
		top_k = [np.argsort(simMatrix[:, userID], axis=-1,
		                    kind='quicksort')[-1:-k-1:-1]]
		for movieID in range(train.shape[1]):
			# similarity of curr user * ratingMatrix of curr item of top k / similarity of curr user sum
			# it is simply one value
			predict[userID, movieID] = simMatrix[userID, :][tuple(top_k)].dot(
				train[:, movieID][tuple(top_k)]) / np.sum(simMatrix[userID, :][tuple(top_k)])
	mse = meanSquaredError(test[test.nonzero()].flatten(),
	                       predict[test.nonzero()].flatten())
	print(100, "% finished.")
	print("MSE of predict based on " + str(k) + " is: " + str(mse) + ".")
	return predict

if __name__ == '__main__':
	# Part 1
	# Load and reformat data
	# Default data position: data/ratings.csv
	# dataFile = 'data/ratings.csv'
	print("Initializing...")
	dataFile = 'data-1m/ratings.dat'
	if len(sys.argv) > 1:
		# try command line input file as data resource
		dataFile = sys.argv[1]
		if not os.path.isfile(dataFile):
			exit("Input file name Error: Check your argument or use default data file.")

	header = ["userID", "movieID", "rating", "timestamp"]
	dataFrame = pd.read_csv(dataFile, skiprows=1, sep='::', names=header, engine='python')
	nUsers = max(dataFrame.userID)

	#get userid which we want to recommend movies to
	print("Input UserID you want to predict.")
	try:
		target_user = int(sys.stdin.readline())
		if target_user < 1 or target_user > nUsers:
			exit(f"Invalid UserID. UserID should be an integer within [1, {nUsers}].")
	except:
		exit(f"Invalid UserID. UserID should be an integer within [1, {nUsers}].")
	print(f"UserID: {target_user}")
	print("Data read success.")

	# Part 2
	# Construct rating matrix
	# User x MovieItem = Rating
	# ID starts from 1, index starts from 0
	nUsers = max(dataFrame.userID)
	nMovies = max(dataFrame.movieID)
	ratingMatrix = np.zeros((nUsers, nMovies))
	for row in dataFrame.itertuples():
		ratingMatrix[row[1]-1, row[2]-1] = row[3]
	entryCount = dataFrame.shape[0]
	print("Rating matrix established.")
	
	# Part 3
	# calculate similarity matrix based on training set
	# use cosine similarity function from the lecture notes
	train = ratingMatrix.copy()
	test = ratingMatrix.copy()
	sim = np.dot(train, train.T) + 1e-6
	norms = np.array([np.sqrt(np.diagonal(sim))])  # diagonal values
	simMatrix = sim/(norms*norms.T)
	print("Similarity matrix established.")
	
	# Part 4
	# From main.py we know when predicting top 50, mse reaches bottom
	# Predict top 50 and use all data to train
	min_mse_k = 50
	print(f"Using top {min_mse_k} most similar users to predict.")
	predict_matrix = predict_k_list(min_mse_k, simMatrix, train, test)

	# Part 5 
	# Get top 10 movies predicted with highest rankings and haven't seen before for input userID
	predict_list = predict_matrix[target_user - 1]
	predict_list_with_index = pd.DataFrame(predict_list)
	movieID_list = np.arange(1, nMovies + 1)
	predict_list_with_index["movieID"] = movieID_list
	seen_movieID_list = dataFrame[dataFrame["userID"] == target_user]
	for i in predict_list_with_index.itertuples():
		for j in seen_movieID_list.itertuples():
			if (i[2])  == j[2]:
				get_index = i[2] - 1
				predict_list_with_index.drop([get_index], inplace = True)
	sorted_predict_list_with_index = predict_list_with_index.copy()
	sorted_predict_list_with_index = sorted_predict_list_with_index.sort_values(by=[0], ascending=False).head(10)
	print("Top 10 predicted movies:")
	movie_id_to_name(sorted_predict_list_with_index["movieID"])

	
	
	




