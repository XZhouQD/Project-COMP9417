#!/usr/bin/python3

# UNSW COMP9417 Group Project
# Topic 3.4 Recommender system using collaborative filtering
# Group members:
# XIMING FAN z5092028
# JIEXIN ZHOU z5199357
# XIAOWEI ZHOU z5108173

# Environment:
# Python 3.7.3
# pandas 0.23.3
# numpy 1.16.1
# matplotlib 3.0.2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os

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

	print(dataFrame.head())
