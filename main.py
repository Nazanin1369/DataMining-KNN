import pandas as pd
import numpy as np
import math
import operator


TRAIN_FILE_PATH = './data/train.csv'
FINAL_TEST_FILE = './data/training.csv'
TEST_FILE_PATH = './data/test.csv'

train_data_frame = []
test_data_frame = []

'''
    Reads a provided csv file into pandas dataFrame
'''
def loadData(filePath):
    dataFrame = pd.read_csv(filePath)
    return dataFrame

'''
    Adds Movement columns for each Lag into dataframes
'''
def processData(df):
    df['Movement1'] = np.where((df['Lag1'] >= 0), 'Up', 'Down')
    df['Movement2'] = np.where((df['Lag2'] >= 0), 'Up', 'Down')
    return df

'''
 Calculates similarity between two given data samples
 so we can locaate k most similar data instances in the
 training dataset for a given sample in test dataset
'''
def euclidianDistance(sample1, sample2, length):
    distance = 0
    for x in range(length):
        distance += pow(sample1[x] - sample2[x], 2)
    return math.sqrt(distance)

'''
    Uses similarity to collect the K most nearest samples for a given unseen data
    Here we wil calculate the distance for all samples and select a subset with smallest
    distance values.
'''
def getNeighbors(train_set, test_sample, k):
    distances = []
    neighbors = []
    length = len(test_sample - 1)

    for x in range(len(train_set)):
        dist = euclidianDistance(test_sample, train_set[x], length)
        distances.append((train_set[x], dist))

    distances.sort(key=operator.itemgetter(1))

    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors

'''
    After identifying the most similar neighbors for the test sample
    now wenhave to get those neighbors to devise a predicted response.
    In this way each neibor will vote to their class attributes, and take
    the majority vote as predicted category.
'''
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


'''
    Calculates the accuarcy of predictions by taking
    a ratio of total correct prediction out of all predictions
'''
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0



## read training data into a dataFrame
train_data_frame  = processData(loadData(TRAIN_FILE_PATH))

## read testing data into a dataFrame
test_data_frame = processData(loadData(TEST_FILE_PATH))

## verigying data retrieval
#print('Training data: ', train_data_frame.shape[0] ,'samples.')
#print('Testing data: ', test_data_frame.shape[0], 'samples.')
print(train_data_frame.head())
print(test_data_frame.head())




