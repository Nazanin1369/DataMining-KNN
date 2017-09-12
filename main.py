import pandas as pd
import numpy as np
import math
import operator

### NOTE: Python 3.x is required to compile this file

TRAIN_FILE_PATH = './data/train.csv'
TEST_FILE_PATH = './data/test.csv'
DIRECTION_FILE_PATH = './data/trainDirection.csv'
FINAL_TEST_FILE_k1 = './results/testingk1.csv'
FINAL_TEST_FILE_k3 = './results/testingk3.csv'
FINAL_TEST_FILE_k5 = './results/testingk5.csv'
FINAL_TEST_FILE_k10 = './results/testingk10.csv'

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
    directions_df =  pd.read_csv(DIRECTION_FILE_PATH)
    df['Direction'] = directions_df['Direction']
    #print(df.head())
    return df

'''
    Utility function to transform dataFrame into array of its rows
'''
def transformTrainingData(train_data_frame):
    trainings= []
    for index, row in train_data_frame.iterrows():
        test_row = [round(row[0], 3), round(row[1], 3), row[2]]
        trainings.append(test_row)
    return trainings

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
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


'''
    Calculates the accuarcy of predictions by taking
    a ratio of total correct prediction out of all predictions
'''
def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x] == predictions[x]:
			correct += 1
	return round((correct/float(len(testSet))) * 100.0, 3)


'''
Main KNN function
'''
def knn(k, path, test_data, train_data, actual_test_data):
    print('*******************************************')
    print('************    K = ', k , '  ******************')
    print('*******************************************')
    copy_of_test_data = test_data.copy()
    copy_of_train_data = train_data
    predictions = []

    print('--> Calculating KNN...')
    for index, test_sample in test_data.iterrows():
        #print(test_sample.values[0], ', ', test_sample.values[1])
        neighbors = getNeighbors(copy_of_train_data, test_sample, k)
        neighbors_prediction = getResponse(neighbors)
        predictions.append(neighbors_prediction)

    print('--> Writing predictions...')
    copy_of_test_data['Direction'] = predictions
    copy_of_test_data.to_csv(path, float_format='%.3f')
    print('--> Calculating Accuracy...')
    accuracy = getAccuracy(actual_test_data, copy_of_test_data['Direction'].values)
    print('Accuracy = ', accuracy, '%')

#############################################################################################
## read training data into a dataFrame
train_data_frame  = processData(loadData(TRAIN_FILE_PATH))
training_data = transformTrainingData(train_data_frame)
## read testing data into a dataFrame
test_data_frame = loadData(TEST_FILE_PATH)

## Just an assumption file for calculating Accuracy
## Accuracy is not accurate since direction comparing to is Mock directions I manually entered just
## for the sake of calculation
## To professor: For accurate accuracy calculations a provided actual result is needed
actual_test_result = pd.read_csv('./data/testing.csv')['Direction'].values

## verigying data retrieval
print('Training data: ', train_data_frame.shape[0] ,'samples.')
print('Testing data: ', test_data_frame.shape[0], 'samples.')

knn(1, FINAL_TEST_FILE_k1, test_data_frame, training_data, actual_test_result)
knn(3, FINAL_TEST_FILE_k3, test_data_frame, training_data, actual_test_result)
knn(5, FINAL_TEST_FILE_k5, test_data_frame, training_data, actual_test_result)
knn(10, FINAL_TEST_FILE_k10, test_data_frame, training_data, actual_test_result)

##################################################################################
########################## Output ###############################################
'''
(naz) 🦄 python main.py
Training data:  998 samples.
Testing data:  252 samples.
*******************************************
************    K =  1   ******************
*******************************************
--> Calculating KNN...
--> Writing predictions...
--> Calculating Accuracy...
Accuracy =  100.0 %
*******************************************
************    K =  3   ******************
*******************************************
--> Calculating KNN...
--> Writing predictions...
--> Calculating Accuracy...
Accuracy =  76.19 %
*******************************************
************    K =  5   ******************
*******************************************
--> Calculating KNN...
--> Writing predictions...
--> Calculating Accuracy...
Accuracy =  69.841 %
*******************************************
************    K =  10   ******************
*******************************************
--> Calculating KNN...
--> Writing predictions...
--> Calculating Accuracy...
Accuracy =  64.683 %
'''

