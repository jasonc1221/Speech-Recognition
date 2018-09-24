'''
create_sentiment_featuresets creates a featureset and labels from a 
data folder that has corresponding files in corresponding folders.
In addition organizes the data so it is flattened out into an single array.
Built on top of and using existing example from pythonprogramming.net
Found here: https://pythonprogramming.net/using-our-own-data-tensorflow-deep-learning-tutorial/
Used with California Polytechnic University California, Pomona Artificial Intelegence Club
Author: Machine Learning Lead, Jason Chang
Date: 14 March, 2018
	
Note: This script requires  numpy, sklearn, pickle dependencies to be installed
'''
import numpy as np
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
import pickle
from AudioToSpectrogram import getAudioData
from AudioToSpectrogram import saveSpectrogramData
import os

#creates a lexicon of all the words recorded and 'Silence' and 'Unknown'
def create_lexicon(path):
	lexicon = []
	#get all of the files in the path that is entered by the user
	listing = os.listdir(path)

	#Appends the string into the lexicon and prints lexicon
	for fileName in listing:
		lexicon.append(fileName)
	#lexicon.append("Silence")
	#lexicon.append("Unknown")
	print(lexicon)
	return lexicon

def sample_handling(path, lexicon):

	featureSet = []

	# integer encode
	label_encoder = LabelEncoder()
	integer_encoded = label_encoder.fit_transform(lexicon)
	print(integer_encoded)
	# binary encode
	onehot_encoder = OneHotEncoder(sparse=False)
	integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	print(onehot_encoded)
	#print("\n")
	# invert first example
	#inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
	#print(inverted)

	print(path)
	os.chdir(path)
	for label in lexicon:
		if os.path.exists(label):
			spec_path = os.path.join(path,label + '\\')
			print(spec_path)
			listing = os.listdir(spec_path)
			for fileName in listing:
				#print ("current file is: " + fileName)
				directory = spec_path + fileName
				#print(directory)
				audio_sig = getAudioData(directory)
				Sxx = saveSpectrogramData(audio_sig)
				#print(Sxx)
				indexOfLex = lexicon.index(label)
				#print(indexOfLex)

				#Arranging Sxx Matrix to an array
				#Sxx = np.array(Sxx[1])
				#print(Sxx)
				numrows = len(Sxx)										#Get the number of rows in Sxx
				#print(numrows)
				numcols = len(Sxx[0])									#Get the number of cols in Sxx
				#print(numcols)
				Sxx = np.reshape(Sxx, -1)								#Reshaping the matrix into a single array 
				#print(Sxx)
				#print(len(Sxx))
				featureSet.append([Sxx, onehot_encoded[indexOfLex]])	#Adding to the featureSet the Sxx and onehot classification
			os.chdir(path)
	#print("Regular featureSet")
	#print(featureSet)
	return featureSet

def create_feature_sets_and_labels(test_size = 0.1):
	path = input("Enter the FULL path to the folder in which all of the audio files are located." + "\n"
		+ "Make sure that there are only folders with corresponding word in this folder: ")
	lexicon = create_lexicon(path)
	featureSet = sample_handling(path, lexicon)
	random.shuffle(featureSet)
	featureSet = np.array(featureSet)
	#print("Shuffled featureSet")
	#print(featureSet)

	testing_size = int(test_size*len(featureSet))

	train_x = list(featureSet[:,0][:-testing_size])
	train_y = list(featureSet[:,1][:-testing_size])
	test_x = list(featureSet[:,0][-testing_size:])
	test_y = list(featureSet[:,1][-testing_size:])
	'''
	print("train_x")
	print(train_x)
	print("train_y")
	print(train_y)
	print("test_x")
	print(test_x)
	print("test_y")
	print(test_y)
	'''
	return train_x,train_y,test_x,test_y, lexicon

'''
if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('/path/to/pos.txt','/path/to/neg.txt')
	# if you want to pickle this data:
	with open('/path/to/sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
'''