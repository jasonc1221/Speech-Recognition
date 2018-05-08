'''
TrainTest is a neural network that trains on the featureset and labels
from create_sentiment_featureset. The model is then saved to a new folder ANNmodel.
Built on top of and using existing example from pythonprogramming.net
Found here: https://pythonprogramming.net/using-our-own-data-tensorflow-deep-learning-tutorial/
Used with California Polytechnic University California, Pomona Artificial Intelegence Club
Author: Machine Learning Lead, Jason Chang
Date: 14 March, 2018
	
Note: This script requires tensorflow, numpy dependencies to be installed
'''
from create_sentiment_featuresets import create_feature_sets_and_labels
from RecordAudioData import recordAudioTest
from AudioToSpectrogram import getAudioData
from AudioToSpectrogram import saveSpectrogramData
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import numpy as np
from numpy import argmax
import os

orginal_path = os.getcwd()
train_x,train_y,test_x,test_y, lexicon = create_feature_sets_and_labels()

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100
n_nodes_hl4 = 100

n_classes = 10				#Number of outputs (10 words minus "Silence" and "Unknown")
batch_size = 10
hm_epochs = 10

#Dropout
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

print("nodes = " + str(n_nodes_hl1))
print("epochs = " + str(hm_epochs))

x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

hidden_4_layer = {'f_fum':n_nodes_hl4,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl4]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    #l1 = tf.nn.dropout(l1, keep_rate)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    #l2 = tf.nn.dropout(l2, keep_rate)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)
    #l3 = tf.nn.dropout(l3, keep_rate)
    
    l4 = tf.add(tf.matmul(l3,hidden_4_layer['weight']), hidden_4_layer['bias'])
    l4 = tf.nn.relu(l4)

    output = tf.matmul(l4,output_layer['weight']) + output_layer['bias']

    return output

saver = tf.train.Saver()

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	spec_path = os.path.join(orginal_path, 'ANNmodel\\')
	if not os.path.exists(spec_path):
		#create the ANNmodel folder if it doesn't already exist
		os.makedirs(spec_path)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		spec_path = os.path.join(spec_path, 'ANNmodelTest2.ckpt')
	    
		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
				epoch_loss += c
				i+=batch_size

			save_path = saver.save(sess, spec_path)
			print("Model saved in path: %s" % save_path)	
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

def use_neural_network():
	#Changes directory to the ANNmodel folder
	spec_path = os.path.join(orginal_path, 'ANNmodel\\')
	os.chdir(spec_path)
	print(spec_path)

	#Creates a new .wav file to test audio and gets the Sxx into single array
	fileName = recordAudioTest()
	directory = spec_path + fileName
	print(directory)
	audio_sig = getAudioData(directory)
	Sxx = saveSpectrogramData(audio_sig)
	Sxx = np.reshape(Sxx, -1)
	print(Sxx)

	spec_path = os.path.join(spec_path, 'ANNmodelTest2.ckpt')
	print(spec_path)

	prediction = neural_network_model(x)
	init = tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess, spec_path)
		print("Model restored from file: %s" % spec_path)

		result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[Sxx]}),1)))
		print(result)

		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(lexicon)
		onehot_encoder = OneHotEncoder(sparse=False)
		integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

		# invert first example
		inverted = label_encoder.inverse_transform([argmax(onehot_encoded[result, :])])
		print(inverted)

train_neural_network(x)
#use_neural_network()