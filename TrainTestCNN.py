'''
TrainTest is a convolutional neural network that trains on the featureset and labels
from create_sentiment_featureset. The model is then saved to a new folder CNNmodel.
Built on top of and using existing example from pythonprogramming.net
Found here: https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
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

n_classes = 10
batch_size = 10
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
    weights = {
        # 5 x 5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 conv, 32 inputs, 64 outputs 
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # fully connected, 33*19*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([33*19*64, 1024])), 
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    x = tf.reshape(x, shape=[-1, 129, 73, 1])
    print(x)

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    print(conv1)
    
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    print(conv2)

    fc = tf.reshape(conv2,[-1, 33*19*64])
    print(fc)
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

    output = tf.matmul(fc, weights['out'])+biases['out']
    print(output)

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    spec_path = os.path.join(orginal_path, 'CNNmodel\\')
    if not os.path.exists(spec_path):
        #create the ANNmodel folder if it doesn't already exist
        os.makedirs(spec_path)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        spec_path = os.path.join(spec_path, 'CNNmodelTest.ckpt')
        
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
    spec_path = os.path.join(orginal_path, 'CNNmodel\\')
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

    spec_path = os.path.join(spec_path, 'CNNmodel.ckpt')
    print(spec_path)

    prediction = convolutional_neural_network(x)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, spec_path)
        print("Model restored from file: %s" % spec_path)

        result = (sess.run(tf.argmax(prediction.eval(feed_dict={x:[Sxx]}),1)))
        print(result)

        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(lexicon)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        # invert result to corresponding label
        inverted = label_encoder.inverse_transform([argmax(onehot_encoded[result, :])])
        print(inverted)

#train_neural_network(x)
use_neural_network()