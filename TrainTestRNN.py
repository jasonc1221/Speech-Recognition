'''
TrainTest is a recurrent neural network that trains on the featureset and labels
from create_sentiment_featureset. The model is then saved to a new folder RNNmodel.
Built on top of and using existing example from pythonprogramming.net
Found here: hhttps://pythonprogramming.net/rnn-tensorflow-python-machine-learning-tutorial/
Used with California Polytechnic University California, Pomona Artificial Intelegence Club
Author: Machine Learning Lead, Jason Chang
Date: 14 March, 2018
    
Note: This script requires tensorflow, numpy dependencies to be installed
'''
from create_sentiment_featuresets import create_feature_sets_and_labels
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os

orginal_path = os.getcwd()
train_x,train_y,test_x,test_y = create_feature_sets_and_labels()

hm_epochs = 10
n_classes = 10
batch_size = 10
chunk_size = 73
n_chunks = 129
rnn_size = 1700

x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    spec_path = os.path.join(orginal_path, 'RNNmodel\\')
    if not os.path.exists(spec_path):
        #create the ANNmodel folder if it doesn't already exist
        os.makedirs(spec_path)
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        spec_path = os.path.join(spec_path, 'RNNmodel.ckpt')

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i=0
            while i < len(train_x)/batch_size:            #Not sure if its going through all data since dividing by batch_size
                start = i
                end = i+batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                batch_x = batch_x.reshape((batch_size,n_chunks,chunk_size))        #possibily need to change reshape arguments
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size

            save_path = saver.save(sess, spec_path)
            print("Model saved in path: %s" % save_path) 
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:np.array(test_x).reshape((-1, n_chunks, chunk_size)), y:test_y}))    #possibily need to change reshape arguments

def use_neural_network():
    #Changes directory to the ANNmodel folder
    spec_path = os.path.join(orginal_path, 'RNNmodel\\')
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

    spec_path = os.path.join(spec_path, 'RNNmodel.ckpt')
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