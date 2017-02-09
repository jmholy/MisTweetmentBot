import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from data_process import tweet_splitter
import numpy as np
import math

batch = 5
total_length = 120
rnn_size = 128

train_input, train_output, validate_input, validate_output = tweet_splitter()

s = tf.placeholder(tf.float32, [None, len(train_input[0])])
output = tf.placeholder(tf.float32)

"""filequeue = tf.train.string_input_producer(["unicodeHate.txt", "good.txt"])
key, value = tf.TextLineReader.read(filequeue)

record_defaults = [[""],[0]]
tweet, hatescale = tf.decode_csv(value, record_defaults=record_defaults)
words = tweet.split(" ")
print(words)
#maintensor = tf.pack([tweet, hatescale])

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1000):
        # Retrieve a single instance:
        for iword in range(len(words)):
            example, label = sess.run([iword, hatescale])

    coord.request_stop()
    coord.join(threads)"""

def recurrent_nn(s):
    level = {'weight': tf.Variable(tf.random_normal([rnn_size, 2])), 'bias': tf.Variable(tf.random_normal([2]))}
    mem_cell = rnn_cell.BasicLSTMCell(rnn_size)
    #s = np.reshape(s, [1, len(train_input[0])])
    s = tf.split(0, len(train_input), s)
    outputs, states = rnn.rnn(mem_cell, s, dtype=tf.float32)
    final = tf.matmul(outputs[-1], level['weight']) + level['bias']
    return final


prediction = recurrent_nn(s)
#softmax = (tf.nn.softmax_cross_entropy_with_logits(prediction, output))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, output))
optimize = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

num_epochs = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_run in range(num_epochs):
        epoch_loss = 0
        i = 0
        while i < len(train_input):
            epoch_input = np.array(train_input[i:i+batch])
            epoch_output = np.array(train_output[i:i+batch])
            temp_dict = {s: epoch_input, output: epoch_output}
            _, count = sess.run([optimize, loss], feed_dict=temp_dict)
            i += batch
            epoch_loss += count
        print("completed epoch #", epoch_run, "out of", num_epochs, "Loss:", epoch_loss)
    checker = tf.equal(tf.argmax(prediction, 1), tf.argmax(output, 1))
    percent = tf.reduce_mean(tf.cast(checker, 'float'))
    print("Percentage:", percent.eval({s:validate_input, output:validate_output}))