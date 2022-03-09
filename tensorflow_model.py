import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
tf.logging.set_verbosity(tf.logging.ERROR)

class WordSegmenter:

    def __init__(self, unicab_size, bicab_size, embedding_sizes, hidden_size, out_size, device='/cpu:0'):

        self.unicab_size = unicab_size
        self.bicab_size = bicab_size
        self.unibedding_size, self.bibedding_size = embedding_sizes
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.unigrams = tf.placeholder(tf.int32, shape=[None, None])
        self.bigrams = tf.placeholder(tf.int32, shape=[None, None])
        self.labels = tf.placeholder(tf.int64, shape=[None, None, out_size])
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.input_prob = tf.placeholder(tf.float32, shape=[])
        self.req_prob = tf.placeholder(tf.float32, shape=[])
        self.rnn_prob = tf.placeholder(tf.float32, shape=[])
        self.seq_length = tf.count_nonzero(self.unigrams, axis=-1)
        self.reg = tf.placeholder(tf.float32, shape=[])

        with tf.device(device):
            with tf.variable_scope("unigram_embeddings"):
                unibed_weights = tf.get_variable("unigram_embeddings", shape=[self.unicab_size, self.unibedding_size])
                self.unibeddings = tf.nn.embedding_lookup(unibed_weights, self.unigrams)

            with tf.variable_scope("bigram_embeddings"):
                bibed_weights = tf.get_variable("bigram_embeddings", shape = [self.bicab_size, self.bibedding_size])
                self.bibeddings = tf.nn.embedding_lookup(bibed_weights, self.bigrams)

                self.embeddings = tf.concat([self.unibeddings, self.bibeddings], 2)
            with tf.variable_scope("backward_lstm"):
                self.embeddings = tf.reverse(self.embeddings, [2])
                forward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                forward_cell = tf.nn.rnn_cell.DropoutWrapper(forward_cell, input_keep_prob=self.input_prob, 
                                                                            output_keep_prob=self.rnn_prob, 
                                                                            state_keep_prob=self.req_prob)

                f_out, _ = tf.nn.dynamic_rnn(forward_cell, self.embeddings, sequence_length=self.seq_length, dtype=tf.float32)

                f_out = tf.reverse(f_out, [2])
            with tf.variable_scope("forward_lstm"):
                backward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                backward_cell = tf.nn.rnn_cell.DropoutWrapper(backward_cell, input_keep_prob=self.input_prob, 
                                                                            output_keep_prob=self.rnn_prob, 
                                                                            state_keep_prob=self.req_prob)
                #The fully-connected layer of the newtork
                backward_cell = tf.contrib.rnn.OutputProjectionWrapper(backward_cell, self.out_size)
                b_out, _ = tf.nn.dynamic_rnn(backward_cell, f_out, sequence_length=self.seq_length, dtype=tf.float32)
                self.logits = b_out

        self.out = tf.nn.softmax(self.logits)
        with tf.variable_scope("loss"):
            # self.loss = tf.losses.softmax_cross_entropy(self.labels, self.logits)
            var = tf.trainable_variables() 
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in var if 'bias' not in v.name]) * self.reg
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits) + lossL2)

        with tf.variable_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        with tf.variable_scope("accuracy"):
            probs = tf.nn.softmax(self.logits)
            
            predictions = tf.cast(tf.floor(probs+0.5), tf.int64)
            self.acc = tf.reduce_mean(tf.cast(tf.equal(predictions, self.labels), tf.float32))

        self.saver = tf.train.Saver()

    def train(self, sess, unigrams, bigrams, labels, learning_rate=0.001, input_prob=0.8, rnn_prob=0.6, req_prob=0.6,  reg=0.01):
        return sess.run([self.optimizer, self.loss, self.acc], feed_dict={self.unigrams:unigrams, self.bigrams:bigrams, self.labels: labels,
                                                                          self.learning_rate:learning_rate, self.reg:reg,
                                                                          self.input_prob:input_prob, self.rnn_prob:rnn_prob, self.req_prob:req_prob})

    def evaluate(self, sess, unigrams, bigrams, labels, input_prob=0.8, rnn_prob=0.6, req_prob=0.6, reg=0.01):
        return sess.run([self.loss, self.acc], feed_dict={self.unigrams:unigrams, self.bigrams:bigrams, self.labels: labels,
                                                            self.input_prob:input_prob, self.rnn_prob:rnn_prob, self.req_prob:req_prob,self.reg:reg})
    def run(self, sess , unigrams, bigrams):
        return sess.run(self.out, feed_dict={self.unigrams:unigrams, self.bigrams:bigrams,self.input_prob:1.0, self.rnn_prob:1.0, self.req_prob:1.0})

    def __call__(self, sess, unigrams, bigrams):
        return sess.run(self.logits, feed_dict={self.unigrams:unigrams, self.bigrams:bigrams})

    def save(self, sess, filepath):
        return self.saver.save(sess, filepath)
    
    def restore(self, sess, filepath):
        return self.saver.restore(sess, filepath)

    