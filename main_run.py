import os 
# os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import dataset
from preprocessing import preprocess
from tensorflow_model import WordSegmenter
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import tensorflow as tf 
tf.logging.set_verbosity(tf.logging.ERROR)

def create_dataset(test_set, max_length):

    test_uni, test_bi, test_label = test_set[:, 0], test_set[:, 1], test_set[:, 2]
    
    # In order to feed them as Tensors, we need all of them to have equal length,
    # so we truncate long sentences and we pad short ones.
    print("\nPadding sequence to {} tokens".format(max_length))

    # When truncating, get rid of initial words and keep last part of the review. (longer sentences)
    # When padding, pad at the end of the sentence. (shorter sentences)
    test_uni = pad_sequences(test_uni, truncating='pre', padding='post', maxlen=max_length)
    test_bi = pad_sequences(test_bi, truncating='pre', padding='post', maxlen=max_length)
    # test_label = pad_sequences(test_label, truncating='pre', padding='post', maxlen=max_length)

    print("Test set shape:", test_uni.shape)
        
    # Take 5% of the training set and use it as dev set
    # stratify makes sure that the development set follows the same distributions as the training set:
    # half positive and half nevative.

    test_x = [(u, b) for u,b in zip(test_uni, test_bi)]

    
    return (test_x, test_label)

def batch_generator(X, Y, batch_size, shuffle=False):
    if not shuffle:
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield np.array(X[start:end]), np.array(Y[start:end])
    else:
        perm = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            yield np.array(X[perm[start:end]]), np.array(Y[perm[start:end]])



def run(input_path, out_path):

    batch_size = 1

    preprocess(input_path)
    dataset.test_set('predict', 'data/msr_training_unicab.json', 'data/msr_training_bicab.json')
    test_set = np.load('data/predict.npy')

    (test_x, test_label) = create_dataset(test_set, 30)
    with open('data/msr_training_unicab.json', 'r') as f:
        unicab = json.load(f)
        f.close()
    with open('data/msr_training_bicab.json', 'r') as f:
        bicab = json.load(f)
        f.close()
    unicab_size = len(unicab.keys())
    bicab_size = len(bicab.keys())

    out_file = open(out_path, 'a')

    tf.reset_default_graph()
    with tf.variable_scope("seg1"):
        segmenter1 = WordSegmenter(unicab_size, bicab_size, [64, 16], 256, 4, device='/gpu:0')


    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        #LOAD the Model
        segmenter1.restore(sess, 'models/msr03')
        print("\nEvaluating test...")
        for batch_x, _ in batch_generator(test_x, test_label, batch_size):
            unigrams, bigrams = batch_x[:,0], batch_x[:, 1]
            out = segmenter1.run(sess, unigrams, bigrams)
            # print(out)
            sentence = []
            for i in np.squeeze(out):
                l = np.argmax(i)
                if l==0:
                    sentence.append("B")
                elif l==1:
                    sentence.append("I")
                elif l==2:
                    sentence.append("E")
                elif l==3:
                    sentence.append("S")
            out_file.write(''.join(sentence)+'\n')
