import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow_model import WordSegmenter
import json


tf.logging.set_verbosity(tf.logging.ERROR)

def create_dataset(train_set, val_set, test_set, max_length):

    train_uni, train_bi, train_label = train_set[:, 0], train_set[:, 1], train_set[:, 2]
    val_uni, val_bi, val_label = val_set[:, 0], val_set[:, 1], val_set[:, 2]
    test_uni, test_bi, test_label = test_set[:, 0], test_set[:, 1], test_set[:, 2]
    
    #BIES to numerical representation
    train_label = enum(train_label)
    val_label = enum(val_label)
    test_label = enum(test_label)


    # In order to feed them as Tensors, we need all of them to have equal length,
    # so we truncate long sentences and we pad short ones.
    print("\nPadding sequence to {} tokens".format(max_length))

    # When truncating, get rid of initial words and keep last part of the review. (longer sentences)
    # When padding, pad at the end of the sentence. (shorter sentences)
    train_uni = pad_sequences(train_uni, truncating='pre', padding='post', maxlen=max_length)
    train_bi = pad_sequences(train_bi, truncating='pre', padding='post', maxlen=max_length)
    train_label = pad_sequences(train_label, truncating='pre', padding='post', maxlen=max_length)

    val_uni = pad_sequences(val_uni, truncating='pre', padding='post', maxlen=max_length)
    val_bi = pad_sequences(val_bi, truncating='pre', padding='post', maxlen=max_length)
    val_label = pad_sequences(val_label, truncating='pre', padding='post', maxlen=max_length)

    test_uni = pad_sequences(test_uni, truncating='pre', padding='post', maxlen=max_length)
    test_bi = pad_sequences(test_bi, truncating='pre', padding='post', maxlen=max_length)
    test_label = pad_sequences(test_label, truncating='pre', padding='post', maxlen=max_length)

    
    print("Training set shape:", train_uni.shape)
    print("Validation set shape:", train_uni.shape)
    print("Test set shape:", test_uni.shape)
        
    # Take 5% of the training set and use it as dev set
    # stratify makes sure that the development set follows the same distributions as the training set:
    # half positive and half nevative.
    train_x = [(u, b) for u,b in zip(train_uni, train_bi)]
    val_x = [(u, b) for u,b in zip(val_uni, val_bi)]
    test_x = [(u, b) for u,b in zip(test_uni, test_bi)]

    train_x, dev_x, train_y, dev_y = train_test_split(train_x, train_label, test_size=.05, stratify=None)
    
    return (train_x, train_y), (dev_x, dev_y), (val_x, val_label), (test_x, test_label)

def enum(labels):
    one_hot_labels = []
    for sentence in labels:
        one_hot_sentence = []
        for label in sentence:
            if label=='B':
                one_hot_sentence.append(1)
            elif label=='I':
                one_hot_sentence.append(2)
            elif label=='E':
                one_hot_sentence.append(3)
            elif label=='S':
                one_hot_sentence.append(4)
            else:
                print("Invalid Label: {}".format(label))
        one_hot_labels.append(np.array(one_hot_sentence))
    return np.array(one_hot_labels)

def one_hot(labels):
    lB, lI, lE, lS = np.eye(4)
    one_hot_batch = []
    for b in labels:
        one_hot_sentence = []
        for l in b:
            if not l:
                one_hot_sentence.append(np.zeros(4))
            else:
                one_hot_sentence.append(np.eye(4)[l-1])
        one_hot_batch.append(np.array(one_hot_sentence))
    return np.array(one_hot_batch)

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


lr = 0.001
in_prob = 0.8
rnn_prob = 0.7
req_prob = 0.7
reg = 0.001
epochs = 4
batch_size = 128
model_num = 7

if __name__ == '__main__':
    train_set = np.load('datasets/msr_training.npy')
    gold_set = np.load('datasets/msr_test_gold.npy')
    test_set = np.load('datasets/msr_test.npy')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    (train_x, train_label), (dev_x, dev_y), (val_x, val_label), (test_x, test_label) = create_dataset(train_set, gold_set, test_set, 30)

    with open('datasets/msr_training_unicab.json', 'r') as f:
        unicab = json.load(f)
        f.close()
    with open('datasets/msr_training_bicab.json', 'r') as f:
        bicab = json.load(f)
        f.close()
    unicab_size = len(unicab.keys())
    bicab_size = len(bicab.keys())

    
    tf.reset_default_graph()
    with tf.variable_scope("seg1"):
        segmenter1 = WordSegmenter(unicab_size, bicab_size, [64, 16], 256, 4, device='/gpu:0')

    
    n_iterations = int(np.ceil(len(train_x)/batch_size))
    n_dev_iterations = int(np.ceil(len(dev_x)/batch_size))
    tr_losses = []
    tr_acc = []
    dev_losses = []
    dev_accs = []
    val_losses = []
    val_accs = []

    config = tf.ConfigProto(allow_soft_placement = True)
    # config = tf.ConfigProto()
    # config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        print("Starting training sequence.")
        sess.run(tf.global_variables_initializer())
        
        for epoch in range(epochs):
            epoch_loss, epoch_acc = 0., 0.
            mb = 0
            for batch_x, batch_y in batch_generator(train_x, train_label, batch_size, shuffle=False):
                mb += 1
                unigrams, bigrams = batch_x[:,0], batch_x[:, 1]
                labels = one_hot(batch_y)
                _, loss_val, acc_val = segmenter1.train(sess, unigrams, bigrams, labels, learning_rate=lr, input_prob=in_prob, rnn_prob=rnn_prob, req_prob=req_prob, reg=reg)
                # Accumulate loss and acc as we scan through the dataset
                tr_losses.append(loss_val)
                tr_acc.append(acc_val)
                epoch_loss += loss_val
                epoch_acc += acc_val
                print("{:.2f}%\tEpoch:{:.2f}\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f} ".format(100.*mb/n_iterations, epoch, epoch_loss/mb, epoch_acc/mb), end="\r")
            # lr -= (0.03-0.001)/epochs
            epoch_loss /= n_iterations
            epoch_acc /= n_iterations

            print("\n")
            print("Train Loss: {:.4f}\tTrain Accuracy: {:.4f}".format(epoch_loss, epoch_acc))
            print("======="*10)

            dev_loss, dev_acc = 0.0, 0.0
            for batch_x, batch_y in batch_generator(dev_x, dev_y, batch_size):
            # Don't run train_op, set keep_prob to 1.0
            # No dropout should happen when we are predicting and/or evaluating
                unigrams, bigrams = batch_x[:,0], batch_x[:, 1]
                labels = one_hot(batch_y)
                loss_val, acc_val = segmenter1.evaluate(sess, unigrams, bigrams, labels, input_prob=in_prob, rnn_prob=rnn_prob, req_prob=req_prob, reg=reg)
                dev_losses.append(loss_val)
                dev_accs.append(acc_val)
                dev_loss += loss_val
                dev_acc += acc_val
            dev_loss /= n_dev_iterations
            dev_acc /= n_dev_iterations
            # kp += (0.9 - 0.6)/epochs
            segmenter1.save(sess, 'models/msr02')

            print("\nDev Loss: {:.4f}\tDev Accuracy: {:.4f}".format(dev_loss, dev_acc))
        np.save('tr_losses{}'.format(model_num), tr_losses)
        np.save('tr_acc{}'.format(model_num), tr_acc)
        np.save('dev_losses{}'.format(model_num), dev_losses)
        np.save('dev_acc{}'.format(model_num), dev_accs)


        print("\nTraining on validation set...")
        n_val_iterations = int(np.ceil(len(test_x)/batch_size))
        val_loss, val_acc = 0.0, 0.0
        mb = 0
        for batch_x, batch_y in batch_generator(val_x, val_label, batch_size):
            unigrams, bigrams = batch_x[:,0], batch_x[:, 1]
            mb += 1
            labels = one_hot(batch_y)
            _, loss_val, acc_val = segmenter1.train(sess, unigrams, bigrams, labels, learning_rate=0.001, input_prob=0.9, rnn_prob=0.9, req_prob=0.9, reg=reg)
            # Accumulate loss and acc as we scan through the dataset
            val_losses.append(loss_val)
            val_accs.append(acc_val)
            val_loss += loss_val
            val_acc += acc_val
            print("{:.2f}%\tEpoch:{:.2f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f} ".format(100.*mb/n_val_iterations, epoch, val_loss/mb, val_acc/mb), end="\r")
        np.save('val_losses{}'.format(model_num), val_losses)
        np.save('val_acc{}'.format(model_num), val_accs)
        print("\nEvaluating test...")
        n_test_iterations = int(np.ceil(len(test_x)/batch_size))
        test_loss, test_acc = 0.0, 0.0
        for batch_x, batch_y in batch_generator(test_x, test_label, batch_size):
            unigrams, bigrams = batch_x[:,0], batch_x[:, 1]
            labels = one_hot(batch_y)
            loss_val, acc_val = segmenter1.evaluate(sess, unigrams, bigrams, labels, input_prob=1.0, rnn_prob=1.0, req_prob=1.0, reg=reg)
            test_loss += loss_val
            test_acc += acc_val
        test_loss /= n_test_iterations
        test_acc /= n_test_iterations
        print("\nTest Loss: {:.4f}\tTest Accuracy: {:.4f}".format(test_loss, test_acc))