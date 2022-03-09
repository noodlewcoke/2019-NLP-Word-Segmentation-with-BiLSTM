import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
import json

def unigram_vocab(dataset):
    f_name = 'data/'+dataset
    input_name = f_name + '_simple.utf8'
    save_file = 'datasets/'+ dataset


    with open(input_name, 'r') as f:
        data = f.read()
        f.close()

    lines = data.split()
    vocab = sorted(list(set(''.join(lines))))

    uni_dict = {k:v+2 for v,k in enumerate(vocab)}
    uni_dict['<PAD>'] = 0
    uni_dict['<UNK>'] = 1
    with open(save_file+'_unicab.json', 'w') as f:
        json.dump(uni_dict, f)


def bigram_vocab(dataset):
    f_name = 'data/'+dataset
    input_name = f_name + '_simple.utf8'
    save_file = 'datasets/'+ dataset

    with open(input_name, 'r') as f:
        data = f.read()
        f.close()

    lines = data.split()

    bigrams = []
    for l in lines:
        for i in range(len(l)-1):
            bigrams.append(l[i:i+2])

    bigrams = sorted(list(set(bigrams)))
    bi_dict = {k:v+2 for v,k in enumerate(bigrams)}
    bi_dict['<PAD>'] = 0
    bi_dict['<UNK>'] = 1
    with open(save_file+'_bicab.json', 'w') as f:
        json.dump(bi_dict, f)


def training_set(data_name):
    f_name = 'data/' + data_name
    input_name = f_name + '_simple.utf8'
    label_name = f_name + '_labels.txt'
    save_file = 'datasets/' + data_name

    with open(input_name, 'r') as f:
        sentences = f.read()
        f.close()

    with open(label_name, 'r') as f:
        labels = f.read()
        f.close()
    
    with open(save_file+'_unicab.json', 'r') as f:
        unicab = json.load(f)
        f.close()
    
    with open(save_file+'_bicab.json', 'r') as f:
        bicab = json.load(f)
        f.close()

    sentences = sentences.split()
    labels = labels.split()
    dataset = []
    for sentence, label in zip(sentences, labels):
        assert len(sentence)==len(label), "Unmatching sentence and labels!"
        uniset = np.array([unicab[w] for w in sentence])
        biset = np.array([bicab[sentence[i:i+2]] for i in range(len(sentence)-1)])
        dataset.append((uniset, biset, label))
        
    np.save(save_file, dataset)

def test_set(data_name, unicab_file, bicab_file):
    f_name = 'data/' + data_name
    input_name = f_name + '_simple.utf8'
    label_name = f_name + '_labels.txt'
    save_file = 'data/' + data_name

    with open(input_name, 'r') as f:
        sentences = f.read()
        f.close()

    with open(label_name, 'r') as f:
        labels = f.read()
        f.close()
    
    with open(unicab_file, 'r') as f:
        unicab = json.load(f)
        f.close()
    
    with open(bicab_file, 'r') as f:
        bicab = json.load(f)
        f.close()

    sentences = sentences.split()
    labels = labels.split()
    dataset = []
    u_count = 0
    b_count = 0
    for sentence, label in zip(sentences, labels):
        assert len(sentence)==len(label), "Unmatching sentence and labels!"
        uniset = []
        for w in sentence:
            try:
                uniset.append(unicab[w])
            except KeyError:
                uniset.append(1) # 1 for <UNK> i.e. OOV word
                u_count += 1
        # uniset = np.array([unicab[w] for w in sentence])
        biset = []
        for i in range(len(sentence)-1):
            try:
                biset.append(bicab[sentence[i:i+2]])
            except KeyError:
                biset.append(1) # 1 for <UNK> i.e. OOV word
                b_count += 1

        # biset = np.array([bicab[sentence[i:i+2]] for i in range(len(sentence)-1)])
        dataset.append((np.array(uniset), np.array(biset), label))
    
    print("OOVs in uniset: {}, biset: {}".format(u_count, b_count))
    np.save(save_file, dataset)


def neighboorhood(dataset, win_size, save_name):
    data = []
    for sentence, _ in dataset:
        for index, word in enumerate(sentence):
            for nword in sentence[max(index - win_size, 0) : min(index + win_size, len(sentence))+1]:
                if nword != word:
                    data.append((word, nword))
    data = np.array(data)
    np.save(save_name, data)




if __name__ == '__main__':
    # dataset = np.load('datasets/msr_training.npy')
    # neighboorhood(dataset, 3, 'datasets/win3/msr_data')
    # print("MSR Dataset Completed")
    # dataset = np.load('datasets/as_training.npy')
    # neighboorhood(dataset, 3, 'datasets/win3/as_data')
    # print("AS Dataset Completed")
    # dataset = np.load('datasets/cityu_training.npy')
    # neighboorhood(dataset, 3, 'datasets/win3/cityu_data')
    # print("CITYU Dataset Completed")
    # dataset = np.load('datasets/pku_training.npy')
    # neighboorhood(dataset, 3, 'datasets/win3/pku_data')
    # print("PKU Dataset Completed")
    # unigram_vocab('msr_training')
    # bigram_vocab('msr_training')
    # training_set('msr_training')
    test_set('msr_test1', 'datasets/msr_training_unicab.json', 'datasets/msr_training_bicab.json')

    pass