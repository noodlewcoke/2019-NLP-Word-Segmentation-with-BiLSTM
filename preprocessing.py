import os 
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import numpy as np 
from hanziconv import HanziConv


def preprocess(input_file):

    in_file = open('data/predict_simple.utf8', 'a')
    label_file = open('data/predict_labels.txt', 'a')


    for line in open(input_file, 'r'):
        h = HanziConv.toSimplified(line)
        arr = h.split()
        label = str()
        for c in arr:
            if len(c)==1:
                label += "S"
            else:
                label += "B" + "I"*(len(c)-2) + "E"
        label_file.write(label+'\n')
        in_file.write(''.join(arr)+'\n')
        assert len(label)==len(''.join(arr)), "Mismatch"
    in_file.close()
    label_file.close()
