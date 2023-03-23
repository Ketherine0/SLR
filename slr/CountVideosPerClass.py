import pandas as pd
import numpy as np
import os

def CountVideosPerClass(label):

    '''
    Input:
    label:          list of labels y

    Output:
    count_label:        number of labels for each class in a list
    '''

    count_label = []
    for i in range(4):
        count_label.append(np.sum(np.array(label)==(i+1)))

    return count_label
