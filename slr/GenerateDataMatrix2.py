import os
import pandas as pd
import numpy as np


def GenerateDataMatrix(path, num_frame):
    '''
    Second method for generating data: fixed number of frames as a whole and give each a label

    Input:
    path:       the path where data is stored (e.g. "../stroke_data/data_new")
    num_frame:      how many number of frames we treat as a total to label

    Output:
    Data_x:     Data x stacked column-wise
    labels:     list of label y
    '''

    filename = []
    for i in os.listdir(path):
        filename.append(path+"/"+i)

    data_name = []
    for file in filename:
        d = []
        for pos in os.listdir(file):
            d.append(file+"/"+pos)
        data_name.append(d)

    x_file = "Joint_Positions.csv"
    y_file = "Labels.csv"

    Data_x = np.zeros((75,1))
    labels = []
    begin = False

    for i in range(len(data_name)):
        for j in range(len(data_name[i])):
            x = pd.read_csv(data_name[i][j] + "/" + x_file, header=None)
            y = pd.read_csv(data_name[i][j] + "/" + y_file, header=None)

            split = x.values.reshape(75,-1).shape[1] // num_frame
            if not begin:
                Data_x = (x.values.reshape(75,-1))[:,:split*num_frame]
                begin = True
            else:
                Data_x = np.hstack((Data_x,(x.values.reshape(75,-1))[:,:split*num_frame]))

            for z in range(split):
                label = y.values[z*num_frame][0]
                labels = labels + [label]*num_frame

    return Data_x, labels
