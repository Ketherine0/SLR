import pandas as pd
import numpy as np
import os

def GenerateDataMatrix(path):
    '''
    First method for generating data: treating a video as a whole and give same label

    Input:
    path:     the path where data is stored (e.g. "../stroke_data/data_new")

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

    labels = []
    begin = False

    for i in range(len(data_name)):
        for j in range(len(data_name[i])):
            x = pd.read_csv(data_name[i][j] + "/" + x_file, header=None)
            y = pd.read_csv(data_name[i][j] + "/" + y_file, header=None)
            y_uni = np.unique(y)
            if len(y_uni)!=3:
                if not begin:
                    Data_x = x.values.reshape(75,-1)
                    begin = True
                else:
                    Data_x = np.hstack((Data_x,x.values.reshape(75,-1)))

            if len(y_uni)==1:
                labels = labels+[0]*len(y)
            elif len(y_uni)==2:
                labels = labels+[y_uni[1]]*len(y)


    return Data_x, labels
