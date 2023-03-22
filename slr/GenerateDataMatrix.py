import os
import pandas as pd
import numpy as np


def GenerateDataMatrix(path):

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

    Data_x = np.zeros((1))
    labels = []
    begin = False

    for i in range(len(data_name)):
        for j in range(len(data_name[i])):
            x = pd.read_csv(data_name[i][j] + "/" + x_file, header=None)
            y = pd.read_csv(data_name[i][j] + "/" + y_file, header=None)
            labels = labels+list(y.iloc[:,0].values)
            if not begin:
                Data_x = x.values.reshape(75,-1)
                begin = True
            else:
                Data_x = np.hstack((Data_x,x.values.reshape(75,-1)))

    return Data_x, labels

# GenerateDataMatrix("../stroke_data/data_new")