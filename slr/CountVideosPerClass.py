import pandas as pd
import numpy as np
import os

def CountVideosPerClass(label):

    # filename = []
    # for i in os.listdir(path):
    #     filename.append(path+"/"+i)
    #
    # data_name = []
    # for file in filename:
    #     d = []
    #     for pos in os.listdir(file):
    #         d.append(file+"/"+pos)
    #     data_name.append(d)
    #
    # x_file = "Joint_Positions.csv"
    # y_file = "Labels.csv"
    #
    # count_label = pd.DataFrame(np.zeros((1,4)),columns = ['No_comp','Lean_Fwd','Shoud_Ele','Trunk_Rot'])
    # labels = []
    #
    # for i in range(len(data_name)):
    #     for j in range(len(data_name[i])):
    #         y = pd.read_csv(data_name[i][j] + "/" + y_file, header=None)
    #         y_uni = np.unique(y)
    #         if len(y_uni)==1:
    #             labels.append([0]*y.shape[0])
    #             count_label.iloc[0,0]+=y.shape[0]
    #         elif len(y_uni)==2:
    #             labels.append(y)
    #             num = (y==1).sum()
    #             count_label.iloc[0, 0] += num
    #             count_label.iloc[0, y_uni[1]-1] += (len(y)-num)
    count_label = []
    for i in range(4):
        count_label.append(np.sum(np.array(label)==(i+1)))

    return count_label
