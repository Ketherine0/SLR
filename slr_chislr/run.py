# from GenerateDataMatrix2 import GenerateDataMatrix

### switch to the first method of generating data
### Give each video a label
from GenerateDataMatrix1 import GenerateDataMatrix
from CountVideosPerClass import CountVideosPerClass
from GetClassIndex import GetClassIndex
from ccSolveModel import ccSolveModel
from Normalize import Normalize
from sklearn.metrics import confusion_matrix, accuracy_score
import platform
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


path = "../../SLR/stroke_data/data_new"
# X, y = GenerateDataMatrix(path,30)
X, y = GenerateDataMatrix(path)

### Use second generation method
#X, y = GenerateDataMatrix(path, 10)
print('Generating Train and Test Indices \n')
num_videos_per_class=CountVideosPerClass(y)
class_idx =  GetClassIndex(y)

# train_ratio = 0.8
# test_ratio = 1-train_ratio
# train_indices = []
# test_indices = []
#
# train_x = np.zeros((1))
# test_x = np.zeros((1))
# train_y = []
# test_y = []
# begin = False
train_sample_per_class = 2000
# test_frame_per_sample = 10
# test_sample_per_class = 120


# for i in range(4):
#     train_idx = random.sample(class_idx[i], k=round(len(class_idx[i]) * 0.5))
#     test_idx = list(set(class_idx[i]) - set(train_idx))
#     train_indices.append(train_idx)
#     test_indices.append(test_idx)
#     train_y = (train_y+(np.array(y)[train_idx][:train_sample_per_class].tolist()))
#     test_y = (test_y + (np.array(y)[test_idx][:test_sample_per_class*test_sample_per_class].tolist()))
#     if not begin:
#         # train_x = X[:,train_idx]
#         # test_x = X[:,test_idx]
#         train_x = X[:,train_idx][:,:train_sample_per_class]
#         test_x = X[:,test_idx][:,:test_frame_per_sample*test_sample_per_class]
#         begin = True
#     else:
#         # train_x = np.hstack((train_x, X[:,train_idx]))
#         # test_x = np.hstack((test_x, X[:, test_idx]))
#         train_x = np.hstack((train_x, X[:,train_idx][:,:train_sample_per_class]))
#         test_x = np.hstack((test_x, X[:, test_idx][:,:test_frame_per_sample*test_sample_per_class]))


# dictionary = Normalize(train_x)
# test_x = Normalize(test_x)
# dictionary = train_x

# global_max_iter=150
global_max_iter=120
lasso_max_iter=100
alpha = 5
delta = 30
lambdaG = 20
# confussion_matrix=np.zeros((7,7))



def run(test_frame_per_sample,test_sample_per_class, global_max_iter):
    train_ratio = 0.5
    test_ratio = 1 - train_ratio
    train_indices = []
    test_indices = []

    train_x = np.zeros((1))
    test_x = np.zeros((1))
    train_y = []
    test_y = []
    begin = False

    for i in range(4):
        train_idx = random.sample(class_idx[i], k=round(len(class_idx[i]) * 0.5))
        test_idx = list(set(class_idx[i]) - set(train_idx))
        train_indices.append(train_idx)
        test_indices.append(test_idx)
        train_y = (train_y + (np.array(y)[train_idx][:train_sample_per_class].tolist()))
        test_y = (test_y + (np.array(y)[test_idx][:test_sample_per_class * test_sample_per_class].tolist()))
        if not begin:
            # train_x = X[:,train_idx]
            # test_x = X[:,test_idx]
            train_x = X[:, train_idx][:, :train_sample_per_class]
            test_x = X[:, test_idx][:, :test_frame_per_sample * test_sample_per_class]
            begin = True
        else:
            # train_x = np.hstack((train_x, X[:,train_idx]))
            # test_x = np.hstack((test_x, X[:, test_idx]))
            train_x = np.hstack((train_x, X[:, train_idx][:, :train_sample_per_class]))
            test_x = np.hstack((test_x, X[:, test_idx][:, :test_frame_per_sample * test_sample_per_class]))
    dictionary = train_x

    summary = pd.DataFrame(columns=['Match', 'Real'])
    num_correct_classified = 0
    num_experiments_run = 0
    num_class = 4
    print("Generating")
    idx = 0
    T1 = time.process_time()
    error_total = []
    for i in range(2,num_class+1):
        test_data = test_x[:, (i - 1) * test_frame_per_sample*test_sample_per_class:(i) * test_frame_per_sample*test_sample_per_class]
        for j in range(test_sample_per_class):
            test_frame = test_data[:,j*test_frame_per_sample:(j+1)*test_frame_per_sample]
            matched_label, Xr, Lr, error = ccSolveModel(dictionary, train_y, test_frame, num_class, global_max_iter, lasso_max_iter, alpha, lambdaG, delta)
            idx += 1
            print("Sample: %d"%idx)
            print('Label: Matched %d - Real %d \n'%(matched_label,i))
            if (matched_label==i):
                num_correct_classified += 1
            summary.loc[num_experiments_run] = [matched_label,i]
            num_experiments_run += 1
            # pd.DataFrame(error).to_csv("Error"+str(i)+".csv")
            print('Partial Recognition Rate = %f \n'%(num_correct_classified / num_experiments_run))
            error_total.append(error)
        # plt.plot(np.arange(len(error) - 1), np.log10(np.array(error[:-1] - error[-1])), 'b')
        # plt.xlabel("Number of iteration")
        # plt.ylabel("Error")
        # plt.show()
        # T2 = time.process_time()
        # print('FastSolver' % ((T2 - T1)*1000))
    return num_correct_classified / num_experiments_run, np.sum(error_total)/len(error_total), summary



per_sample = [10,15,20]
per_class = [40,60,80]
global_li = [80,100,130]

err_total = []
for i in per_sample:
    for j in per_class:
        for z in global_li:
            a,b,c = run(i,j,z)
            err_total.append(a)

# a,b, summary = run(alpha,lambdaG,global_max_iter)
err_total.to_csv("summary_acc.csv")

# print(err_total)
# err_total_pd = pd.DataFrame(err_total)
# err_total_pd.to_csv("acc_summary3.csv")