from GenerateDataMatrix2 import GenerateDataMatrix

### switch to the first method of generating data
### Give each video a label
# from GenerateDataMatrix1 import GenerateDataMatrix
from CountVideosPerClass import CountVideosPerClass
from GetClassIndex import GetClassIndex
from ccSolveModel import ccSolveModel
from Normalize import Normalize
from sklearn.metrics import confusion_matrix, accuracy_score
import random
import numpy as np


path = "../stroke_data/data_new"
X, y = GenerateDataMatrix(path,30)

### Use second generation method
#X, y = GenerateDataMatrix(path, 10)
print('Generating Train and Test Indices \n')
num_videos_per_class=CountVideosPerClass(y)
class_idx =  GetClassIndex(y)

train_ratio = 0.8
test_ratio = 1-train_ratio
train_indices = []
test_indices = []

train_x = np.zeros((1))
test_x = np.zeros((1))
train_y = []
test_y = []
begin = False
train_sample_per_class = 2000
test_sample_per_class = 50


for i in range(4):
    train_idx = random.sample(class_idx[i], k=round(len(class_idx[i]) * 0.8))
    test_idx = list(set(class_idx[i]) - set(train_idx))
    train_indices.append(train_idx)
    test_indices.append(test_idx)
    train_y = (train_y+(np.array(y)[train_idx][:train_sample_per_class].tolist()))
    test_y = (test_y + (np.array(y)[test_idx][:test_sample_per_class].tolist()))
    if not begin:
        # train_x = X[:,train_idx]
        # test_x = X[:,test_idx]
        train_x = X[:,train_idx][:,:train_sample_per_class]
        test_x = X[:,test_idx][:,:test_sample_per_class]
        begin = True
    else:
        # train_x = np.hstack((train_x, X[:,train_idx]))
        # test_x = np.hstack((test_x, X[:, test_idx]))
        train_x = np.hstack((train_x, X[:,train_idx][:,:train_sample_per_class]))
        test_x = np.hstack((test_x, X[:, test_idx][:,:test_sample_per_class]))
    # print(train_x.shape)


dictionary = Normalize(train_x)

global_max_iter=600
lasso_max_iter=100
alpha = 10

global_max_iter=20
lasso_max_iter=200
alpha = 1
# confussion_matrix=np.zeros((7,7))
num_correct_classified=0
num_experiments_run=0
num_class = 4

print(test_x.shape)
print("Generating")
for i in range(num_class):
    for j in range(test_sample_per_class):
# matched_pred, Xr, Lr = ccSolveModel(dictionary, train_y, test_x, num_class, global_max_iter, lasso_max_iter, alpha)
# print('Label: Matched %d - Real %d \n',matched_pred)


conf_mat = confusion_matrix(test_y, matched_pred)
