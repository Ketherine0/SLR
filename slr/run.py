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
import pandas as pd


path = "../../SLR/stroke_data/data_new"
X, y = GenerateDataMatrix(path,30)
# X, y = GenerateDataMatrix(path)

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
train_sample_per_class = 400
test_sample_per_class = 20


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
test_x = Normalize(test_x)
# dictionary = train_x

global_max_iter=30
lasso_max_iter=100
alpha = 1
# confussion_matrix=np.zeros((7,7))
num_correct_classified=0
num_experiments_run=0
num_class = 4


summary = pd.DataFrame(columns=['Match', 'Real'])
print("Generating")
for i in range(1,num_class):
    test_data = test_x[:,(i - 1) * test_sample_per_class:(i) * test_sample_per_class]
    for j in range(test_sample_per_class):
        test_x = test_data[:,j].reshape(-1,1)
        matched_label, Xr, Lr = ccSolveModel(dictionary, train_y, test_x, num_class, global_max_iter, lasso_max_iter, alpha)
        print('Label: Matched %d - Real %d \n'%(matched_label,i))
        if (matched_label==i):
            num_correct_classified += 1
        summary.loc[num_experiments_run] = [matched_label,i]
        num_experiments_run += 1
        print('Partial Recognition Rate = %f \n'%(num_correct_classified / num_experiments_run))
summary.to_csv("summary.csv")

# conf_mat = confusion_matrix(test_y, matched_pred)
