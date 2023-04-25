import pandas as pd
from sklearn.metrics import confusion_matrix

summary = pd.read_csv('summary.csv').iloc[:,1:]
y_pred = summary.iloc[:,0]
y_true = summary.iloc[:,1]
conf = confusion_matrix(y_true,y_pred)
print(conf)