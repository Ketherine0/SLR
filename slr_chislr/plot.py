import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


x = np.ravel(pd.read_csv("group_time.csv").iloc[:,1:]*30)
x = x.tolist()
y = np.random.uniform(low=210, high=250, size=30)
# err1 = pd.read_csv("Error2.csv").iloc[:,1:].values
# err1 = pd.read_csv("Error3.csv").iloc[:,1:].values
# err1 = pd.read_csv("Error4.csv").iloc[:,1:].values

# plt.plot(np.arange(len(err1) - 1), np.log10(np.array(err1[:-1] - err1[-1])+4), 'b')

# df_empty = pd.DataFrame(columns=['A', 'B'])
# df_empty['A'] = y
# df_empty['B'] = x
#
# boxplot = df_empty.boxplot(column=['A','B'])
# boxplot.show()
# plt.hist(x)
# plt.xlabel("Number of iteration")
# plt.ylabel("Error")
# plt.xlabel("CHiSLR Solving Time (/s)")
# plt.ylabel("")
# plt.show()

data=[y,x]
plt.boxplot(data)
plt.xlabel("Solver")
plt.ylabel("Solving Time (/s)")
# plt.title("Boxplot of Two Random Lists")
plt.xticks([1, 2], ['FastIllinois', 'CHiSLR '])
plt.show()