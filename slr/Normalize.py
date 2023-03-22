from sklearn import preprocessing

def Normalize(x):
    # Data normalization with l2 norm along column
    return x/preprocessing.normalize(x,axis=0)
