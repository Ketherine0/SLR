from sklearn import preprocessing

def Normalize(x):
    '''
    Input:
    x:          Data matrix x

    Output:
    normalized version with l2 norm normalization along column
    '''
    # Data normalization with l2 norm along column
    return x/preprocessing.normalize(x,axis=0)
