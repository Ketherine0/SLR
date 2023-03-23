def GetClassIndex(y):
    '''
    Input:
    y:      list of labels

    Output:
    class_ind:      list consists of indexes of each class
    '''

    class_ind = []

    for i in range(1,5):

        ind_li =  [index for (index,value) in enumerate(y) if value == i]
        class_ind.append(ind_li)

    return class_ind