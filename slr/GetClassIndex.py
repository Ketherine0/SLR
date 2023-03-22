def GetClassIndex(y):

    class_ind = []

    for i in range(1,5):

        ind_li =  [index for (index,value) in enumerate(y) if value == i]
        class_ind.append(ind_li)

    return class_ind