import numpy as np
from FastSolver import FastSolver

def ccSolveModel(X_train, y_train, X_test, num_classes, global_max_iter, lasso_max_iter, alpha):
    '''
    Input:
    X_train:                training frames being stacked column-wise
    y_train:                training labels for each frame
    X_test:                 test sample
    num_classes:            total number of classes
    global_max_iter:        FastSolver hyperparameter
    lasso_max_iter:         FastSolver hyperparameter
    alpha:                  FastSolver hyperparameter

    Output:
    nearest_class_index:    label assigned to test sample
    X_recovered:            sparse matrix
    L_recovered:            low-rank matrix     
    '''
    X_recovered, L_recovered = FastSolver(X_test, X_train, alpha, global_max_iter, lasso_max_iter)
    nearest_class_distance = np.Inf
    nearest_class_index = -1
    print(X_train.shape)
    for i in range(1, num_classes+1):
        print(y_train)
        idx = [i for i,x in enumerate(y_train) if x==i]
        print(idx)
        class_matrix = X_train[:, idx]
        print(class_matrix.shape)
        print(X_recovered[y_train == i,:].shape)
        class_representation = class_matrix @ X_recovered[y_train == i,:]
        class_representation += L_recovered
        error = np.linalg.norm(class_representation - X_test)
        if error < nearest_class_distance:
            nearest_class_distance = error
            nearest_class_index = i
    return nearest_class_index, X_recovered, L_recovered