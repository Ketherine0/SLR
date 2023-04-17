import numpy as np
from FastSolver import FastSolver
from DenseErrorSolver import DenseErrorSolver
from CHiSLRSolver import group_sparse_rep

def ccSolveModel(X_train, y_train, X_test, num_classes, global_max_iter, lasso_max_iter, alpha, lambdaG, delta = 0):
    '''
    Input:
    X_train:                training frames being stacked column-wise
    y_train:                training labels for each frame
    X_test:                 test sample
    num_classes:            total number of classes
    global_max_iter:        FastSolver hyperparameter
    lasso_max_iter:         FastSolver hyperparameter
    alpha:                  FastSolver hyperparameter
    delta:                  DenseErrorSolver hyperparameter

    Output:
    nearest_class_index:    label assigned to test sample
    X_recovered:            sparse matrix
    L_recovered:            low-rank matrix     
    '''
    X_recovered, L_recovered, error1 = FastSolver(X_test, X_train, alpha, global_max_iter, lasso_max_iter)
    #X_recovered, L_recovered = DenseErrorSolver(X_test, X_train, alpha, delta, global_max_iter, lasso_max_iter)
    # X_recovered, L_recovered, error1 = group_sparse_rep(X_test, X_train, y_train, alpha, lambdaG, global_max_iter)
    print(len(error1))
    nearest_class_distance = np.Inf
    nearest_class_index = -1
    for i in range(2, num_classes+1):
        idx = [index for index, element in enumerate(y_train) if element == i]
        class_matrix = X_train[:, idx]
        class_representation = class_matrix @ X_recovered[idx,:]
        class_representation += L_recovered
        error = np.linalg.norm(class_representation - X_test)
        print(error, i)
        if error < nearest_class_distance:
            nearest_class_distance = error
            nearest_class_index = i

    return nearest_class_index, X_recovered, L_recovered, error1