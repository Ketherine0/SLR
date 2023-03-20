import numpy as np
from shrink import shrink

def FastIllinoisSolver(DtD, Dtb, x, tauInv, betaTauInv, lasso_max_iter):
    '''
    Solve lasso problem: x_{k+1}= argmin x : (2/beta)||x||_1 + ||b-Dx||_2^2

    Input:
    DtD:                D.T @ D
    Dtb:                D.T @ b
    x:                  initialization of x
    tauInv:             1 / tau
    betaTauInv:         1 / (beta * tau)
    lasso_max_iter:     maximum number of iterations

    Output:
    x:  solution of lasso
    '''

    tol_apg = 1e-6 # tolerance
    t_prev = 1
    z = x
    for i in range(lasso_max_iter):
        x_prev = x
        x = shrink(z - tauInv * (DtD@z - Dtb), betaTauInv)
        if np.linalg.norm(x_prev - x) < tol_apg * np.linalg.norm(x_prev):
            break
        t = (1 + np.sqrt(1 + 4*t_prev**2)) / 2
        z = x + ((t_prev-1) / t) * (x - x_prev)
        t_prev = t
    return x