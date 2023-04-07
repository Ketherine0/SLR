import numpy as np
from shrink import shrink

def FastIllinoisSolver(DtD, Dtb, x, D, f, tauInv, betaTauInv, lasso_max_iter):
    '''
    Solve lasso problem: x_{k+1}= argmin x : (2/beta)||x||_1 + ||b-Dx||_2^2

    Input:
    DtD:                D.T @ D
    Dtb:                D.T @ b
    x:                  initialization of x
    f:                  objective function taking value x
    tauInv:             1 / tau
    betaTauInv:         1 / (beta * tau)
    lasso_max_iter:     maximum number of iterations

    Output:
    x:  solution of lasso
    '''
    x = x.reshape((x.shape[0],1))
    Dtb = Dtb.reshape((x.shape[0],1))
    tol_apg = 1e-6 # tolerance
    t_prev = 1
    p = x
    p_prev = x
    for i in range(lasso_max_iter):
        x_prev = x
        grad = DtD@p - Dtb
        x = shrink(p - tauInv * grad, betaTauInv)
        t = (1 + np.sqrt(1 + 4*t_prev**2)) / 2
        p_prev = p
        p = x + ((t_prev-1) / t) * (x - x_prev)
        t_prev = t
        if np.linalg.norm(x_prev - x) < tol_apg * np.linalg.norm(x_prev):
            break
        #if np.dot(grad.T, x-x_prev) > 0:
        if f(x) > f(x_prev):
            p = p_prev
            x = x_prev
            t = 1
            t_prev = t
            continue
    x = np.squeeze(x)
    return x