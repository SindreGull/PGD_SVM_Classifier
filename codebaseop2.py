import numpy as np
from scipy.optimize import minimize

def f(alpha, A):
    return 0.5 * np.dot(alpha, A @ alpha) - np.sum(alpha)

def grad_f(alpha, A):
    return A @ alpha - np.ones_like(alpha)


def projection(beta, y, C, tol=1e-6, maxit=10000):

    #define the function g
    def g(lmbd):
        alpha_temp = np.clip(beta + lmbd * y, 0, C)
        return np.dot(y, alpha_temp)
    
    #initialize bracketing for λ.

    lambda_low, lambda_high = -1e5, 1e5
    while g(lambda_low) > 0:
        lambda_low *= 2
    while g(lambda_high) < 0:
        lambda_high *= 2

    #bisection loop 
    for _ in range(maxit):
        lmbd_mid = 0.5 * (lambda_low + lambda_high)
        g_mid = g(lmbd_mid)
        if np.abs(g_mid) < tol:
            break
        if g_mid > 0:
            lambda_high = lmbd_mid
        else:
            lambda_low = lmbd_mid

    #compute the projected vector 
    alpha_proj = np.clip(beta + lmbd_mid * y, 0, C)
    return alpha_proj


def barzilai_borwein(alpha_prev, alpha, grad_prev, grad):

    taumin = 1e-5
    taumax = 1e5
    s = alpha - alpha_prev
    z = grad - grad_prev
    s_dot_z = np.dot(s, z)
    if s_dot_z <= 1e-5:
        return taumax
    tau = np.dot(s, s) / s_dot_z
    return np.clip(tau, taumin, taumax)


def exact_linesearch(alpha, d, A):

    numerator = np.dot(np.ones_like(alpha), d) - np.dot(d, A @ alpha)
    denominator = np.dot(d, A @ d)
    if denominator <= 0:
        return 1.0
    theta = numerator / denominator
    return np.clip(theta, 0, 1)


def PGD(Y, G, C, tol=1e-5, maxit=10000):
    
    A = Y @ G @ Y
    #extract label vector from the diagonal of Y
    y = np.diag(Y)
    #initialize alpha at zero 
    alpha = np.zeros_like(y)
    tau = 1.0   # initial step length
    alpha_prev = alpha.copy()
    grad_prev = np.zeros_like(alpha)
    
    
    s = 0
    
    for iteration in range(maxit):
        grad = grad_f(alpha, A)
        #update step length using Barzilai–Borwein rule 
        if iteration > 0:
            tau = barzilai_borwein(alpha_prev, alpha, grad_prev, grad)
        
        #compute the gradient step
        beta = alpha - tau * grad
        #project the gradient step onto Ω using the improved projection
        alpha_proj = projection(beta, y, C, tol=1e-6, maxit=10000)
        #descent direction is the difference between the projected point and current alpha
        d = alpha_proj - alpha

        #exact line search along direction d
        theta = exact_linesearch(alpha, d, A)

        #update variables for next iteration
        alpha_prev_prev = alpha_prev.copy()
        alpha_prev = alpha.copy()
        grad_prev = grad.copy()
        alpha = alpha + theta * d
        
        s = iteration
        
        #check for convergence via the norm of the projected step
        if np.linalg.norm(d) < tol:
            break
        
        #if np.linalg.norm(alpha-alpha_prev_prev) < tol:
            #break

    print(s)    
    return alpha


def recoverwb(alphastar, x, y):
    support_indices = np.where(alphastar > 1e-5)[0]
    wstar = np.zeros(x.shape[1])
    for i in support_indices:
        wstar += alphastar[i] * y[i] * x[i]
    #compute b for support vectors
    b = y[support_indices] - x[support_indices] @ wstar
    bstar = np.mean(b)
    return wstar, bstar


def PGDsolver(x, y, C=10):
    Gram = x @ x.T
    Y = np.diag(y)
    alphastar = PGD(Y, Gram, C)
    wstar, bstar = recoverwb(alphastar, x, y)
    return wstar, bstar

