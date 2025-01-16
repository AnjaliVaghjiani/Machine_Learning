############################################################################################
# Copyright (C) 2024 by Sandro Cumani                                                      #
#                                                                                          #
# This file is provided for didactic purposes only, according to the Politecnico di Torino #
# policies on didactic material.                                                           #
#                                                                                          #
# Any form of re-distribution or online publication is forbidden.                          #
#                                                                                          #
# This file is provided as-is, without any warranty                                        #
############################################################################################

import numpy
import scipy.special
import sklearn.datasets
import matplotlib.pyplot as plt
import bayesRisk
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def load_project_data(file_path):
    data = numpy.loadtxt(file_path, delimiter=',')
    D = data[:, :-1].T
    L = data[:, -1].astype(int)
    return D, L


# Optimize SVM
def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad
 
    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    # Primal loss
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    # Compute primal solution for extended data matrix
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

    # primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    dualityGap = primalLoss - dualLoss
    print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, dualityGap))
  
    
    return w, b

# We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# The inner function will be able to access the arguments of the outer function
def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc

# kernelFunc: function that computes the kernel matrix from two data matrices
def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    # Dual objective with gradient
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples
def plot_dcf_vs_C(C_values, min_dcf_values, act_dcf_values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, min_dcf_values, label='Min DCF, πT=0.1', linestyle='--', marker='o')
    plt.plot(C_values, act_dcf_values, label='Actual DCF, πT=0.1', marker='o')
    plt.xscale('log', base=10)
    plt.xlabel('C (Regularization Parameter)')
    plt.ylabel('Detection Cost Function (DCF)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    


if __name__ == '__main__':

    file_path = 'Traindata.txt'  # Replace with your actual file path
    D, L = load_project_data(file_path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # Define the range of C values (regularization parameter)
    C_values = numpy.logspace(-5, 0, 11)
    
    min_dcf_values = []
    act_dcf_values = []
    pi_t = 0.1  # Target prior
 
    # Loop over the C values
    for C in C_values:
        w, b = train_dual_SVM_linear(DTR, LTR, C, K=1.0)  # Train SVM with K=1.0
        SVAL = (vrow(w) @ DVAL + b).ravel()
        
        min_dcf = bayesRisk.compute_minDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        act_dcf = bayesRisk.compute_actDCF_binary_fast(SVAL, LVAL, 0.1, 1.0, 1.0)
        print("minDCF-: %f", min_dcf)
        print("actDCF-: %f", act_dcf)
        min_dcf_values.append(min_dcf)
        act_dcf_values.append(act_dcf)
    
    # Plotting the results
    plot_dcf_vs_C(C_values, min_dcf_values, act_dcf_values, "Linear SVM: DCF vs. Regularization Parameter (C)")

   
        
    