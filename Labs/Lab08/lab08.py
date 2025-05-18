# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:27:06 2024

@author: anjal
"""

import numpy
import numpy as np
import scipy.special
import sklearn.datasets

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

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']    
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L




#################################################################

#Step 1 suggestion using the explicit gradient value
def f_with_grad(x):
    y, z = x[0], x[1]
    value = (y + 3)**2 + np.sin(y) + (z + 1)**2
    grad_y = 2 * (y + 3) + np.cos(y)
    grad_z = 2 * (z + 1)
    gradient = np.array([grad_y, grad_z])
    return value, gradient

#Step 2 using the apprximate gradient
def f_obj(x):
    y, z = x[0], x[1]
    return (y + 3)**2 + np.sin(y) + (z + 1)**2

def train_logreg(DTR, LTR, l):
    Z = 2 * LTR - 1  # Convert labels to -1, +1
    n = DTR.shape[1]
    d = DTR.shape[0]

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        S = (np.dot(w.T, DTR) + b).ravel()
        loss = np.logaddexp(0, -Z * S).mean()
        reg = 0.5 * l * np.dot(w, w)
        J = reg + loss

        # Gradient computation
        G = -Z / (1.0 + np.exp(Z * S))
        grad_w = l * w + (np.dot(DTR, G.T) / n)
        grad_b = np.sum(G) / n
        gradient = np.hstack([grad_w, grad_b])
        return J, gradient

    x0 = np.zeros(d + 1)
    result = scipy.optimize.fmin_l_bfgs_b(
        func=logreg_obj,
        x0=x0,
        approx_grad=False
    )
    return result[0][:-1], result[0][-1]  # w, b

# STEP 4: Evaluation
def compute_accuracy(D, L, w, b):
    S = 1 / (1 + np.exp(-(np.dot(w.T, D) + b)))
    P = (S > 0.5).astype(int)
    acc = (P == L).mean()
    return acc

def log_likelihood(LTR):
    pi_emp = (LTR == 1).mean()
    log_odds = np.log(pi_emp / (1 - pi_emp))
    S_llr = S - log_odds
    
#cfn is false negative and cfp is false positive
def optimal_bayes_decision(LLR, prior, cfn, cfp):
    threshold = -np.log((prior * cfn) / ((1-prior) * cfp))
    return np.int32(LLR > threshold) 
    
def compute_confusion_matrix(L_pred, L_true, num_classes):
    CM = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(L_true, L_pred):
        CM[pred_label, true_label] += 1  # Rows: predicted, Columns: actual
    return CM

def empirical_bayes_risks(cf, prior, cfn, cpn, normalize=True):
    pfn = cf[0,1] / (cf[0,1] + cf[1,1])
    ppn = cf[1,0] / (cf[0,0] + cf[1,0])
    DCF = prior * cfn * pfn + (1-prior)*cpn*ppn
    if normalize:
        return DCF / np.minimum(prior * cfn, (1-prior)*cpn)
    return DCF

def compute_min_DCF(llr, labels, prior, cfn, cpn, dcf_norm):
    thresholds = np.concatenate([np.array([-np.inf]), llr[np.argsort(llr)], np.array([np.inf])])
    min_dcf = None
    dcf_th = None
    for t in thresholds:
        predictions = np.int32(llr > t)  # Binary decisions: 1 if LLR > t, else 0
        cm = compute_confusion_matrix(predictions, labels, 2)
        dcf = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
        if min_dcf is None or dcf < min_dcf:
            min_dcf = dcf
            dcf_th = t
    return min_dcf

def train_prior_weighted_logreg(DTR, LTR, l, pi_T):
    Z = 2 * LTR - 1  # ZTR
    nT = np.sum(LTR == 1)
    nF = np.sum(LTR == 0)
    xi = np.where(LTR == 1, pi_T / nT, (1 - pi_T) / nF)

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        S = np.dot(w.T, DTR) + b  # shape (N,)
        G = -Z / (1.0 + np.exp(Z * S))
        loss = (l / 2) * np.sum(w ** 2) + np.sum(xi * np.logaddexp(0, -Z * S))

        grad_w = l * w + np.dot(DTR, xi * G)
        grad_b = np.sum(xi * G)

        vgrad = np.hstack([grad_w, grad_b])
        return loss, vgrad

    x0 = np.zeros(DTR.shape[0] + 1)
    res, f, d = scipy.optimize.fmin_l_bfgs_b(
        func=logreg_obj,
        x0=x0,
        approx_grad=False
    )
    # Compute gradient at the solution
    return res[0:-1], res[-1], f  # return w, b
    

if __name__ == '__main__':
    
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    #Step 1 Explicit evaluation
    x0 = np.array([0.0, 0.0])
    x_min, f_min, info = scipy.optimize.fmin_l_bfgs_b(f_with_grad, x0)

    print("Minimum x:", x_min)
    print("Minimum value f(x):", f_min)
    print("Extra info:", info)
    
    x_min2, f_min2, info2 = scipy.optimize.fmin_l_bfgs_b(f_obj, x0, approx_grad = True)
    print("Minimum x:", x_min2)
    print("Minimum value f(x):", f_min2)
    print("Extra info:", info2)
    
    
    #Binary logistic regression
    for lambda_val in [1e-3, 0.1, 1]:
        print('Lamda: ', lambda_val)
        w, b = train_logreg(DTR, LTR, lambda_val)
        acc = compute_accuracy(DTE, LTE, w, b)
        print("Validation Accuracy:", acc)
        
        #score
        S = (w.T @ DTE + b).ravel()
        #error rate
        LP = (S > 0).astype(int)
        accuracy = (LP == LTE).mean()
        error_rate = 1 - accuracy
        print(f"The error rate is: {error_rate * 100:.2f}%")
    
        pi_emp = (LTR == 1).mean()
        log_odds = np.log(pi_emp / (1 - pi_emp))
        S_llr = S - log_odds
        
        prior = 0.5
        cfn = 1
        cpn = 1
    
        decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)

        cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
        # print(cm)

        dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
        print(f"Actual DCF : {dcf_norm}\n")
        
        #Minimum Detection Cost
        min_cost = compute_min_DCF(S_llr, LTE, prior, cfn, cpn, dcf_norm)
        print(f"min DCF: {min_cost}\n") 
    


    #Prior-weighted logistic regression ans calabration
    for lambda_val in [1e-3, 0.1, 1]:
        print('Lamda: ', lambda_val)
        pi_T = 0.8
        w, b, grad = train_prior_weighted_logreg(DTR, LTR, lambda_val, pi_T)
        acc = compute_accuracy(DTE, LTE, w, b)
        
        #score
        S = (w @ DTE + b).ravel()
        s_llr = S - np.log(pi_T / (1 - pi_T))
        prior = 0.5
        cfn = 1
        cpn = 1
        print("Gradient at optimal point:", grad)
        decisions = optimal_bayes_decision(s_llr ,pi_T , cfn, cpn)
        cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
        # print(cm)
        dcf_norm = empirical_bayes_risks(cm, pi_T, cfn, cpn, normalize=True)
        print(f"Actual DCF : {dcf_norm}\n")
        min_cost = compute_min_DCF(s_llr, LTE, pi_T, cfn, cpn, dcf_norm)
        print(f"min DCF: {min_cost}\n")
    

