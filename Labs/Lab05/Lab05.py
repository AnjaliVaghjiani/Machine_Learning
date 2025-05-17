# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:49:10 2024

@author: anjal
"""

import sklearn.datasets
import numpy as np
import scipy.special


# In this case the data set is already available from the sklearn library
# here the data is not the file name its just some argument about which have to find out what it is
# D=model training and L = ecalutaion(validation set)

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)
#loading the iris data using skklearn library
def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

#splitting the data for training and validation
def split_db_2tot(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    
    return (DTR, LTR), (DTE, LTE)
#computation  of log densities  (likelihood) for each classes
#storing the class-conditional probabilities in S(Score) matrix
def loglikelihood(DTE, data):
    S = np.zeros((len(data), DTE.shape[1]))
    for i in range(S.shape[0]):
        #data[i][0] is mean and data[i][1] is covaraince
        S[i, :] = logpdf_GAU_ND(DTE, data[i][0], data[i][1])
    return S

def ML_estimares_class(LTR, DTE):
    data = {}
    for i in set(LTR):
       DX = DTE[:, LTR==i]
       mu, C = compute_mu_C(DX)
       data[i] = compute_mu_C(DX)
       # print(mu,C) 
    return data   

#computation of class posteriror probability by combining the score matrix with priory information
def compute_class_posterior(S, value):
    SJoint = S + vcol(np.log(value))
    # s = np.load('logSJoint_MVG.npy')
    # print("S is : ", s)
    # print("SJoint is: ", SJoint)
    #summing over all classes the joint probability
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    # m = np.load('logMarginal_MVG.npy')
    # print("M is : ", m)
    # print("logMarginal is: ", SMarginal)
    # SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint - SMarginal

    return SPost

def naive_ML_estimation(LTR, DTE):
    data ={}
    for i in set(LTR):
        DX = DTE[:, LTR==i]
        mu, C = compute_mu_C(DX)
        data[i] = (mu, C * np.eye(DTE.shape[0]))
        
    return data    

def tied_ML_estimation(LTR, DTE):
    data={}
    CGlobal = 0
    m = {}
    for i in set(LTR):
        DX = DTE[:, LTR==i]
        mu, C = compute_mu_C(DX)
        CGlobal += C * DX.shape[1]
        m[i] = mu
    CGlobal = CGlobal/D.shape[1]
    for i in set(LTR):
        data[i] = (m[i], CGlobal)
    return data

if __name__ == '__main__':
    D, L = load_iris()
    
    #DTR and LTR are traininig and DTE and LTE as evaluation
    (DTR, LTR), (DTE, LTE) = split_db_2tot(D,L)

    data = ML_estimares_class(LTR, DTR);

    # computation of the log-density for all test sample   
    S_logLikelihood = loglikelihood(DTE, data);
    # print(S_logLikelihood)
    
    #Here for prior  probabity for each class we are assuming that each class correspond to 1/3
    S_logPost = compute_class_posterior(S_logLikelihood, 1/3)
    # print(S_logPost)
    
    
#     #error rate
    print ("Max absolute error w.r.t. pre-computed solution - log-posterior matrix")
    print (np.abs(S_logPost - np.load('logPosterior_MVG.npy')).max())

    #get predicted label
    predicted_label = S_logPost.argmax(0)  
    print("MVG - Error rate: %.1f%%" % ((predicted_label != LTE).sum() / float(LTE.size)*100))
    
    # Naive Bayes Gaussian Classifier
    data_naive = naive_ML_estimation(LTR, DTR)
    S_logLikelihood = loglikelihood(DTE, data_naive)
    S_logPost = compute_class_posterior(S_logLikelihood, 1/3)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))
        
    print()
    
    #Tied Covariance Gaussian Classifier
    data_tied = tied_ML_estimation(LTR, DTR)
    S_logLikelihood = loglikelihood(DTE, data_tied)
    S_logPost = compute_class_posterior(S_logLikelihood, 1/3)
    PVAL = S_logPost.argmax(0)
    print("Tied Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))
        
    print()
    
    #Binary tasks: log-likelihood ratios amd MVG
    D = D[:, L !=0]
    L = L[L !=0]
    (DTR, LTR), (DTE, LTE) = split_db_2tot(D, L)
    data = ML_estimares_class(LTR, DTR)
    LLR = logpdf_GAU_ND(DTE, data[2][0], data[2][1]) - logpdf_GAU_ND(DTE, data[1][0], data[1][1]) # Compute LLRs
    #prediction
    PVAL = np.zeros(DTE.shape[1], dtype=np.int32) #initaializing PVAL to 0. DTE.shape[1] gives the number of evaluation samples.
    TH = 0 #decision threshold which assumes the binary classification
    PVAL[LLR >= TH] = 2 #If the log-likelihood ratio is greater than or equal to 0, then the prediction is class 2
    PVAL[LLR < TH] = 1 #If the LLR is less than 0, the prediction is class 1 (negative class).
    print("MVG - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))     
    
    #Tied model
    data = tied_ML_estimation(LTR, DTR)
    LLR = logpdf_GAU_ND(DTE, data[2][0], data[2][1]) - logpdf_GAU_ND(DTE, data[1][0], data[1][1])

    PVAL = np.zeros(DTE.shape[1], dtype=np.int32)
    TH = 0
    PVAL[LLR >= TH] = 2
    PVAL[LLR < TH] = 1
    print("Tied - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100)) 


      
