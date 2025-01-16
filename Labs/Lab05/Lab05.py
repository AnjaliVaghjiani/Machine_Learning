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

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

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
        S[i, :] = logpdf_GAU_ND(DTE, data[i][0], data[i][1])
    return S

def ML_estimares_class(LTR, DTE):
    data = {}
    for i in np.unique(LTR):
       DX = DTR[:, LTR==i]
       mu, C = compute_mu_C(DX)
       data[i] = compute_mu_C(DX)
       # print(mu,C) 
    return data   

#computation of class posteriror probability by combining the score matrix with priory information
def compute_class_posterior(S, value):
    SJoint = S + vcol(np.log(value))
    #summing over all classes the joint probability
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    # SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint - SMarginal
    return SPost




def naive_ML_estimation(LTR, DTE):
    data ={}
    tmp ={}
    C_Total=0
    for i in DTE:
        DX = DTE[:, LTR==i]
        mu, C = compute_mu_C(DX)
        C_Total+= C * DX.shape[1]
        tmp[i] = mu
    C_Total = C_Total / LTR.shape[1]
    for i in DTE:
        data[i] = (tmp[i], C_Total)
        
    return data    

if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2tot(D,L)
   
    data = ML_estimares_class(LTR, DTE);
    
 # computation of the log-density for all test sample   
    S_logLikelihood = loglikelihood(DTE, data);
    # print(S_logLikelihood)
    
#Here for prior  probabity for each class we are assuming that each class correspond to 1/3
    S_logPost = compute_class_posterior(S_logLikelihood, 1/3)
    # print(S_logPost)
    
    
    #error rate
    # print (np.abs(S_logPost - np.load('logPosterior_MVG.npy')).max())

#get predicted label

    predicted_label = S_logPost.argmax(0)  
    
    #compute the predicted classes
    #For now accurracy :  the number of correctly labeled points over the number of evaluation samples
    total_correct = (predicted_label == LTE).sum()
    total_sample = float(LTE.size)
    accuracy = total_correct/total_sample
    print(accuracy)
    error = (1 - accuracy)*100
    print(error)
    
    
    #alternate way to compute the error
    
    
    # total_incorrect = (predicted_label != LTE).sum()
    # error2= total_incorrect/total_sample * 100
    # print(error2)
    
    
#compute the predicted classes
#For now accurracy :  the number of correctly labeled points over the number of evaluation samples



      
