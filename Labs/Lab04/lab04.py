# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:19:58 2024

@author: anjal
"""

import numpy as np
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def covariance(D, mu):
    return ((D-mu) @ (D-mu).T) / float(D.shape[1])

#function for computing mean and Co-variance of the matrix
def compute_mu_C(D):
    mu = vcol(D.mean(1)) #mu is the numpy aray of shape (M, 1)
    C = covariance(D, mu)
    return mu, C

def logpdf_GAU_ND(X, mu, C):
    M = X.shape[0] #Size of feature vector x
    # N = X.shape[1]
    
    # Compute the inverse of the covariance matrix(Presicion)
    C_inv = np.linalg.inv(C)
    
    # Compute the log determinant of the covariance matrix
    sign, log_det = np.linalg.slogdet(C)
    
    # Compute the constant term
    constant_term = -0.5 * (M * np.log(2 * np.pi) + log_det)
    
    # Compute the difference between X and mu
    diff = (X - mu)
    
    # Compute the exponent term(ravel is used to change a 2-dimensional array or a multi-dimensional array into a contiguius flattened array)
    #examle X=([1,2,3], [4,5,6]) then ravel(X)=> X=[1 2 3 4 5 6]
    exponent = -0.5 * (diff * (C_inv @ diff)).sum(0)  
    
    # Compute the log density
    log_density = constant_term + exponent
    
    return log_density



def loglikelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()

def logpdf_GAU_ND_slow(X, mu, C):
    ll = [logpdf_GAU_ND(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return np.array(ll).ravel()
#done using broadcasting
def logpdf_GAU_ND_fast(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)



if __name__ == '__main__':

    # check if our density is correct using the give data(so the difference between provided solution and our solution will ne 0)
    plt.figure()
    XPlot = np.linspace(-8, 12, 1000)
    m = np.ones((1,1)) * 1.0
    C = np.ones((1,1)) * 2.0
    X=vrow(XPlot)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(X, m, C)))
    plt.show()
    
    #for the comparision of the recived solution with the one in the file
    pdfSol = np.load('llGAU.npy')
    pdfGau = logpdf_GAU_ND(X, m, C)
    print(np.abs(pdfSol - pdfGau).max())

    
    
    # #check for the density for th emulti-dimensional case using sample
    XND = np.load('XND.npy')
    mu = np.load('muND.npy')
    C = np.load('CND.npy')
    pdfSol = np.load('llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    print(np.abs(pdfSol - pdfGau).max())
    # pdfGau = logpdf_GAU_ND_fast(XND, mu, C)
    # print(np.abs(pdfSol - pdfGau).max())
    
    m_ML, C_ML = compute_mu_C(XND)
    print(m_ML)
    print(C_ML)
    print(loglikelihood(XND, m_ML, C_ML))
    
    X1D = np.load('X1D.npy')
    m_ML, C_ML = compute_mu_C(X1D)
    print(m_ML)
    print(C_ML)
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND_slow(vrow(XPlot), m_ML, C_ML)))
    
    ll = loglikelihood(X1D, m_ML, C_ML)
    print(ll)
