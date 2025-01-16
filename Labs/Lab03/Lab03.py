import numpy as np
import sklearn.datasets 
import matplotlib.pyplot as plt

import scipy.linalg

def load_iris():


    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1)) #if mean(0) then means of the rows and mean(1) is mean of the coloumn
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):

    mu, C = compute_mu_C(D)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    return P

def apply_pca(P, D):
    return P.T @ D
    
def compute_Sb_Sw(D, L):
    Sb = 0
    Sw = 0
    muGlobal = vcol(D.mean(1))
    for i in np.unique(L):
        DCls = D[:, L == i]
        mu = vcol(DCls.mean(1))
        Sb += (mu - muGlobal) @ (mu - muGlobal).T * DCls.shape[1]
        Sw += (DCls - mu) @ (DCls - mu).T
    return Sb / D.shape[1], Sw / D.shape[1]

def compute_lda_geig(D, L, m):
    
    Sb, Sw = compute_Sb_Sw(D, L)
    s, U = scipy.linalg.eigh(Sb, Sw)
    return U[:, ::-1][:, 0:m]

def compute_lda_JointDiag(D, L, m):

    Sb, Sw = compute_Sb_Sw(D, L)

    U, s, _ = np.linalg.svd(Sw)
    P = np.dot(U * vrow(1.0/(s**0.5)), U.T)

    Sb2 = np.dot(P, np.dot(Sb, P.T))
    U2, s2, _ = np.linalg.svd(Sb2)

    P2 = U2[:, 0:m]
    return np.dot(P2.T, P).T

def apply_lda(U, D):
    return U.T @ D


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)


def plot_histograms_pca(DTR_pca, LTR, DVAL_pca, LVAL):
    plt.figure(figsize=(8, 6))
    plt.hist(DTR_pca[0,LTR == 1], bins=5, alpha=0.5, label='Versicolor', density=True, color='orange')
    plt.hist(DTR_pca[0,LTR == 2], bins=5, alpha=0.5, label='Virginica ', density=True, color='green')
    # plt.xlabel('PCA Component')
    # plt.ylabel('Frequency')
    plt.title('Model training set (DTR, LTR)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(DVAL_pca[0,LVAL == 1], bins=5, alpha=0.5, label='Versicolor',density=True, color='orange')
    plt.hist(DVAL_pca[0, LVAL == 2], bins=5, alpha=0.5, label='Virginica',density=True, color='green')
    # plt.xlabel('PCA Component')
    # plt.ylabel('Frequency')
    plt.title('Validation set (DVAL, LVAL)')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
#Principal Component Analysis

    D, L = load_iris()
    # print( L)
    mu, C = compute_mu_C(D)
    P = compute_pca(D, m = 4)
    dp  = apply_pca(P, D)
    dp2 = apply_pca(-P, D) #with inverted eigenvalue to get the same result as prof graph
    label_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    
    #scatter for pca
    plt.figure(figsize=(12, 6))
    for label_value in np.unique(L):
        indices = np.where(L == label_value)
        plt.scatter(dp[0, indices], dp2[1, indices], label=label_names[label_value])
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('PCA Projection of IRIS Dataset')
    plt.legend()
    plt.show()
    
    
    #histogram for pca
    plt.figure(figsize=(10, 6))
    # plt.subplot(1, 2, 1)
    plt.hist(dp[0, L==0], bins=10, alpha=0.5, color='blue', density=True, label='Setosa')
    plt.hist(dp[0, L==1], bins=10, alpha=0.5, color='orange', density = True, label='Versicolor')
    plt.hist(dp[0, L==2], bins=10, alpha=0.5, color='green', density=True, label='Virginica')
    plt.title('PCA Component')
    plt.xlabel('Projected Value')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    
#Linear Discriminant Analysis

    U = compute_lda_geig(D, L, m = 2)
    D_lda = apply_lda(U, D)
    
    #scatter for lda
    plt.figure(figsize=(8, 6))
    for label_value in np.unique(L):
        indices = np.where(L == label_value)
        plt.scatter(D_lda[0, indices], D_lda[1, indices], label=label_names[label_value])
    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title('LDA Projection of IRIS Dataset')
    plt.legend()
    plt.show()
    
    #histogram for lda
    plt.figure(figsize=(10, 6))
    plt.hist(D_lda[0, L == 0], bins=10, alpha=0.5, label='Setosa', color='blue', density=True)
    plt.hist(D_lda[0, L == 1], bins=10, alpha=0.5, label='Versicolor', color='orange', density=True)
    plt.hist(D_lda[0, L == 2], bins=10, alpha=0.5, label='Virginica', color='green', density=True)
    plt.title('LDA Component')
    plt.xlabel('Projected Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
 
     
#PCA and LDA for classification   
    DIris, LIris = load_iris()
    D = DIris[:, LIris != 0]
    L = LIris[LIris != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    # Solution without PCA pre-processing and threshold selection. The threshold is chosen half-way between the two classes
    ULDA = compute_lda_JointDiag(DTR, LTR, m=1)

    DTR_lda = apply_lda(ULDA, DTR)
    
    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation.
    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==2].mean():
        ULDA = -ULDA
        DTR_lda = apply_lda(ULDA, DTR)

    DVAL_lda  = apply_lda(ULDA, DVAL)
    #model traning and validation set plots
    plot_histograms_pca(DTR_lda, LTR, DVAL_lda, LVAL)
    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0 # Estimated only on model training data

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    # print('Labels:     ', LVAL)
    # print('Predictions:', PVAL)
    # print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    # print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
          
    # Solution with PCA pre-processing with dimension m.
    m = 2
    UPCA = compute_pca(DTR, m = m) # Estimated only on model training data
    DTR_pca = apply_pca(UPCA, DTR)   # Applied to original model training data
    DVAL_pca = apply_pca(UPCA, DVAL) # Applied to original validation data

    ULDA = compute_lda_JointDiag(DTR_pca, LTR, m = 1) # Estimated only on model training data, after PCA has been applied

    DTR_lda = apply_lda(ULDA, DTR_pca)   # Applied to PCA-transformed model training data, the projected training samples are required to check the orientation of the direction and to compute the threshold
    # Check if the Virginica class samples are, on average, on the right of the Versicolor samples on the training set. If not, we reverse ULDA and re-apply the transformation
    if DTR_lda[0, LTR==1].mean() > DTR_lda[0, LTR==2].mean():
        ULDA = -ULDA
        DTR_lda = apply_lda(ULDA, DTR_pca)

    DVAL_lda = apply_lda(ULDA, DVAL_pca) # Applied to PCA-transformed validation data

    threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0 # Estimated only on model training data

    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 2
    PVAL[DVAL_lda[0] < threshold] = 1
    # print('Labels:     ', LVAL)
    # print('Predictions:', PVAL)
    # print('Number of erros:', (PVAL != LVAL).sum(), '(out of %d samples)' % (LVAL.size))
    # print('Error rate: %.1f%%' % ( (PVAL != LVAL).sum() / float(LVAL.size) *100 ))
    
    
 
# Display the plot
plt.show()
  