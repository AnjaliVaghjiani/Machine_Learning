import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

def load(f):
    label = []
    x = []
    with open(f, 'r') as myData:
        for line in myData:
            features, bool_value = line.strip().split(",")[:-1], line.strip().split(",")[-1]
            features = np.array([float(i) for i in features])
            x.append(features)
            label.append(bool_value)
    x = np.array(x)
    return x.T, np.array(label, dtype=np.int32)

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C

def compute_pca(D, m):
    mu, C = compute_mu_C(D)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, :m]
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

def plot_histograms_pca(D, L):
    for i in range(D.shape[0]):
        plt.figure(figsize=(10, 6))
        plt.hist(D[i, L == 0], bins=20, alpha=0.5, label='Counterfeit', density=True, color='orange')
        plt.hist(D[i, L == 1], bins=20, alpha=0.5, label='Genuine', density=True, color='green')
        plt.xlabel('Projected Value')
        plt.ylabel('Frequency')
        plt.title(f'PCA Component {i+1}')
        plt.legend()
        plt.show()

def plot_histogram_lda(D_lda, L):
    plt.figure(figsize=(10, 6))
    plt.hist(D_lda[0, L == 0], bins=20, alpha=0.5, label='Counterfeit', density=True, color='orange')
    plt.hist(D_lda[0, L == 1], bins=20, alpha=0.5, label='Genuine', density=True, color='green')
    plt.title('LDA Component')
    plt.xlabel('Projected Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Load the data
    file_path = 'trainData.txt'  # Change this to your actual data file path
    D, L = load(file_path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Apply PCA and analyze the features
    P_pca = compute_pca(D, m=6)
    DTR_pca = apply_pca(P_pca, D)
    # DVAL_pca = apply_pca(P_pca, DVAL)

    plot_histograms_pca(DTR_pca, L)

    # Apply LDA (1-dimensional) and compute the histogram of the projected LDA samples
    P_lda = compute_lda_geig(D, L, m=1)
    DTR_lda = apply_lda(P_lda, D)
    

    # Plot histogram for LDA
    plot_histogram_lda(DTR_lda, L)
    
    
    #LDA as classifier
    DTR_lda = apply_lda(P_lda, DTR)
    DVAL_lda = apply_lda(P_lda, DVAL)
    
    # Ensure the orientation is correct
    if DTR_lda[0, LTR == 1].mean() < DTR_lda[0, LTR == 0].mean():
        P_lda = -P_lda
        DTR_lda = apply_lda(P_lda, DTR)
        DVAL_lda = apply_lda(P_lda, DVAL)
    
    # Compute the threshold
    threshold = (DTR_lda[0, LTR == 1].mean() + DTR_lda[0, LTR == 0].mean()) / 2
    
    # Predict the class labels for the validation set
    PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    
    # Compute and print the error rate
    error_rate = np.sum(PVAL != LVAL) / LVAL.size
    print(f'Error rate: {error_rate * 100:.2f}%')
    
    # Plot histogram for LDA with validation data
    plot_histogram_lda(DVAL_lda, LVAL)
    
    # Analyze different threshold values
    thresholds = np.linspace(DTR_lda[0].min(), DTR_lda[0].max(), 100)
    error_rates = []
    
    for th in thresholds:
        PVAL = np.zeros(shape=LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= th] = 1
        PVAL[DVAL_lda[0] < th] = 0
        error_rate = np.sum(PVAL != LVAL) / LVAL.size
        error_rates.append(error_rate)
        
    min_error_rate = min(error_rates)
    best_threshold = thresholds[np.argmin(error_rates)]
    print(f'Best threshold: {best_threshold:.2f}')
    print(f'Minimum error rate: {min_error_rate * 100:.2f}%')