import numpy as np
import matplotlib.pyplot as plt
import scipy.special

def vcol(x):
    return x.reshape((x.size, 1))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

def load_project_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    D = data[:, :-1].T
    L = data[:, -1].astype(int)
    return D, L

def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

def logpdf_GAU_ND(X, mu, C):
    P = np.linalg.inv(C)  # Inverse of the covariance matrix
    const = -0.5 * X.shape[0] * np.log(2 * np.pi)  # Constant term
    log_det_C = -0.5 * np.linalg.slogdet(C)[1]  # Log determinant term
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        diff = x - mu
        exponent = -0.5 * (diff.T @ P @ diff)
        Y.append(const + log_det_C + exponent)
    return np.array(Y).ravel()

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

# Helper functions
def compute_effective_prior(pi1, Cfn, Cfp):
    return (pi1 * Cfn) / (pi1 * Cfn + (1 - pi1) * Cfp)

def compute_confusion_matrix(predictions, labels):
    TP = np.sum((predictions == 1) & (labels == 1))
    FP = np.sum((predictions == 1) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))
    TN = np.sum((predictions == 0) & (labels == 0))
    return np.array([[TN, FP], [FN, TP]])

def compute_dcf(cm, pi1, Cfn, Cfp):
    Pfn = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    Pfp = cm[0, 1] / (cm[0, 1] + cm[0, 0])
    return pi1 * Cfn * Pfn + (1 - pi1) * Cfp * Pfp

def compute_min_dcf(llr, labels, pi1, Cfn, Cfp):
    thresholds = np.sort(llr)
    dcf_values = []
    for threshold in thresholds:
        predictions = (llr > threshold).astype(int)
        cm = compute_confusion_matrix(predictions, labels)
        dcf = compute_dcf(cm, pi1, Cfn, Cfp)
        dcf_values.append(dcf)
    return min(dcf_values)

def compute_log_likelihood_Gau(D, hParams):
    n_classes = len(hParams)
    n_samples = D.shape[1]
    log_likelihoods = np.zeros((n_classes, n_samples))
    for i, (mu, C) in hParams.items():
        log_likelihoods[i, :] = logpdf_GAU_ND(D, mu, C)
    return log_likelihoods

def compute_logPosterior(S_logLikelihood, prior_array):
    log_joint = S_logLikelihood + np.log(prior_array).reshape(-1, 1)
    log_marginal = scipy.special.logsumexp(log_joint, axis=0)
    log_posterior = log_joint - log_marginal
    return log_posterior

# Function to evaluate models and compute DCF
def evaluate_model(llr, labels, pi1, Cfn, Cfp):
    predictions = (llr > 0).astype(int)  # Assuming binary classification with LLR
    cm = compute_confusion_matrix(predictions, labels)
    actual_dcf = compute_dcf(cm, pi1, Cfn, Cfp)
    min_dcf = compute_min_dcf(llr, labels, pi1, Cfn, Cfp)
    return actual_dcf, min_dcf

# Function to apply PCA
def apply_pca(D, m):
    mu = np.mean(D, axis=1, keepdims=True)
    DC = D - mu
    U, _, _ = np.linalg.svd(DC, full_matrices=False)
    P = U[:, :m]
    DP = P.T @ D
    return DP

# Function to evaluate models with and without PCA
def evaluate_with_pca(DTR, LTR, DVAL, LVAL, model, pi_eff, Cfn, Cfp, m_values):
    results = {}
    for m in m_values:
        if m > 0 and m < DTR.shape[0]:
            DTR_pca = apply_pca(DTR, m)
            DVAL_pca = apply_pca(DVAL, m)
        else:
            DTR_pca, DVAL_pca = DTR, DVAL

        hParams = model(DTR_pca, LTR)
        S_logLikelihood = compute_log_likelihood_Gau(DVAL_pca, hParams)
        S_logPost = compute_logPosterior(S_logLikelihood, np.ones(2) / 2.)
        llr = S_logPost[1, :] - S_logPost[0, :]  # LLR for binary decision
        actual_dcf, min_dcf = evaluate_model(llr, LVAL, pi_eff, Cfn, Cfp)
        results[m] = (actual_dcf, min_dcf)

    return results

def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    hMeans = {}
    CGlobal = 0
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C_class = compute_mu_C(DX)
        CGlobal += C_class * DX.shape[1]
        hMeans[lab] = mu
    CGlobal = CGlobal / D.shape[1]
    for lab in labelSet:
        hParams[lab] = (hMeans[lab], CGlobal)
    return hParams

def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * np.eye(D.shape[0]))  # Diagonal covariance matrix
    return hParams



# Main function
def main():
    # Load your data (log-likelihood ratios and labels)
    llr = np.load('commedia_llr_infpar.npy')  # Replace with your actual data
    labels = np.load('commedia_labels_infpar.npy')  # Replace with your actual data
    
    file_path = 'Traindata.txt'  # Replace with your actual file path
    llr, labels = load_project_data(file_path)
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2tot(llr, labels)
    # Define the applications with specific (pi1, Cfn, Cfp)
    applications = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0)
    ]

    # Evaluate for different effective priors
    effective_priors = [0.1, 0.5, 0.9]
    
    # Define the PCA dimensions to test
    m_values = [0,1, 2, 3,4, 5,6]
    
    print("Effective Prior Analysis with Cfn=Cfp=1:")
    for pi_eff in effective_priors:
        print(f"Application with Effective Prior (pi_eff={pi_eff}):")
        
        for model in [Gau_MVG_ML_estimates, Gau_Tied_ML_estimates, Gau_Naive_ML_estimates]:
            results = evaluate_with_pca(DTR, LTR, DVAL, LVAL, model, pi_eff, 1.0, 1.0, m_values)
            print(f"Results for model {model.__name__}:")
            for m, (actual_dcf, min_dcf) in results.items():
                print(f"  PCA components: {m}, Actual DCF: {actual_dcf:}, Min DCF: {min_dcf:}")
            print("-" * 50)
    
    # Generate Bayes Error Plots
    effPriorLogOdds = np.linspace(-4, 4, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    for model in [Gau_MVG_ML_estimates, Gau_Tied_ML_estimates, Gau_Naive_ML_estimates]:
        actDCF = []
        minDCF = []
        for effPrior in effPriors:
            results = evaluate_with_pca(DTR, LTR, DVAL, LVAL, model, effPrior, 1.0, 1.0, [0])
            actual_dcf, min_dcf = results[0]  # No PCA in this evaluation
            actDCF.append(actual_dcf)
            minDCF.append(min_dcf)
        plt.plot(effPriorLogOdds, actDCF, label=f'{model.__name__} - actDCF')
        plt.plot(effPriorLogOdds, minDCF, label=f'{model.__name__} - minDCF')

    plt.xlabel('Prior Log Odds')
    plt.ylabel('DCF')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.show()

if __name__ == '__main__':
    main()
