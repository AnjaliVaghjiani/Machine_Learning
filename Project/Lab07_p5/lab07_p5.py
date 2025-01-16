import numpy as np

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

def load_project_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    D = data[:, :-1].T
    L = data[:, -1].astype(int)
    return D, L

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C

# def logpdf_GAU_ND(X, mu, C):
#     P = np.linalg.inv(C)
#     const = -0.5 * X.shape[0] * np.log(2 * np.pi)
#     log_det_C = -0.5 * np.linalg.slogdet(C)[1]
#     Y = []
#     for i in range(X.shape[1]):
#         x = X[:, i:i+1]
#         diff = x - mu
#         exponent = -0.5 * (diff.T @ P @ diff)
#         Y.append(const + log_det_C + exponent)
#     return np.array(Y).ravel()

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

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

def compute_effective_prior(pi1, Cfn, Cfp):
    return (pi1 * Cfn) / (pi1 * Cfn + (1 - pi1) * Cfp)

def compute_optimal_bayes_decisions(llr, pi1, Cfn, Cfp):
    threshold = -np.log(pi1 * Cfn / ((1 - pi1) * Cfp))
    predictions = (llr > threshold).astype(int)
    # return predictions
    return np.int32(llr > threshold)
def compute_confusion_matrix(predictions, labels):
    TP = np.sum((predictions == 1) & (labels == 1))
    FP = np.sum((predictions == 1) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))
    TN = np.sum((predictions == 0) & (labels == 0))
    return np.array([[TN, FP], [FN, TP]])

# def compute_confusion_matrix(predictedLabels, classLabels):
#     nClasses = classLabels.max() + 1
#     M = np.zeros((nClasses, nClasses), dtype=np.int32)
#     for i in range(classLabels.size):
#         M[predictedLabels[i], classLabels[i]] += 1
#     return M

def compute_dcf(cm, pi1, Cfn, Cfp, normalize=True):
    Pfn = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    Pfp = cm[0, 1] / (cm[0, 1] + cm[0, 0])
    bayesError = pi1 * Cfn * Pfn + (1-pi1) * Cfp * Pfp
    if normalize:
        return bayesError / np.minimum(pi1 * Cfn, (1-pi1)*Cfp)
    return bayesError
    

def evaluate_model(llr, labels, pi1, Cfn, Cfp):
    predictions = compute_optimal_bayes_decisions(llr, pi1, Cfn, Cfp)
    cm = compute_confusion_matrix(predictions, labels)
    # actual_dcf = compute_dcf(cm, pi1, Cfn, Cfp, normalize=False)
    actual_dcf = compute_dcf(cm, pi1, Cfn, Cfp, normalize=False)
    DCF_norma = compute_dcf(cm, pi1, Cfn, Cfp)
    min_dcf = compute_minDCF_binary_slow(llr, labels, pi1, Cfn, Cfp)
    return cm,actual_dcf, DCF_norma,min_dcf

# def compute_min_dcf(llr, labels, pi1, Cfn, Cfp):
#     thresholds = np.sort(llr)
#     dcf_values = []
#     for threshold in thresholds:
#         predictions = (llr > threshold).astype(int)
#         cm = compute_confusion_matrix(predictions, labels)
#         dcf = compute_dcf(cm, pi1, Cfn, Cfp)
#         dcf_values.append(dcf)
#     return min(dcf_values)

# def apply_pca(D, m):
#     mu = np.mean(D, axis=1, keepdims=True)
#     DC = D - mu
#     U, _, _ = np.linalg.svd(DC, full_matrices=False)
#     P = U[:, :m]
#     DP = P.T @ D
#     return DP
def compute_pca(D, m):
    mu = np.mean(D, axis=1, keepdims=True)
    DC = D - mu
    U, s, Vh = np.linalg.svd(DC, full_matrices=False)
    P = U[:, :m]
    return mu, P

def apply_pca(D, mu, P):
    DC = D - mu
    DP = P.T @ DC
    return DP

def evaluate_with_pca(DTR, LTR, DVAL, LVAL, model, pi_eff, Cfn, Cfp, m_values):
    results = {}
    for m in m_values:
    # if m > 0 and m < DTR.shape[0]:
        mu_pca, P = compute_pca(DTR,m)
        # Apply PCA transformation
        DTR_pca = apply_pca(DTR, mu_pca, P)
        DVAL_pca = apply_pca(DVAL, mu_pca, P)
        # DTR_pca = apply_pca(DTR, m)
        # DVAL_pca = apply_pca(DVAL, m)
    # else:
        # DTR_pca, DVAL_pca = DTR, DVAL
# 
        hParams = model(DTR_pca, LTR)
        S_logLikelihood = compute_log_likelihood_Gau(DVAL_pca, hParams)
        S_logPost = compute_logPosterior(S_logLikelihood, np.ones(2) / 2.)
        llr = S_logPost[1, :] - S_logPost[0, :]  # LLR for binary decision
        cm,actual_dcf,normalized, min_dcf = evaluate_model(llr, LVAL, pi_eff, Cfn, Cfp)
        results[m] = (cm,actual_dcf, normalized,min_dcf)

    return results

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

def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * np.eye(D.shape[0]))
    return hParams


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

def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    cm = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    # Pfn = M[0,1] / (M[0,1] + M[1,1])
    # Pfp = M[1,0] / (M[0,0] + M[1,0])
    Pfn = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    Pfp = cm[0, 1] / (cm[0, 1] + cm[0, 0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / np.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

def compute_minDCF_binary_slow(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    # llrSorter = numpy.argsort(llr) 
    # llrSorted = llr[llrSorter] # We sort the llrs
    # classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs
    # We can remove this part
    llrSorted = llr # In this function (slow version) sorting is not really necessary, since we re-compute the predictions and confusion matrices everytime
    
    thresholds = np.concatenate([np.array([-np.inf]), llrSorted, np.array([np.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predictedLabels = np.int32(llr > th)
        dcf = compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin
def compute_bayes_error_plot(llr, labels, pi_eff, Cfn, Cfp):
    prior_log_odds = np.linspace(-4, 4, 21)
    actual_dcf = []
    min_dcf = []
    
    for log_odds in prior_log_odds:
        eff_prior = 1.0 / (1.0 + np.exp(-log_odds))
        cm,actual_val, normalized, min_dcf_val = evaluate_model(llr, labels, eff_prior, Cfn, Cfp)
        actual_dcf.append(normalized)
        min_dcf.append(min_dcf_val)
    
    return prior_log_odds, np.array(actual_dcf), np.array(min_dcf)

def plot_combined_bayes_error(prior_log_odds, results):
    plt.figure(figsize=(10, 6))
    
    for model_name, (actual_dcf, min_dcf) in results.items():
        plt.plot(prior_log_odds, actual_dcf, label=f'{model_name} Actual DCF')
        plt.plot(prior_log_odds, min_dcf, label=f'{model_name} Minimum DCF', linestyle='dashed')
    
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('Detection Cost Function (DCF)')
    plt.title('Bayes Error Plots for MVG, Tied MVG, and Naive Bayes')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    # Load your data (log-likelihood ratios and labels)
    # llr = np.load('commedia_llr_infpar_eps1.npy')  # Replace with your actual data
    # labels = np.load('commedia_labels_infpar_eps1.npy')  # Replace with your actual data
    file_path = 'Traindata.txt'  # Replace with your actual file path
    llr, labels = load_project_data(file_path)
    # Define the applications with specific (pi1, Cfn, Cfp)
    D=llr
    L=labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2tot(llr, labels)
    applications = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0)
    ]

    model_functions = {
        'MVG': Gau_MVG_ML_estimates
    }

    # Iterate over each model and application
    for model_name, model_func in model_functions.items():
        print(f"Model: {model_name}")
        hParams = model_func(DTR, LTR)
        S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams)
        S_logPost = compute_logPosterior(S_logLikelihood, np.ones(2) / 2.)
        llr = S_logPost[1, :] - S_logPost[0, :]  # LLR for binary decision

        for pi1, Cfn, Cfp in applications:

            pi_eff = pi1
            # pi_eff = compute_effective_prior(pi1, Cfn, Cfp)
            cm, actual_dcf,DCF_norma, min_dcf = evaluate_model(llr, LVAL, pi_eff, Cfn, Cfp)
            print(f"  Application (pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}):")
            print(f"    Confusion Matrix:\n{cm}")
            print(f"    Actual DCF (non-normalized): {actual_dcf:f}")
            print(f"    Minimum DCF: {min_dcf:f}")
            print(f"    DCF_norma: {DCF_norma:3f}")
            print(f"    Calibration loss: {actual_dcf - min_dcf:.3f}")
        print("-" * 50)

    effective_priors = [0.1, 0.5, 0.9]
    
    # Define the PCA dimensions to test
    m_values = [1, 2,3,4,5,6]
    
    models = {
        'MVG': Gau_MVG_ML_estimates,
        'Naive Bayes': Gau_Naive_ML_estimates,
        'Tied MVG': Gau_Tied_ML_estimates
    }
    (DTR, LTR), (DVAL, LVAL) = split_db_2tot(D, L)
    # for model_name, model in models.items():
    #     print(f"Model: {model_name}")
    #     for pi1, Cfn, Cfp in applications:
    #         print(f"  Application (pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}):")
    #         results = evaluate_with_pca(DTR, LTR, DVAL, LVAL, model, pi_eff, Cfn, Cfp, m_values)
    #         for m, (cm,actual_dcf, normalized, min_dcf) in results.items():
    #             print(f"    PCA Components: {m}")
    #             print(f"    Confusion Matrix:\n{cm}")
    #             print(f"    Actual DCF: {actual_dcf:.6f}")
    #             print(f"    Minimum DCF: {min_dcf:.6f}")
    #             print(f"    DCF_norma: {normalized:f}")
    #             print(f"    Calibration loss: {actual_dcf - min_dcf:.6f}")
    #             print("-" * 50)
    # for model_name, model in models.items():
    #     for m in m_values:
    #         print(f"Model: {model_name} with PCA Components: {m}")
            
    #         for pi_eff, Cfn, Cfp in applications:
    #             results = evaluate_with_pca(DTR, LTR, DVAL, LVAL, model, pi_eff, Cfn, Cfp, [m])
    #             # pi_eff = compute_effective_prior(pi1, Cfn, Cfp)
    #             print(f"  Application (pi1={pi_eff}, Cfn={Cfn}, Cfp={Cfp}):")
    #             cm, actual_dcf, normalized, min_dcf = results[m]
    #             print(f"    Confusion Matrix:\n{cm}")
    #             print(f"    Actual DCF: {actual_dcf:.10f}")
    #             print(f"    Minimum DCF: {min_dcf:.10f}")
    #             print(f"    DCF_norma: {normalized:.10f}")
    #             print(f"    Calibration loss: {actual_dcf - min_dcf:.10f}")
    #             print("-" * 50)
    best_m = 5
    mu_pca, P = compute_pca(DTR,best_m)
    DTR_pca = apply_pca(DTR, mu_pca, P)
    DVAL_pca = apply_pca(DVAL,mu_pca, P)
    # Applications to evaluate
    applications = [
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0)
    ]
    
   # Define the main application with π˜=0.1
    pi_eff = 0.1
    Cfn, Cfp = 1.0, 1.0
    
    results = {}
    
    for model_name, model in models.items():
        print(f"Model: {model_name}")
        
        # Train the model
        hParams = model(DTR_pca, LTR)
        S_logLikelihood = compute_log_likelihood_Gau(DVAL_pca, hParams)
        S_logPost = compute_logPosterior(S_logLikelihood, np.ones(2) / 2.)
        llr = S_logPost[1, :] - S_logPost[0, :]  # LLR for binary decision
        
        prior_log_odds, actual_dcf, min_dcf = compute_bayes_error_plot(llr, LVAL, pi_eff, Cfn, Cfp)
        results[model_name] = (actual_dcf, min_dcf)
        
        
    
    # Plot all results in a single combined plot
    plot_combined_bayes_error(prior_log_odds, results)
   
    
   

if __name__ == '__main__':
    main()
