import numpy as np
import matplotlib.pyplot as plt
import bayesRisk
import logReg
import scipy.special
from scipy.special import logsumexp
from gmm import train_GMM_LBG_EM, logpdf_GMM

# Helper functions
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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

def load_project_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    D = data[:, :-1].T
    L = data[:, -1].astype(int)
    return D, L


def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):
        # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
        return np.exp(-gamma * Z)

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
        grad = Ha.ravel() - np.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples

def kfold_calibration(scores, labels, KFOLD=5, pT=0.1):
    calibrated_scores = []
    labels_kfold = []
    for foldIdx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(scores, foldIdx, KFOLD)
        LCAL, LVAL = extract_train_val_folds_from_ary(labels, foldIdx, KFOLD)
        w, b = logReg.trainWeightedLogRegBinary(vrow(SCAL), LCAL, 0, pT)
        calibrated_SVAL = (w.T @ vrow(SVAL) + b - np.log(pT / (1-pT))).ravel()
        calibrated_scores.append(calibrated_SVAL)
        labels_kfold.append(LVAL)
    return np.hstack(calibrated_scores), np.hstack(labels_kfold)

def extract_train_val_folds_from_ary(X, idx, KFOLD):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]

def plot_bayes_error(scores, labels, title, pT=0.1):
    log_odds = np.linspace(-4, 4, 50)
    priors = 1 / (1 + np.exp(-log_odds))
    act_dcf_values = [bayesRisk.compute_actDCF_binary_fast(scores, labels, p, 1.0, 1.0) for p in priors]
    min_dcf_values = [bayesRisk.compute_minDCF_binary_fast(scores, labels, p, 1.0, 1.0) for p in priors]

    plt.plot(log_odds, min_dcf_values, '--', label='minDCF')
    plt.plot(log_odds, act_dcf_values, '-', label='actDCF')
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
def bayesPlot(S, L, left=-3, right=3, npts=21):
    effPriorLogOdds = np.linspace(left, right, npts)
    effPriors = 1.0 / (1.0 + np.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        actDCF.append(bayesRisk.compute_actDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
        minDCF.append(bayesRisk.compute_minDCF_binary_fast(S, L, effPrior, 1.0, 1.0))
    return effPriorLogOdds, actDCF, minDCF

def quadratic_expansion(D):
    """Expand the feature matrix D quadratically."""
    n_features = D.shape[0]
    expanded_features = [D]
    for i in range(n_features):
        for j in range(i, n_features):
            expanded_features.append(D[i] * D[j])
    return np.vstack(expanded_features)

# Model evaluation and calibration process
def evaluate_and_calibrate_models(DTR, LTR, DVAL, LVAL, num_components=8, covType='diagonal', C=100, gamma=0.1, KFOLD=5, pT=0.1):
    # Train and evaluate GMM
    gmm_class_0 = train_GMM_LBG_EM(DTR[:, LTR == 0], num_components, covType=covType, psiEig=0.01)
    gmm_class_1 = train_GMM_LBG_EM(DTR[:, LTR == 1], num_components, covType=covType, psiEig=0.01)
    scores_gmm = logpdf_GMM(DVAL, gmm_class_1) - logpdf_GMM(DVAL, gmm_class_0)

    # Train and evaluate Logistic Regression
    DTR_quad = quadratic_expansion(DTR)
    DVAL_quad = quadratic_expansion(DVAL)
    w_lr, b_lr = logReg.trainLogRegBinary(DTR_quad, LTR, 1e-4)
    scores_lr = np.dot(w_lr.T, DVAL_quad) + b_lr

    # Train and evaluate SVM
    kernel_func = rbfKernel(gamma)
    score_func = train_dual_SVM_kernel(DTR, LTR, C, kernel_func)
    scores_svm = score_func(DVAL)

    # Calibrate scores using K-fold
    calibrated_gmm, kfold_label = kfold_calibration(scores_gmm, LVAL, KFOLD, pT)
    calibrated_lr, lr_label = kfold_calibration(scores_lr, LVAL, KFOLD, pT)
    calibrated_svm, svm_label = kfold_calibration(scores_svm, LVAL, KFOLD, pT)

    # Evaluate and print DCFs
    print("Evaluating Calibrated Models...\n")
    minDCF_gmm = bayesRisk.compute_minDCF_binary_fast(calibrated_gmm, kfold_label, pT, 1.0, 1.0)
    actDCF_gmm = bayesRisk.compute_actDCF_binary_fast(calibrated_gmm, kfold_label, pT, 1.0, 1.0)
    print(f"GMM - minDCF: {minDCF_gmm:.4f}, actDCF: {actDCF_gmm:.4f}")

    minDCF_lr = bayesRisk.compute_minDCF_binary_fast(calibrated_lr, lr_label, pT, 1.0, 1.0)
    actDCF_lr = bayesRisk.compute_actDCF_binary_fast(calibrated_lr, lr_label, pT, 1.0, 1.0)
    print(f"Logistic Regression - minDCF: {minDCF_lr:.4f}, actDCF: {actDCF_lr:.4f}")

    minDCF_svm = bayesRisk.compute_minDCF_binary_fast(calibrated_svm, svm_label, pT, 1.0, 1.0)
    actDCF_svm = bayesRisk.compute_actDCF_binary_fast(calibrated_svm, svm_label, pT, 1.0, 1.0)
    print(f"SVM - minDCF: {minDCF_svm:.4f}, actDCF: {actDCF_svm:.4f}")

    # Plot Bayes error plots for each model
    # logOdds, actDCF, minDCF = bayesPlot(calibrated_gmm, kfold_label)
    plot_bayes_error(calibrated_gmm, kfold_label, "GMM Model - Bayes Error Plot")
    plot_bayes_error(calibrated_lr, lr_label, "Logistic Regression - Bayes Error Plot")
    plot_bayes_error(calibrated_svm, svm_label, "SVM Model - Bayes Error Plot")

    return calibrated_gmm, calibrated_lr, calibrated_svm

def fuse_and_evaluate_models(calibrated_gmm, calibrated_lr, calibrated_svm, LVAL, pT=0.1):
    # Correct the shape of fused_scores
    fused_scores = np.vstack([calibrated_gmm, calibrated_lr, calibrated_svm]).T  # Shape (n_samples, 3)

    # Train fusion model using logistic regression
    w_fusion, b_fusion = logReg.trainLogRegBinary(fused_scores.T, LVAL, 1e-4)

    # Compute fused scores
    fused_scores_val = np.dot(w_fusion.T, fused_scores.T) + b_fusion

    # Compute and display DCF for fused scores
    minDCF_fusion = bayesRisk.compute_minDCF_binary_fast(fused_scores_val, LVAL, pT, 1.0, 1.0)
    actDCF_fusion = bayesRisk.compute_actDCF_binary_fast(fused_scores_val, LVAL, pT, 1.0, 1.0)
    print(f"Fusion Model - minDCF: {minDCF_fusion:.4f}, actDCF: {actDCF_fusion:.4f}")

    # Plot the Bayes error plot for fusion
    plot_bayes_error(fused_scores_val, LVAL, "Fusion Model - Bayes Error Plot")


###################################################################################################################################################3
def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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

def evaluate_DCF(score, labels, prior):
    predictions = (score > 0).astype(int)
    accuracy = (predictions == labels).mean()
    
    pi_emp = (labels == 1).mean()
    log_odds = np.log(pi_emp / (1 - pi_emp))
    S_llr = score - log_odds
    
    cfn = 1
    cpn = 1

    decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)

    cm = compute_confusion_matrix(decisions, labels, labels.max()+1)
    # print(cm)

    dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
    # print(f"Actual DCF : {dcf_norm}")
    
    #Minimum Detection Cost
    min_cost = compute_min_DCF(S_llr, labels, prior, cfn, cpn, dcf_norm)
    # print(f"min DCF: {min_cost}")
    
    return dcf_norm, min_cost

def logpdf_GAU_ND(X, mu, C):
    """
    Compute log-probability density function for multivariate Gaussian with full covariance
    """
    N = X.shape[0]
    sign, log_det = np.linalg.slogdet(C)
    const = -0.5 * (N * np.log(2 * np.pi) + log_det)
    
    C_inv = np.linalg.inv(C)
    
    Y = X - mu.reshape(-1, 1)
    log_pdf = const - 0.5 * np.sum(Y * np.dot(C_inv, Y), axis=0)
    
    return log_pdf

def logpdf_GAU_ND_diag(X, mu, C_diag):
    """
    Compute log-probability density function for multivariate Gaussian with diagonal covariance
    """
    N = X.shape[0]
    const = -0.5 * (N * np.log(2 * np.pi) + np.sum(np.log(C_diag)))
    
    Y = X - mu.reshape(-1, 1)
    log_pdf = const - 0.5 * np.sum((Y**2) / C_diag.reshape(-1, 1), axis=0)
    
    return log_pdf

def logpdf_GMM(X, gmm):
    """
    Compute log-probability density function for GMM
    """
    S = np.zeros((len(gmm['weights']), X.shape[1]))
    
    for g in range(len(gmm['weights'])):
        if gmm['cov_type'] == 'full':
            S[g] = logpdf_GAU_ND(X, gmm['means'][:, g], gmm['covs'][:, :, g])
        else:  # diagonal covariance
            S[g] = logpdf_GAU_ND_diag(X, gmm['means'][:, g], gmm['covs'][:, g])
    
    S = S + np.log(gmm['weights']).reshape(-1, 1)
    return logsumexp(S, axis=0)


def train_gmms_per_class(D, num_components, cov_type='full', psi=0.01, num_iterations=10):
    """
    Train a GMM for a specific class
    """
    # Initialize GMM parameters
    N, M = D.shape  # N features, M samples
    
    # Initialize weights, means, and covariances
    w = np.ones(num_components) / num_components
    mu = np.zeros((N, num_components))
    
    if cov_type == 'full':
        C = np.zeros((N, N, num_components))
    else:  # diagonal covariance
        C = np.zeros((N, num_components))
    
    # Use k-means or random initialization for means
    # For simplicity, let's use random samples from the data
    if M >= num_components:
        idx = np.random.choice(M, num_components, replace=False)
        mu = D[:, idx]
    else:
        # If we have fewer samples than components, duplicate some
        idx = np.random.choice(M, num_components, replace=True)
        mu = D[:, idx]
    
    # Initialize covariances as identity matrices or data covariance
    if cov_type == 'full':
        data_cov = np.cov(D)
        for g in range(num_components):
            C[:, :, g] = data_cov
    else:  # diagonal covariance
        data_var = np.var(D, axis=1)
        for g in range(num_components):
            C[:, g] = data_var
    
    # EM algorithm
    for iteration in range(num_iterations):
        # E-step: compute responsibilities
        log_S = np.zeros((num_components, M))
        
        for g in range(num_components):
            if cov_type == 'full':
                log_S[g] = logpdf_GAU_ND(D, mu[:, g], C[:, :, g])
            else:  # diagonal covariance
                log_S[g] = logpdf_GAU_ND_diag(D, mu[:, g], C[:, g])
        
        log_S = log_S + np.log(w.reshape(-1, 1))
        log_marginal = logsumexp(log_S, axis=0)
        log_gamma = log_S - log_marginal
        gamma = np.exp(log_gamma)
        
        # M-step: update parameters
        Z = np.sum(gamma, axis=1)
        
        # Update weights
        w = Z / M
        
        # Update means
        for g in range(num_components):
            mu[:, g] = np.dot(D, gamma[g]) / Z[g]
        
        # Update covariances
        for g in range(num_components):
            if cov_type == 'full':
                centered = D - mu[:, g].reshape(-1, 1)
                C[:, :, g] = np.dot(centered * gamma[g], centered.T) / Z[g]
                # Apply eigenvalue flooring
                U, s, _ = np.linalg.svd(C[:, :, g])
                s[s < psi] = psi
                C[:, :, g] = np.dot(U, np.dot(np.diag(s), U.T))
            else:  # diagonal covariance
                centered = D - mu[:, g].reshape(-1, 1)
                C[:, g] = np.sum((centered ** 2) * gamma[g], axis=1) / Z[g]
                # Apply variance flooring
                C[:, g] = np.maximum(C[:, g], psi)
    
    # Return GMM parameters
    gmm = {
        'weights': w,
        'means': mu,
        'covs': C,
        'cov_type': cov_type
    }
    
    return gmm


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

def bayes_error_plot(S_llr, LTE, title="Bayes Error Plot"):
    log_odds_range = np.linspace(-4, 4, 25)
    effective_priors = 1 / (1 + np.exp(-log_odds_range))

    actual_dcf_list = []
    min_dcf_list = []

    cfn = 1
    cpn = 1

    for pi in effective_priors:
        decisions = optimal_bayes_decision(S_llr, pi, cfn, cpn)
        cm = compute_confusion_matrix(decisions, LTE, 2)
        dcf = empirical_bayes_risks(cm, pi, cfn, cpn, normalize=True)
        min_dcf = compute_min_DCF(S_llr, LTE, pi, cfn, cpn, dcf)
        actual_dcf_list.append(dcf)
        min_dcf_list.append(min_dcf)

    plt.figure(figsize=(8, 5))
    plt.plot(log_odds_range, actual_dcf_list, label='Actual DCF', marker='o')
    plt.plot(log_odds_range, min_dcf_list, label='Min DCF', marker='x')
    plt.xlabel("logit(πT)")
    plt.ylabel("DCF")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def k_fold_callibration(llr, LTE, K, title='Bayes Plot'):
    K=K
    prior = 0.2
    SCAL1 =[]
    # SCAL2 =[]
    LCAL = []
    for idx in range(K):
      S1 = llr[idx::K]
      L = LTE[idx::K]
      SCAL1.append(S1)
      LCAL.append(L)
     
    SCAL1 = np.hstack(SCAL1)
    LCAL = np.hstack(LCAL)
    
    w, b, loss = train_prior_weighted_logreg(vrow(SCAL1), LCAL, 0, prior)
    calibrated_CVAL1 = (w.T @ vrow(SCAL1) + b - np.log(prior / (1-prior))).ravel()
    
    
    dcf_norm_1, min_cost_1 = evaluate_DCF(calibrated_CVAL1, LCAL, prior)
    print(f'Score 1: min_cost = {min_cost_1} and act_dcf = {dcf_norm_1}')
    
    bayes_error_plot(calibrated_CVAL1, LCAL, title=title)
    
    return calibrated_CVAL1

def quadratic_feature_expansion(D):
    n_features, n_samples = D.shape
    expanded_features = []
    for i in range(n_samples):
        x = D[:, i]
        phi = [x]
        for j in range(n_features):
            for k in range(j, n_features):
                phi.append(x[j] * x[k])
        expanded_features.append(np.hstack(phi))
    return np.array(expanded_features).T

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

def compute_accuracy(D, L, w, b):
    S = 1 / (1 + np.exp(-(np.dot(w.T, D) + b)))
    P = (S > 0.5).astype(int)
    acc = (P == L).mean()
    return acc

def evaluate_bnr(DTR, LTR, DTE, LTE, lambdas, labels):
    
    prior = 0.1
    cfn = 1
    cpn = 1
    actual_dcf_list = []
    min_dcf_list = []
    
    for lamda in lambdas:
        # print('Lamda: ', lamda)
        w, b = train_logreg(DTR, LTR, lamda)
        acc = compute_accuracy(DTE, LTE, w, b)
        # print("Validation Accuracy:", acc)
        S = (w.T @ DTE + b).ravel()
        #error rate
        LP = (S > 0).astype(int)
        accuracy = (LP == LTE).mean()
        error_rate = 1 - accuracy
        
        pi_emp = (LTR == 1).mean()
        log_odds = np.log(pi_emp / (1 - pi_emp))
        S_llr = S - log_odds
        
        

        decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)
        cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
        dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
        min_cost = compute_min_DCF(S_llr, LTE, prior, cfn, cpn, dcf_norm)
        actual_dcf_list.append(dcf_norm)
        min_dcf_list.append(min_cost)
        
        # Find and print minimum actualDCF and its corresponding lambda
    min_act_dcf = min(actual_dcf_list)
    best_lambda_act = lambdas[actual_dcf_list.index(min_act_dcf)]
    print(f"Minimum Actual DCF: {min_act_dcf:.4f} at λ = {best_lambda_act}")

    # Find and print minimum minDCF and its corresponding lambda
    min_min_dcf = min(min_dcf_list)
    best_lambda_min = lambdas[min_dcf_list.index(min_min_dcf)]
    print(f"Minimum minDCF: {min_min_dcf:.4f} at λ = {best_lambda_min}")

    
    return S_llr

def compute_kernel_svm_dual(DTR, LTR, C, kernel_func, kernel_params):
    Z = 2 * LTR - 1
    Z = Z.astype(np.int32)
    N = DTR.shape[1]

    # Compute kernel matrix K
    K = kernel_func(DTR, DTR, **kernel_params)

    # Compute H = zi zj K(xi, xj)
    H = Z[:, None] * Z[None, :] * K


    def dual_obj(alpha):
        Ha = H @ alpha
        obj = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - np.ones(alpha.size)
        return obj, grad

    bounds = [(0, C) for i in (LTR)]
    x0 = np.zeros(N)

    alpha_opt, f_min, _ = scipy.optimize.fmin_l_bfgs_b(
        func=dual_obj,
        x0=x0,
        bounds=bounds,
        factr=np.nan,
        pgtol=1e-5,
        maxfun=100000,
        maxiter=1000
    )
    

    return alpha_opt, Z, DTR, H

def compute_objectives(alpha, H, Z, C):
    dual_loss = np.sum(alpha) - 0.5 * alpha @ (H @ alpha)

    s = H @ alpha  # scores
    hinge_losses = np.maximum(0, 1 - s)
    primal_loss = 0.5 * alpha @ (H @ alpha) + C * hinge_losses.sum()

    duality_gap = primal_loss - dual_loss
    return primal_loss, dual_loss, duality_gap

def kernel_with_bias(kernel_func, kernel_params, bias_K):
    def biased_kernel(X1, X2):
        base = kernel_func(X1, X2, **kernel_params)
        return base + bias_K**2  # K squared as ξ
    return biased_kernel

def rbf_kernel(X1, X2, gamma):
    X1_sq = np.sum(X1**2, axis=0).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=0).reshape(1, -1)
    dist_sq = X1_sq + X2_sq - 2 * (X1.T @ X2)
    return np.exp(-gamma * dist_sq)


def compute_kernel_svm_scores(DTE, alpha, Z, DTR, kernel_func, kernel_params):
    # Only support vectors
    support = alpha > 1e-6
    alpha_sv = alpha[support]
    Z_sv = Z[support]
    DTR_sv = DTR[:, support]

    K = kernel_func(DTR_sv, DTE, **kernel_params)
    scores = np.sum((alpha_sv * Z_sv)[:, None] * K, axis=0)
    return scores

def k_fold_callibration_fusion(scores, labels, K=5, title="Bayes Error Plot"):
    SCAL = []
    LCAL = []
    for idx in range(K):
        if scores.ndim == 1:
            S_train = np.hstack([scores[i::K] for i in range(K) if i != idx])
            L_train = np.hstack([labels[i::K] for i in range(K) if i != idx])
            S_val = scores[idx::K]
            L_val = labels[idx::K]
        else:
            S_train = np.vstack([scores[i::K] for i in range(K) if i != idx])
            L_train = np.hstack([labels[i::K] for i in range(K) if i != idx])
            S_val = scores[idx::K]
            L_val = labels[idx::K]

        # Train a logistic regressor for calibration
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        lr.fit(S_train, L_train)

        calibrated = lr.predict_log_proba(S_val)[:, 1]  # log odds
        SCAL.append(calibrated)
        LCAL.append(L_val)

    SCAL = np.hstack(SCAL)
    LCAL = np.hstack(LCAL)

    # Optionally plot Bayes error
    bayes_error_plot(SCAL, LCAL, title=title)

    return SCAL
# Main execution
if __name__ == '__main__':
    # Load training data
    D, L = load_project_data('trainData.txt')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    prior = 0.1
    

    
    #GMM
    cov_type = 'diag'
    num_components_0 = 16
    num_components_1 = 32
    psi = 0.01
    
    # Train GMMs
    gmm_class_0 = train_gmms_per_class(DTR[:, LTR==0], num_components_0, cov_type=cov_type, psi=psi)
    gmm_class_1 = train_gmms_per_class(DTR[:, LTR==1], num_components_1, cov_type=cov_type, psi=psi)
    
    # Compute LLRs on test set
    ll_class_0 = logpdf_GMM(DTE, gmm_class_0)
    ll_class_1 = logpdf_GMM(DTE, gmm_class_1)
    llr = ll_class_1 - ll_class_0
    
    #k-fold approch
    Calibrated_GMM = k_fold_callibration(llr, LTE, K=5, title=f"Bayes Error Plot (GMM {cov_type.upper()}, ({num_components_0},{num_components_1}))")
    
    
    
    #Logistic regression
    DTR_quad = quadratic_feature_expansion(DTR)
    DTE_quad = quadratic_feature_expansion(DTE)
    labels = "quadratic logistic regression"
    lambdas = np.logspace(-4, 2, 13)
    S_llr_quad = evaluate_bnr(DTR_quad, LTR, DTE_quad, LTE, lambdas, labels)
    
    Calibrated_LR= k_fold_callibration(S_llr_quad, LTE, K=5, title=f"Bayes Error Plot (Logisteic regression)")

    
    
    kernel = kernel_with_bias(rbf_kernel, {'gamma': 0.1}, bias_K=1)
    
    alpha_opt, Z, DTR_kernel, H = compute_kernel_svm_dual(DTR, LTR, 31.622776601683793, kernel_func=kernel, kernel_params={})
    scores = compute_kernel_svm_scores(DTE, alpha_opt, Z, DTR_kernel, kernel_func=kernel, kernel_params={})
    
    # Convert scores to log-likelihood ratios
    pi_emp = (LTR == 1).mean()
    log_odds = np.log(pi_emp / (1 - pi_emp))
    S_llr = scores - log_odds
    
    Calibrated_SVM = k_fold_callibration(S_llr, LTE, K=5, title=f"Bayes Error Plot (RBF SVM, C={31.622776601683793:.2e}, gamma={0.1})")
    
    #score fusion
    print("GMM calibrated scores:", Calibrated_GMM.shape)
    print("LR calibrated scores:", Calibrated_LR.shape)
    print("SVM calibrated scores:", Calibrated_SVM.shape)

    fused_scores = np.vstack([Calibrated_GMM, Calibrated_LR, Calibrated_SVM]).T
    Calibrated_LR= k_fold_callibration_fusion(fused_scores, LTE, K=5, title=f"Bayes Error Plot (Fusion )")
    
    # # Evaluate and calibrate models
    # calibrated_gmm, calibrated_lr, calibrated_svm = evaluate_and_calibrate_models(DTR, LTR, DVAL, LVAL)

    # # Perform score fusion and final evaluation
    # fuse_and_evaluate_models(calibrated_gmm, calibrated_lr, calibrated_svm, LVAL)

   
    # # Load evaluation dataset
    # print("\nEvaluating models on Project/evalData.txt...")
    # DEVAL, LEVAL = load_project_data('evalData.txt')

    # # Evaluate on the evaluation dataset
    # scores_gmm_eval, scores_lr_eval, scores_svm_eval = evaluate_and_calibrate_models(DEVAL, LEVAL, DVAL, LVAL)

    # # Perform fusion evaluation on the evaluation dataset
    # fuse_and_evaluate_models(scores_gmm_eval, scores_lr_eval, scores_svm_eval, LEVAL)
    
    #Evalution data set is remaining
    
    
    
    
    