import numpy as np
from gmm import train_GMM_LBG_EM, logpdf_GMM
from scipy.special import logsumexp
import bayesRisk
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special

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

def evaluate_gmm_model(DTR, LTR, DVAL, LVAL, num_components_list, cov_type, prior):
    """Train GMMs and evaluate minDCF and actDCF."""
    results = []

    for numC0 in num_components_list:
        for numC1 in num_components_list:
            # Train GMMs for each class
            gmm_class_0 = train_GMM_LBG_EM(DTR[:, LTR == 0], numC0, covType=cov_type, psiEig=0.01)
            gmm_class_1 = train_GMM_LBG_EM(DTR[:, LTR == 1], numC1, covType=cov_type, psiEig=0.01)

            # Log-likelihood ratio for validation set
            SLLR = logpdf_GMM(DVAL, gmm_class_1) - logpdf_GMM(DVAL, gmm_class_0)

            # Compute minDCF and actDCF
            min_dcf = bayesRisk.compute_minDCF_binary_fast(SLLR, LVAL, prior, 1.0, 1.0)
            act_dcf = bayesRisk.compute_actDCF_binary_fast(SLLR, LVAL, prior, 1.0, 1.0)

            # Format results
            results.append({
                "Components (Class 0)": numC0,
                "Components (Class 1)": numC1,
                "minDCF": min_dcf,
                "actDCF": act_dcf
            })


    return results


def plot_results(results, title):
    """Plot minDCF and actDCF for GMM results."""
    for components in results['Components (Class 0)'].unique():
        subset = results[results['Components (Class 0)'] == components]
        plt.plot(subset['Components (Class 1)'], subset['minDCF'], label=f"minDCF (Class 0 = {components})", marker='o')
        plt.plot(subset['Components (Class 1)'], subset['actDCF'], label=f"actDCF (Class 0 = {components})", linestyle='--', marker='x')

    plt.title(title)
    plt.xlabel("Components (Class 1)")
    plt.ylabel("DCF")
    plt.legend()
    plt.grid()
    plt.show()
    

# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = np.dot(vcol(w).T, DTR).ravel() + b

        loss = np.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + np.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * np.linalg.norm(w)**2, np.hstack([GW, np.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = np.zeros(DTR.shape[0]+1))[0]
    # print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

def evaluate_logistic_regression(DTR, LTR, DVAL, LVAL, log_odds_range):
    lambdas = np.logspace(-4, 2, 13)
    best_min_dcf, best_act_dcf = float('inf'), float('inf')

    min_dcf_values = []
    act_dcf_values = []

    for lamb in lambdas:
        w, b = trainLogRegBinary(DTR, LTR, lamb)
        sVal = np.dot(w.T, DVAL) + b

        for log_odds in log_odds_range:
            prior = 1 / (1 + np.exp(-log_odds))
            sValLLR = sVal - np.log(prior / (1 - prior))
            min_dcf = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, prior, 1.0, 1.0)
            act_dcf = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, prior, 1.0, 1.0)
            min_dcf_values.append(min_dcf)
            act_dcf_values.append(act_dcf)

            best_min_dcf = min(best_min_dcf, min_dcf)
            best_act_dcf = min(best_act_dcf, act_dcf)

    return min_dcf_values, act_dcf_values, best_min_dcf, best_act_dcf


    
def plot_bayes_error(log_odds_range, models_results):
    """Plot the Bayes error plot for different models."""
    for model_name, (min_dcf_values, act_dcf_values) in models_results.items():
        if len(log_odds_range) != len(min_dcf_values) or len(log_odds_range) != len(act_dcf_values):
            print(f"Error: log_odds_range and DCF values for {model_name} have mismatched dimensions.")
            print(f"log_odds_range: {len(log_odds_range)}, min_dcf_values: {len(min_dcf_values)}, act_dcf_values: {len(act_dcf_values)}")
            continue  # Skip plotting this model

        plt.plot(log_odds_range, min_dcf_values, label=f"{model_name} minDCF", linestyle="--")
        plt.plot(log_odds_range, act_dcf_values, label=f"{model_name} actDCF", linestyle="-")

    plt.xlabel("Prior log-odds")
    plt.ylabel("DCF value")
    plt.title("Bayes Error Plot")
    plt.legend()
    plt.grid()
    plt.show()


def train_logistic_regression(DTR, LTR, lambdas, DVAL, LVAL, pi_t):
    best_lambda = None
    best_min_dcf = float("inf")
    results = []

    for lamb in lambdas:
        # Train logistic regression
        w, b = trainLogRegBinary(DTR, LTR, lamb)
        
        # Compute validation scores
        sVal = np.dot(w.T, DVAL) + b
        
        # Adjust scores based on prior
        sValLLR = sVal - np.log(pi_t / (1 - pi_t))
        
        # Compute minDCF and actDCF
        min_dcf = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
        act_dcf = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
        
        # Save results
        results.append({'lambda': lamb, 'minDCF': min_dcf, 'actDCF': act_dcf})
        
        # Update best lambda
        if min_dcf < best_min_dcf:
            best_min_dcf = min_dcf
            best_lambda = lamb

    return best_lambda, best_min_dcf, results

def quadratic_expansion(D):
    """Expand the feature matrix D quadratically."""
    n_features = D.shape[0]
    expanded_features = [D]
    for i in range(n_features):
        for j in range(i, n_features):
            expanded_features.append(D[i] * D[j])
    return np.vstack(expanded_features)

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

    # print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

    # Function to compute the scores for samples in DTE
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore # we directly return the function to score a matrix of test samples


# Evaluate SVM with Kernel
def evaluate_svm_kernel(DTR, LTR, DVAL, LVAL, C_values, kernelFunc):
    min_dcf_values = []
    act_dcf_values = []

    for C in C_values:
        fScore = train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, 1.0)
        scores = fScore(DVAL)

        min_dcf = bayesRisk.compute_minDCF_binary_fast(scores, LVAL, 0.5, 1.0, 1.0)
        act_dcf = bayesRisk.compute_actDCF_binary_fast(scores, LVAL, 0.5, 1.0, 1.0)

        min_dcf_values.append(min_dcf)
        act_dcf_values.append(act_dcf)

    return min_dcf_values, act_dcf_values

def qualitative_analysis(DTR, LTR, DVAL, LVAL, log_odds_range, models_results):
    """
    Perform qualitative analysis of the three approaches across different applications.
    Visualize actual and minimum DCF for each model using Bayes error plots.
    """
    # Expand Bayes Error Plot to include all models and a wider range of priors
    plt.figure(figsize=(10, 6))
    
    for model_name, (min_dcf_values, act_dcf_values) in models_results.items():
        plt.plot(log_odds_range, min_dcf_values, label=f"{model_name} minDCF", linestyle="--")
        plt.plot(log_odds_range, act_dcf_values, label=f"{model_name} actDCF", linestyle="-")
    
    plt.xlabel("Prior log-odds")
    plt.ylabel("DCF value")
    plt.title("Bayes Error Plot for Different Models and Applications")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Analyze consistency and calibration
    print("\nQualitative Analysis Summary:")
    for model_name, (min_dcf_values, act_dcf_values) in models_results.items():
        print(f"\nModel: {model_name}")
        print(f"  Minimum DCF Range: {min(min_dcf_values):.4f} to {max(min_dcf_values):.4f}")
        print(f"  Actual DCF Range: {min(act_dcf_values):.4f} to {max(act_dcf_values):.4f}")
        if max(min_dcf_values) - min(min_dcf_values) > 0.1:
            print("  Observation: Model is inconsistent across applications in terms of minimum DCF.")
        if max(act_dcf_values) - min(act_dcf_values) > 0.2:
            print("  Observation: Model shows significant miscalibration for some applications.")
            
            
##############################################################################################################
def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)


def logpdf_GMM(X, gmm):
    S = []
    for w, mu, C in gmm:
        mu = mu.reshape(-1, 1)
        log_comp = logpdf_GAU_ND(X, mu, C) + np.log(w)
        S.append(log_comp)
    S = np.vstack(S)
    logdens = logsumexp(S, axis=0)
    return logdens, S

def E_step(X, gmm):
    logdens, S = logpdf_GMM(X, gmm)
    loggamma = S - logdens
    gamma = np.exp(loggamma)
    return gamma, logdens.sum() / X.shape[1]  # return average log-likelihood

def M_step(X, gamma, cov_type='full', psi=0.01):
    N = X.shape[1]
    M = gamma.shape[0]
    D = X.shape[0]
    new_gmm = []
    
    if cov_type == 'tied':
        # Compute weighted means and covariances for all components first
        means = []
        covs = []
        weights = []

        for g in range(M):
            Zg = gamma[g].sum()
            Fg = (X * gamma[g]).sum(axis=1)
            Sg = (X * gamma[g]) @ X.T

            mu = Fg / Zg
            Sigma = Sg / Zg - np.outer(mu, mu)
            
            means.append(mu)
            covs.append(Sigma)
            weights.append(Zg / N)

        weights = np.array(weights)
        # Compute tied covariance as weighted average
        tied_cov = sum(w * cov for w, cov in zip(weights, covs))

        # Apply psi flooring to tied_cov
        U, s, Vh = np.linalg.svd(tied_cov)
        s[s < psi] = psi
        tied_cov = U @ np.diag(s) @ Vh

        # Each component shares the same tied covariance matrix
        for w, mu in zip(weights, means):
            new_gmm.append((w, mu, tied_cov))

        return new_gmm

    for g in range(M):
        Zg = gamma[g].sum()
        Fg = (X * gamma[g]).sum(axis=1)
        Sg = (X * gamma[g]) @ X.T

        mu = Fg / Zg
        Sigma = Sg / Zg - np.outer(mu, mu)
        w = Zg / N
        
        if cov_type == 'diag':
            Sigma = np.diag(np.diag(Sigma))
            
            # Apply eigenvalue flooring directly to diagonal elements
            diag_values = np.copy(np.diag(Sigma))
            diag_values[diag_values < psi] = psi
            Sigma = np.diag(diag_values)
        else:
            # For full covariance case
            U, s, Vh = np.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = U @ np.diag(s) @ U.T

        new_gmm.append((w, mu, Sigma))

    return new_gmm

def EM_algorithm(X, gmm_init, threshold=1e-6, max_iter=100,cov_type='full', psi=0.01):
    gmm = gmm_init
    prev_ll = None

    for i in range(max_iter):
        gamma, avg_ll = E_step(X, gmm)
        gmm = M_step(X, gamma, cov_type=cov_type, psi=psi)

        if prev_ll is not None and (avg_ll - prev_ll) < threshold:
            # print(f"EM converged at iteration {i+1}, avg log-likelihood = {avg_ll:.8f}")
            break
        prev_ll = avg_ll
        # print(f"Iter {i+1} - Avg log-likelihood: {avg_ll:.8f}")

    return gmm

def init_single_gaussian(X):
    mu = X.mean(axis=1)          # mean vector shape (D,)
    C = np.cov(X)                # covariance matrix shape (D, D)
    return [(1.0, mu, C)]


    
def LBG(X, gmm_init, num_components, alpha=0.1, tol=1e-6, max_iter=100, cov_type='full', psi=0.01):
    gmm = gmm_init
    while len(gmm) < num_components:
        new_gmm = []
        for w, mu, C in gmm:
            print(f"LBG: w={w}, mu.shape={mu.shape}, C.shape={C.shape}")
            if cov_type == 'diag':
                C = C * np.eye(X.shape[0])
            U, s, _ = np.linalg.svd(C)
            d = U[:, 0:1] * np.sqrt(s[0]) * alpha  # Shape (D, 1)

            mu = mu.reshape(-1, 1)  # (D,1)
            print(f"LBG: d.shape={d.shape}, mu.shape after reshape={mu.shape}")

            new_gmm.append((w / 2, (mu + d).flatten(), C))
            new_gmm.append((w / 2, (mu - d).flatten(), C))

        gmm = EM_algorithm(X, new_gmm, threshold=tol, max_iter=max_iter, cov_type=cov_type, psi=psi)
    return gmm




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

def optimal_bayes_decision(llr, prior, Cfn, Cfp):
    """
    Compute optimal Bayes decisions based on log-likelihood ratios
    """
    t = -np.log(prior * Cfn / ((1 - prior) * Cfp))
    return llr > t

def compute_confusion_matrix(predictions, labels, num_classes):
    """
    Compute confusion matrix
    """
    cm = np.zeros((num_classes, num_classes))
    for i in range(len(predictions)):
        cm[predictions[i], labels[i]] += 1
    return cm

def empirical_bayes_risks(cm, prior, Cfn, Cfp, normalize=True):
    """
    Compute empirical Bayes risk
    """
    FNR = cm[0, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    FPR = cm[1, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
    
    DCF = prior * Cfn * FNR + (1 - prior) * Cfp * FPR
    
    if normalize:
        # Normalize by minimum cost of dummy system
        dummy_DCF = min(prior * Cfn, (1 - prior) * Cfp)
        return DCF / dummy_DCF
    
    return DCF

def compute_min_DCF(llr, labels, prior, Cfn, Cfp, dcf_norm=None):
    """
    Compute minimum DCF by searching over thresholds
    """
    thresholds = np.sort(llr)
    min_dcf = float('inf')
    
    for t in thresholds:
        decisions = llr > t
        cm = compute_confusion_matrix(decisions.astype(int), labels, 2)
        dcf = empirical_bayes_risks(cm, prior, Cfn, Cfp, normalize=True)
        if dcf < min_dcf:
            min_dcf = dcf
    
    return min_dcf

#binary logistic regression
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
        
def compute_accuracy(D, L, w, b):
    S = 1 / (1 + np.exp(-(np.dot(w.T, D) + b)))
    P = (S > 0.5).astype(int)
    acc = (P == L).mean()
    return acc
        
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


if __name__ == "__main__":
    D, L = load_project_data('trainData.txt')
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    # Define parameters
    num_components_list = [1, 2, 4, 8, 16, 32]
    prior = 0.1  # Prior probability for the first class
    psi = 0.01  # eigenvalue flooring
    
    cov_types = ['full', 'diag']
    
    results = {}
    
    # for cov_type in cov_types:
    #     print(f"\nCovariance type: {cov_type.upper()}")
    #     results[cov_type] = {}
        
    #     for num_components_0 in num_components_list:
    #         for num_components_1 in num_components_list:
    #             model_key = f"{num_components_0}_{num_components_1}"
    #             print(f"Components class 0: {num_components_0}, class 1: {num_components_1}")
                
    #             # Train GMM for each class with different number of components
    #             gmm_class_0 = train_gmms_per_class(DTR[:, LTR==0], num_components_0, cov_type=cov_type, psi=psi)
    #             gmm_class_1 = train_gmms_per_class(DTR[:, LTR==1], num_components_1, cov_type=cov_type, psi=psi)
                
    #             # Compute log-likelihoods on test set
    #             ll_class_0 = logpdf_GMM(DTE, gmm_class_0)
    #             ll_class_1 = logpdf_GMM(DTE, gmm_class_1)
                
    #             # Log-likelihood ratio
    #             llr = ll_class_1 - ll_class_0
                
    #             # Compute optimal Bayes decisions using given prior
    #             decisions = optimal_bayes_decision(llr, prior, 1, 1)
                
    #             # Compute confusion matrix and performance metrics
    #             cm = compute_confusion_matrix(decisions.astype(int), LTE, 2)
    #             dcf_norm = empirical_bayes_risks(cm, prior, 1, 1, normalize=True)
    #             min_dcf = compute_min_DCF(llr, LTE, prior, 1, 1)
                
    #             results[cov_type][model_key] = {
    #                 'components': (num_components_0, num_components_1),
    #                 'actual_dcf': dcf_norm,
    #                 'min_dcf': min_dcf
    #             }
                
    #             print(f"Actual DCF: {dcf_norm:.4f}, min DCF: {min_dcf:.4f}\n")
    
    # # Print best results for each covariance type
    # print("\n===== SUMMARY OF RESULTS =====")
    # for cov_type in cov_types:
    #     best_model = min(results[cov_type].items(), key=lambda x: x[1]['min_dcf'])
    #     comp_0, comp_1 = best_model[1]['components']
    #     min_dcf = best_model[1]['min_dcf']
        
    #     print(f"\nBest {cov_type.upper()} covariance model:")
    #     print(f"Components for class 0: {comp_0}, Components for class 1: {comp_1}")
    #     print(f"Minimum DCF: {min_dcf:.4f}")
    
    
    #bayes plot for best model found
    # Choose GMM model configuration
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
    
    # Plot Bayes Error Plot
    bayes_error_plot(llr, LTE, title=f"Bayes Error Plot (GMM {cov_type.upper()}, ({num_components_0},{num_components_1}))")

    # # GMM Parameters
    # num_components_class0 = 8
    # num_components_class1 = 16

    # min_min_dcf=[]
    # min_act_dcf=[]
    # gmm_results = {}
    # # Evaluate GMM Models
    # for cov_type in ["full", "diagonal"]:
    #     print(f"\nEvaluating GMMs with {cov_type.capitalize()} Covariance:")
    #     for prior in priors:
    #         results = evaluate_gmm_model(DTR, LTR, DVAL, LVAL, num_components_list, cov_type, prior)
    #         print(f"\nPrior: {prior}")
    #         # print(results)
    #         min_dcf_values = [r['minDCF'] for r in results]
    #         act_dcf_values = [r['actDCF'] for r in results]
    #         gmm_results[f"GMM {cov_type.capitalize()} π={prior}"] = (min_dcf_values, act_dcf_values)
            

    # # Evaluate Logistic Regression
    # print("\nEvaluating Logistic Regression:")
    # DTR_quad = quadratic_expansion(DTR)
    # DVAL_quad = quadratic_expansion(DVAL)
    # lr_results = {}
    # lambdas = np.logspace(-4, 2, 13)  # λ from 10^-4 to 10^2
    # pi_t = 0.1  # Prior for target class
    
    # # Train and find the best λ
    # best_lambda, best_min_dcf, results = train_logistic_regression(DTR_quad, LTR, lambdas, DVAL_quad, LVAL, pi_t)
    # min_act_dcf_entry = min(results, key=lambda x: x['actDCF'])
    # print(f"Best λ: {best_lambda}, Best minDCF: {best_min_dcf}, Minimum actDCF: {min_act_dcf_entry['actDCF']:.4f}")
    # lr_results["Logistic Regression"] = (best_min_dcf, min_act_dcf_entry)
    
    # # Evaluate SVM
    # print("\nEvaluating SVM:")
    # svm_results = {}
    # C_values = np.logspace(-3, 3, 7)
    # kernelFunc = rbfKernel(0.1)
    # min_dcf_values, act_dcf_values = evaluate_svm_kernel(DTR, LTR, DVAL, LVAL, C_values, kernelFunc)
    # print(f" minDCF: {min(min_dcf_values):.4f}, actDCF: {min(act_dcf_values):.4f}")
    # svm_results["SVM"] = (min_dcf_values, act_dcf_values)
    # # # Combine Results
    # # models_results = {
    # #     "GMM Full": (min_min_dcf, min_act_dcf),
    # #     "Logistic Regression": (best_min_dcf, min_act_dcf_entry),
    # #     "SVM": (min_dcf_values, act_dcf_values)
    # # }
    
    # models_results = {**gmm_results, **lr_results, **svm_results}

    # # Plot Bayes Error
    # qualitative_analysis(DTR, LTR, DVAL, LVAL, log_odds_range, models_results)