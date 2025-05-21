import numpy as np
from scipy.special import logsumexp
import sklearn.datasets
import matplotlib.pyplot as plt
import scipy.special

import json

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def save_gmm(gmm, filename):
    gmmJson = [(i, j.tolist(), k.tolist()) for i, j, k in gmm]
    with open(filename, 'w') as f:
        json.dump(gmmJson, f)
    
def load_gmm(filename):
    with open(filename, 'r') as f:
        gmm = json.load(f)
    return [(i, np.asarray(j), np.asarray(k)) for i, j, k in gmm]

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

def plot_gmm_density_1D(X, gmm, num_points=1000):
    # Create a grid of x values to evaluate the density
    x_min, x_max = X.min(), X.max()
    x_grid = np.linspace(x_min - 1, x_max + 1, num_points).reshape(1, -1)  # Shape (1, num_points)

    # Compute the log-density from the GMM
    log_dens, S = logpdf_GMM(x_grid, gmm)
    dens = np.exp(log_dens)  # Convert to actual density

    # Plot the GMM density
    plt.figure(figsize=(8, 4))
    plt.plot(x_grid.flatten(), dens, label='Estimated GMM Density')
    plt.hist(X.flatten(), bins=50, density=True, alpha=0.5, label='Empirical Histogram')
    plt.title('GMM Estimated Density vs Empirical Histogram')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def init_single_gaussian(X):
    mu = X.mean(axis=1, keepdims=True)
    C = np.cov(X)
    return [(1.0, mu.flatten(), C)]
    
def LBG(X, gmm_init, num_components, alpha=0.1, tol=1e-6, max_iter=100, cov_type='full', psi=0.01):
    
    gmm = gmm_init
    while len(gmm) < num_components:
        new_gmm = []
        for w, mu, C in gmm:
            if cov_type == 'diag':
                C = C * np.eye(X.shape[0])
            U, s, _ = np.linalg.svd(C)
            d = U[:, 0:1] * np.sqrt(s[0]) * alpha  # Shape (D, 1)

            mu = mu.reshape(-1, 1)  # Ensure shape (D, 1)
            new_gmm.append((w / 2, (mu + d).flatten(), C))
            new_gmm.append((w / 2, (mu - d).flatten(), C))

        # Run EM on the 2G GMM
        gmm = EM_algorithm(X, new_gmm, threshold=tol, max_iter=max_iter, cov_type=cov_type, psi=psi)
    return gmm

def diagonal_covariance(cov):
    return cov * np.eye(cov.shape[0])

def tied_covariance(covariances, weights):
    # covariances: list or array of shape (M, D, D)
    # weights: array of shape (M,)
    tied_cov = sum(w * cov for w, cov in zip(weights, covariances))
    return tied_cov  

def train_gmms_per_class(DTR, LTR, num_components=4, alpha=0.1, psi=0.01, em_tol=1e-6, em_max_iter=100, cov_type = 'full'):
    classes = np.unique(LTR)
    class_gmms = {}

    for c in classes:
        # print(f"Training GMM for class {c}")
        Xc = DTR[:, LTR == c]
        gmm_init = init_single_gaussian(Xc)
        gmm = LBG(Xc, gmm_init, num_components=num_components, alpha=alpha, tol=em_tol, max_iter=em_max_iter,cov_type=cov_type, psi=psi)

        # Constrain eigenvalues after each M-step (optional, see note below)
        # You could add that inside EM_algorithm's M_step if you want stricter control.

        class_gmms[c] = gmm

    return class_gmms


def predict_gmm(DTE, class_gmms, priors=None):
    classes = list(class_gmms.keys())
    N = DTE.shape[1]
    log_posteriors = np.zeros((len(classes), N))

    if priors is None:
        priors = {c: 1 / len(classes) for c in classes}

    for i, c in enumerate(classes):
        gmm = class_gmms[c]
        log_likelihood, _ = logpdf_GMM(DTE, gmm)
        log_prior = np.log(priors[c])
        log_posteriors[i, :] = log_likelihood + log_prior

    predicted_indices = np.argmax(log_posteriors, axis=0)
    predicted_classes = np.array([classes[i] for i in predicted_indices])
    return predicted_classes


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

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

    
if __name__ == '__main__':
    D, L = load_iris()
    
    #DTR and LTR are traininig and DTE and LTE as evaluation
    (DTR, LTR), (DTE, LTE) = split_db_2tot(D,L)
    
    X = np.load('GMM_data_4D.npy')
    gmm = load_gmm('GMM_4D_3G_init.json')
    ll_ref_4D = np.load("GMM_4D_3G_init_ll.npy")
    
    # log_densities, S = logpdf_GMM(X, gmm)
    
    # #chech with reference
    # max_abs_diff = np.max(np.abs(log_densities - ll_ref_4D))
    # print("Max absolute difference:", max_abs_diff)
    
    # # Check closeness
    # if np.allclose(log_densities, ll_ref_4D, atol=1e-6):
    #     print("Log-densities match the reference!")
    # else:
    #     print("Log-densities do not match!")
        
        
    # #GMM estimation and the EM algorithms
    # X_1D = np.load("GMM_data_1D.npy")
    # gmm_1D = load_gmm("GMM_1D_3G_EM.json")
    # trained_gmm = EM_algorithm(X, gmm)
    # final_loglikelihood, _ = logpdf_GMM(X, trained_gmm)
    # print("Final average log-likelihood:", final_loglikelihood.mean())
    
    # #plot for 1d dataset check
    # plot_gmm_density_1D(X_1D, gmm_1D)
    
    #LBG algorithms
    # gmm_init = init_single_gaussian(X)
    # gmm_4G = LBG(X, gmm_init, num_components=4, alpha=0.1)
    # save_gmm(gmm_4G, "GMM_4D_4G_EM_LBG.json")
    
    #plot for 1D data
    # Xlbg_1D  = np.load("GMM_data_1D.npy")
    # gmmLbg_1D = load_gmm('GMM_1D_4G_EM_LBG.json')
    # plot_gmm_density_1D(Xlbg_1D, gmmLbg_1D)
    
    
    #constrainig the eigenvalue of the covariance matrices (Full covarianec)
    components = [1, 2, 4, 8, 16]
    cov_types = ['full', 'diag', 'tied']
    psi = 0.01  # eigenvalue flooring
    
    # for cov_type in cov_types:
    #     print(f"\nCovariance type: {cov_type.upper()}")
    #     for num_components in components:
    #         class_gmms = train_gmms_per_class(DTR, LTR, num_components=num_components, cov_type=cov_type, psi=psi)
    
    #         pred_train = predict_gmm(DTR, class_gmms)
    #         pred_test = predict_gmm(DTE, class_gmms)
    
    #         train_acc = accuracy_score(LTR, pred_train)
    #         test_acc = accuracy_score(LTE, pred_test)
    
    #         print(f"Components: {num_components} | Test Error: {(1 - test_acc)*100:.2f}%")
   
    #mindcf /actdcf
    D= np.load('ext_data_binary.npy')
    L = np.load('ext_data_binary_labels.npy')
    (DTR, LTR), (DTE, LTE) = split_db_2tot(D,L)
    for cov_type in cov_types:
        print(f"\nCovariance type: {cov_type.upper()}")
        for num_components in components:
            print(f"Component: {num_components}")
            class_gmms = train_gmms_per_class(DTR, LTR, num_components=num_components, cov_type=cov_type, psi=psi)
    
            # Compute log-likelihoods on validation set for each class GMM
            log_likelihoods_class0, _ = logpdf_GMM(DTE, class_gmms[0])
            log_likelihoods_class1, _ = logpdf_GMM(DTE, class_gmms[1])
            
            llr = log_likelihoods_class1 - log_likelihoods_class0
            
            pi_T = prior= 0.5
            cfn = 1  # cost false negative
            cpn = 1  # cost false positive
            
            decisions = optimal_bayes_decision(llr ,pi_T , cfn, cpn)

            cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
            # print(cm)
            #Minimum Detection Cost
            dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
            print(f"Actual DCF : {dcf_norm:.4f}")
            min_cost = compute_min_DCF(llr, LTE, prior, cfn, cpn, dcf_norm)
            print(f"min DCF: {min_cost:.4f}") 
            print()
            
            
            

