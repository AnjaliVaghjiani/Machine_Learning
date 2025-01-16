# -*- coding: utf-8 -*-

import numpy
import numpy as np
import scipy.special
import sklearn.datasets
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))
def load_project_data(file_path):
    data = numpy.loadtxt(file_path, delimiter=',')
    D = data[:, :-1].T
    L = data[:, -1].astype(int)
    return D, L

def split_db_2to1(D, L, seed=0):
    
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']    
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

# Optimize the logistic regression loss
def trainLogRegBinary(DTR, LTR, l):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        GW = (vrow(G) * DTR).mean(1) + l * w.ravel()
        Gb = G.mean()
        return loss.mean() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Log-reg - lambda = %e - J*(w, b) = %e" % (l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

# Optimize the weighted logistic regression loss
def trainWeightedLogRegBinary(DTR, LTR, l, pT):

    ZTR = LTR * 2.0 - 1.0 # We do it outside the objective function, since we only need to do it once
    
    wTrue = pT / (ZTR>0).sum() # Compute the weights for the two classes
    wFalse = (1-pT) / (ZTR<0).sum()

    def logreg_obj_with_grad(v): # We compute both the objective and its gradient to speed up the optimization
        w = v[:-1]
        b = v[-1]
        s = numpy.dot(vcol(w).T, DTR).ravel() + b

        loss = numpy.logaddexp(0, -ZTR * s)
        loss[ZTR>0] *= wTrue # Apply the weights to the loss computations
        loss[ZTR<0] *= wFalse

        G = -ZTR / (1.0 + numpy.exp(ZTR * s))
        G[ZTR > 0] *= wTrue # Apply the weights to the gradient computations
        G[ZTR < 0] *= wFalse
        
        GW = (vrow(G) * DTR).sum(1) + l * w.ravel()
        Gb = G.sum()
        return loss.sum() + l / 2 * numpy.linalg.norm(w)**2, numpy.hstack([GW, numpy.array(Gb)])

    vf = scipy.optimize.fmin_l_bfgs_b(logreg_obj_with_grad, x0 = numpy.zeros(DTR.shape[0]+1))[0]
    print ("Weighted Log-reg (pT %e) - lambda = %e - J*(w, b) = %e" % (pT, l, logreg_obj_with_grad(vf)[0]))
    return vf[:-1], vf[-1]

def evaluate_logistic_regression(DTR, LTR, DVAL, LVAL, lambdas, priors):
    results = {}
    
    for l in lambdas:
        results[l] = {}
        for pi_t in priors:
            # Standard Logistic Regression
            w, b = trainLogRegBinary(DTR, LTR, l)
            sVal = numpy.dot(w.T, DVAL) + b
            pEmp = (LTR == 1).sum() / LTR.size
            sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))

            # Compute DCFs
            min_dcf = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
            act_dcf = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
            results[l][pi_t] = {'min_dcf': min_dcf, 'act_dcf': act_dcf}
    
    return results

def plot_dcf_vs_lambda(lambdas, results):
    plt.figure(figsize=(10, 6))
    
    for pi_t in results[lambdas[0]].keys():
        min_dcf_values = [results[l][pi_t]['min_dcf'] for l in lambdas]
        act_dcf_values = [results[l][pi_t]['act_dcf'] for l in lambdas]
        
        plt.plot(lambdas, min_dcf_values, label=f"Min DCF, πT={pi_t}", linestyle='--', marker='o')
        plt.plot(lambdas, act_dcf_values, label=f"Actual DCF, πT={pi_t}", marker='o')
    
    plt.xscale('log', base=10)
    plt.xlabel("λ (Regularization Parameter)")
    plt.ylabel("Detection Cost Function (DCF)")
    plt.title("Actual and Minimum DCF vs. Regularization Parameter (λ) for Different Priors")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def evaluate_prior_weighted_logistic_regression(DTR, LTR, DVAL, LVAL, lambdas, pi_t):
    min_dcf_values = []
    actual_dcf_values = []

    for lamb in lambdas:
        w, b = trainWeightedLogRegBinary(DTR, LTR, lamb, pi_t)  # Train the prior-weighted logistic regression model
        sVal = numpy.dot(w.T, DVAL) + b  # Compute the validation scores

        # Since we trained with a prior-weighted model, we need to adjust the scores accordingly
        sValLLR = sVal - numpy.log(pi_t / (1 - pi_t))

        # Compute min DCF and actual DCF
        min_dcf = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
        actual_dcf = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)

        min_dcf_values.append(min_dcf)
        actual_dcf_values.append(actual_dcf)

    return min_dcf_values, actual_dcf_values

# Plot function for prior-weighted logistic regression results
def plot_prior_weighted_dcf_vs_lambda(lambdas, min_dcf_values, actual_dcf_values, pi_t):
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, min_dcf_values, label=f'Min DCF, πT={pi_t}', linestyle='--', marker='o')
    plt.plot(lambdas, actual_dcf_values, label=f'Actual DCF, πT={pi_t}', linestyle='-', marker='o')
    plt.xscale('log')
    plt.xlabel('λ (Regularization Parameter)')
    plt.ylabel('Detection Cost Function (DCF)')
    plt.title('Prior-Weighted Logistic Regression: Actual and Minimum DCF vs. Regularization Parameter (λ)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
import bayesRisk 

def center_data(DTR, DVAL):
    mean_TR = DTR.mean(axis=1, keepdims=True)
    DTR_centered = DTR - mean_TR
    DVAL_centered = DVAL - mean_TR  # Center using training mean
    return DTR_centered, DVAL_centered

def z_normalize(DTR, DVAL):
    mean_TR = DTR.mean(axis=1, keepdims=True)
    std_TR = DTR.std(axis=1, keepdims=True)
    DTR_norm = (DTR - mean_TR) / std_TR
    DVAL_norm = (DVAL - mean_TR) / std_TR  # Normalize using training mean and std
    return DTR_norm, DVAL_norm

def whiten_data(DTR, DVAL):
    mean_TR = DTR.mean(axis=1, keepdims=True)
    DTR_centered = DTR - mean_TR
    DVAL_centered = DVAL - mean_TR

    cov_matrix = np.cov(DTR_centered)
    U, S, _ = np.linalg.svd(cov_matrix)
    whitening_matrix = np.dot(U, np.diag(1.0 / np.sqrt(S + 1e-5)))
    
    DTR_whitened = np.dot(whitening_matrix.T, DTR_centered)
    DVAL_whitened = np.dot(whitening_matrix.T, DVAL_centered)
    return DTR_whitened, DVAL_whitened

def plot_dcf_comparison(lambdas, results, labels):
    plt.figure(figsize=(10, 6))
    
    for i, (min_dcf_values, actual_dcf_values) in enumerate(results):
        plt.plot(lambdas, min_dcf_values, label=f"{labels[i]} Min DCF", linestyle='--')
        plt.plot(lambdas, actual_dcf_values, label=f"{labels[i]} Actual DCF", linestyle='-')
    
    plt.xscale('log', base=10)
    plt.xlabel("λ (Regularization Parameter)")
    plt.ylabel("Detection Cost Function (DCF)")
    plt.title("Effect of Preprocessing on Logistic Regression DCF")
    plt.legend()
    plt.grid(True)
    plt.show()

def quadratic_expansion(D):
    """Expand the feature matrix D quadratically."""
    n_features = D.shape[0]
    expanded_features = [D]
    for i in range(n_features):
        for j in range(i, n_features):
            expanded_features.append(D[i] * D[j])
    return np.vstack(expanded_features)

def evaluate_logistic_regression_with_preprocessing(DTR, LTR, DVAL, LVAL, lambdas, pi_t, preprocess_fn):
    min_dcf_values = []
    actual_dcf_values = []

    DTR_processed, DVAL_processed = preprocess_fn(DTR, DVAL)
    
    for lamb in lambdas:
        w, b = trainLogRegBinary(DTR_processed, LTR, lamb)
        sVal = np.dot(w.T, DVAL_processed) + b
        
        # Remove log-odds of the training set empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        sValLLR = sVal - np.log(pEmp / (1 - pEmp))
        
        # Compute DCFs
        min_dcf = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
        act_dcf = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
        
        min_dcf_values.append(min_dcf)
        actual_dcf_values.append(act_dcf)
    
    return min_dcf_values, actual_dcf_values

def apply_pca(DTR, DVAL, m):
    mu = np.mean(DTR, axis=1, keepdims=True)
    DTR_centered = DTR - mu
    DVAL_centered = DVAL - mu
    
    U, S, Vh = np.linalg.svd(DTR_centered, full_matrices=False)
    P = U[:, :m]
    
    DTR_pca = np.dot(P.T, DTR_centered)
    DVAL_pca = np.dot(P.T, DVAL_centered)
    
    return DTR_pca, DVAL_pca

def evaluate_logistic_regression_with_pca(DTR, LTR, DVAL, LVAL, lambdas, pi_t, m_values):
    results = {}

    for m in m_values:
        results[m] = {}
        DTR_pca, DVAL_pca = apply_pca(DTR, DVAL, m)

        for l in lambdas:
            w, b = trainLogRegBinary(DTR_pca, LTR, l)
            sVal = np.dot(w.T, DVAL_pca) + b
            pEmp = (LTR == 1).sum() / LTR.size
            sValLLR = sVal - np.log(pEmp / (1 - pEmp))

            # Compute DCFs
            min_dcf = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
            act_dcf = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
            results[m][l] = {'min_dcf': min_dcf, 'act_dcf': act_dcf}

    return results

def plot_dcf_vs_lambda_pca(lambdas, results):
    plt.figure(figsize=(10, 6))

    for m in results.keys():
        min_dcf_values = [results[m][l]['min_dcf'] for l in lambdas]
        act_dcf_values = [results[m][l]['act_dcf'] for l in lambdas]

        plt.plot(lambdas, min_dcf_values, label=f"Min DCF, m={m}", linestyle='--', marker='o')
        plt.plot(lambdas, act_dcf_values, label=f"Actual DCF, m={m}", marker='o')

    plt.xscale('log', base=10)
    plt.xlabel("λ (Regularization Parameter)")
    plt.ylabel("Detection Cost Function (DCF)")
    plt.title("Actual and Minimum DCF vs. Regularization Parameter (λ) with PCA")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    
    # D, L = load_iris_binary()
    # (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    file_path = 'Traindata.txt'  # Replace with your actual file path
    D, L = load_project_data(file_path)
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    
    lambdas = numpy.logspace(-4, 2, 13)
    priors = [0.1]

   # Evaluate logistic regression model
    results = evaluate_logistic_regression(DTR, LTR, DVAL, LVAL, lambdas, priors)

   # Plot results
    plot_dcf_vs_lambda(lambdas, results)
    
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # Reduce the training set to 1 out of every 50 samples
    DTR_reduced = DTR[:, ::50]
    LTR_reduced = LTR[::50]
    
    # Define the range of lambda values (regularization parameter)
    lambdas = numpy.logspace(-4, 2, 13)
    
    # To store the results
    min_dcf_values = []
    actual_dcf_values = []
    
    # Prior for the primary application
    pi_t = 0.1
    
    # Loop over the lambda values
    for lamb in lambdas:
        # Train model on reduced dataset
        w, b = trainLogRegBinary(DTR_reduced, LTR_reduced, lamb)
        
        # Compute validation scores
        sVal = numpy.dot(w.T, DVAL) + b
        
        # Compute empirical prior
        pEmp = (LTR == 1).sum() / LTR.size
        
        # Compute LLR-like scores
        sValLLR = sVal - numpy.log(pEmp / (1 - pEmp))
        
        # Compute min DCF and actual DCF
        min_dcf = bayesRisk.compute_minDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
        actual_dcf = bayesRisk.compute_actDCF_binary_fast(sValLLR, LVAL, pi_t, 1.0, 1.0)
        
        # Store results
        min_dcf_values.append(min_dcf)
        actual_dcf_values.append(actual_dcf)
    
    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, min_dcf_values, label='Min DCF, πT=0.1', marker='o', linestyle='--')
    plt.plot(lambdas, actual_dcf_values, label='Actual DCF, πT=0.1', marker='o', linestyle='-')
    plt.xscale('log')
    plt.xlabel('λ (Regularization Parameter)')
    plt.ylabel('Detection Cost Function (DCF)')
    plt.title('Actual and Minimum DCF vs. Regularization Parameter (λ) for Reduced Training Set')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #Prior-version
    lambdas = numpy.logspace(-4, 2, 13)
    pi_t = 0.1
    
    # Evaluate prior-weighted logistic regression
    min_dcf_values_pw, actual_dcf_values_pw = evaluate_prior_weighted_logistic_regression(DTR, LTR, DVAL, LVAL, lambdas, pi_t)
    
    # Plot the results
    plot_prior_weighted_dcf_vs_lambda(lambdas, min_dcf_values_pw, actual_dcf_values_pw, pi_t)
    
    # Quadratic expansion of features
    DTR_quad = quadratic_expansion(DTR)
    DVAL_quad = quadratic_expansion(DVAL)
 
    lambdas = np.logspace(-4, 2, 13)
    pi_t = [0.1]
    results = evaluate_logistic_regression(DTR_quad, LTR, DVAL_quad, LVAL, lambdas,pi_t)
    # print(results)
    plot_dcf_vs_lambda(lambdas, results)
    
    # # Analyze different preprocessing strategies
    preprocess_fns = [center_data, z_normalize, whiten_data]
    preprocess_labels = ["Centered", "Z-normalized", "Whitened"]
    
    results = []
    pi_t = 0.1
    pEmp = [(LTR == 1).sum() / LTR.size]
    for fn in preprocess_fns:
        min_dcf, act_dcf = evaluate_logistic_regression_with_preprocessing(DTR, LTR, DVAL, LVAL, lambdas, pi_t, fn)
        results.append((min_dcf, act_dcf))
   
    plot_dcf_comparison(lambdas, results, preprocess_labels)
    #ploat using pca
    m_values = [1,2,3,4,5,6]  # Adjust these values based on your dataset's dimensionality
    pi_t = 0.1
 
    results_pca = evaluate_logistic_regression_with_pca(DTR, LTR, DVAL, LVAL, lambdas, pi_t, m_values)
 
    plot_dcf_vs_lambda_pca(lambdas, results_pca)