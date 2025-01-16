import numpy as np
from gmm import train_GMM_LBG_EM, logpdf_GMM
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

if __name__ == "__main__":
    # Load data
    D, L = load_project_data('trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Define parameters
    num_components_list = [1, 2, 4, 8, 16, 32]
    priors = [0.1]
    log_odds_range = np.linspace(-4, 4, 50)
    
    # GMM Parameters
    num_components_class0 = 8
    num_components_class1 = 16

    min_min_dcf=[]
    min_act_dcf=[]
    gmm_results = {}
    # Evaluate GMM Models
    for cov_type in ["full", "diagonal"]:
        print(f"\nEvaluating GMMs with {cov_type.capitalize()} Covariance:")
        for prior in priors:
            results = evaluate_gmm_model(DTR, LTR, DVAL, LVAL, num_components_list, cov_type, prior)
            print(f"\nPrior: {prior}")
            # print(results)
            min_dcf_values = [r['minDCF'] for r in results]
            act_dcf_values = [r['actDCF'] for r in results]
            gmm_results[f"GMM {cov_type.capitalize()} π={prior}"] = (min_dcf_values, act_dcf_values)
            

    # Evaluate Logistic Regression
    print("\nEvaluating Logistic Regression:")
    DTR_quad = quadratic_expansion(DTR)
    DVAL_quad = quadratic_expansion(DVAL)
    lr_results = {}
    lambdas = np.logspace(-4, 2, 13)  # λ from 10^-4 to 10^2
    pi_t = 0.1  # Prior for target class
    
    # Train and find the best λ
    best_lambda, best_min_dcf, results = train_logistic_regression(DTR_quad, LTR, lambdas, DVAL_quad, LVAL, pi_t)
    min_act_dcf_entry = min(results, key=lambda x: x['actDCF'])
    print(f"Best λ: {best_lambda}, Best minDCF: {best_min_dcf}, Minimum actDCF: {min_act_dcf_entry['actDCF']:.4f}")
    lr_results["Logistic Regression"] = (best_min_dcf, min_act_dcf_entry)
    
    # Evaluate SVM
    print("\nEvaluating SVM:")
    svm_results = {}
    C_values = np.logspace(-3, 3, 7)
    kernelFunc = rbfKernel(0.1)
    min_dcf_values, act_dcf_values = evaluate_svm_kernel(DTR, LTR, DVAL, LVAL, C_values, kernelFunc)
    print(f" minDCF: {min(min_dcf_values):.4f}, actDCF: {min(act_dcf_values):.4f}")
    svm_results["SVM"] = (min_dcf_values, act_dcf_values)
    # # Combine Results
    # models_results = {
    #     "GMM Full": (min_min_dcf, min_act_dcf),
    #     "Logistic Regression": (best_min_dcf, min_act_dcf_entry),
    #     "SVM": (min_dcf_values, act_dcf_values)
    # }
    
    models_results = {**gmm_results, **lr_results, **svm_results}

    # Plot Bayes Error
    qualitative_analysis(DTR, LTR, DVAL, LVAL, log_odds_range, models_results)