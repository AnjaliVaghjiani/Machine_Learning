import numpy as np
import matplotlib.pyplot as plt
import bayesRisk
import logReg
import scipy.special
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



# Main execution
if __name__ == '__main__':
    # Load training data
    D, L = load_project_data('trainData.txt')
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    # Evaluate and calibrate models
    calibrated_gmm, calibrated_lr, calibrated_svm = evaluate_and_calibrate_models(DTR, LTR, DVAL, LVAL)

    # Perform score fusion and final evaluation
    fuse_and_evaluate_models(calibrated_gmm, calibrated_lr, calibrated_svm, LVAL)

   
    # Load evaluation dataset
    print("\nEvaluating models on Project/evalData.txt...")
    DEVAL, LEVAL = load_project_data('evalData.txt')

    # Evaluate on the evaluation dataset
    scores_gmm_eval, scores_lr_eval, scores_svm_eval = evaluate_and_calibrate_models(DEVAL, LEVAL, DVAL, LVAL)

    # Perform fusion evaluation on the evaluation dataset
    fuse_and_evaluate_models(scores_gmm_eval, scores_lr_eval, scores_svm_eval, LEVAL)