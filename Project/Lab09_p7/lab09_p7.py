############################################################################################
# Copyright (C) 2024 by Sandro Cumani                                                      #
#                                                                                          #
# This file is provided for didactic purposes only, according to the Politecnico di Torino #
# policies on didactic material.                                                           #
#                                                                                          #
# Any form of re-distribution or online publication is forbidden.                          #
#                                                                                          #
# This file is provided as-is, without any warranty                                        #
############################################################################################

import numpy as np
import scipy.special
import sklearn.datasets
import matplotlib.pyplot as plt
import bayesRisk
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
    # Dlist = []
    # labelsList = []
    # # hLablels = {
    #     # 'False': 0,
    #     # 'True':1
    # # }
    # with open(file_path) as f:   
    #     for line in f:
    #         try:
    #             attrs = line.split(',')[0:-1] # cuz we don't want the last value
    #             attrs = vcol(np.array([float(i) for i in attrs])) # here we have an arr like this [5.1,3.5,1.4,0.2, 1.2,3.2] 6 features
    #             # But we want 6x1 arr. hence
    #             label = line.split(',')[-1].strip() # here we save the last value which is the name of the lable eg. True or False
    #             Dlist.append(attrs)
    #             labelsList.append(int(label)) # 0 is False and 1 is True
    #         except: 
    #             pass
    # return np.hstack(Dlist), np.array(labelsList, dtype=np.int32) 


# # Optimize SVM
# def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
#     ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
#     DTR_EXT = np.vstack([DTR, np.ones((1,DTR.shape[1])) * K])
#     H = np.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

#     # Dual objective with gradient
#     def fOpt(alpha):
#         Ha = H @ vcol(alpha)
#         loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
#         grad = Ha.ravel() - np.ones(alpha.size)
#         return loss, grad

#     alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
#     # Primal loss
#     def primalLoss(w_hat):
#         S = (vrow(w_hat) @ DTR_EXT).ravel()
#         return 0.5 * np.linalg.norm(w_hat)**2 + C * np.maximum(0, 1 - ZTR * S).sum()

#     # Compute primal solution for extended data matrix
#     w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
#     # Extract w and b - alternatively, we could construct the extended matrix for the samples to score and use directly v
#     w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K # b must be rescaled in case K != 1, since we want to compute w'x + b * K

#     primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
#     # print("The primal loss ", primalLoss)
#     # print("dualLoss ", dualLoss)
#     print ('SVM - C %e - K %e - primal loss %e - dual loss %e - duality gap %e' % (C, K, primalLoss, dualLoss, primalLoss - dualLoss))
    
#     return w, b

# # We create the kernel function. Since the kernel function may need additional parameters, we create a function that creates on the fly the required kernel function
# # The inner function will be able to access the arguments of the outer function
# def polyKernel(degree, c):
    
#     def polyKernelFunc(D1, D2):
#         return (np.dot(D1.T, D2) + c) ** degree

#     return polyKernelFunc

# def rbfKernel(gamma):

#     def rbfKernelFunc(D1, D2):
#         # Fast method to compute all pair-wise distances. Exploit the fact that |x-y|^2 = |x|^2 + |y|^2 - 2 x^T y, combined with broadcasting
#         D1Norms = (D1**2).sum(0)
#         D2Norms = (D2**2).sum(0)
#         Z = vcol(D1Norms) + vrow(D2Norms) - 2 * np.dot(D1.T, D2)
#         return np.exp(-gamma * Z)

#     return rbfKernelFunc

# # kernelFunc: function that computes the kernel matrix from two data matrices
# def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

#     ZTR = LTR * 2.0 - 1.0 # Convert labels to +1/-1
#     K = kernelFunc(DTR, DTR) + eps
#     H = vcol(ZTR) * vrow(ZTR) * K

#     # Dual objective with gradient
#     def fOpt(alpha):
#         Ha = H @ vcol(alpha)
#         loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
#         grad = Ha.ravel() - np.ones(alpha.size)
#         return loss, grad

#     alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, np.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

#     print ('SVM (kernel) - C %e - dual loss %e' % (C, -fOpt(alphaStar)[0]))

#     # Function to compute the scores for samples in DTE
#     def fScore(DTE):
        
#         K = kernelFunc(DTR, DTE) + eps
#         H = vcol(alphaStar) * vcol(ZTR) * K
#         return H.sum(0)

#     return fScore # we directly return the function to score a matrix of test samples
# def plot_dcf_vs_C(C_values, min_dcf_values, act_dcf_values, title):
#     plt.figure(figsize=(10, 6))
#     plt.plot(C_values, min_dcf_values, label='Min DCF, πT=0.1', linestyle='--', marker='o')
#     plt.plot(C_values, act_dcf_values, label='Actual DCF, πT=0.1', marker='o')
#     plt.xscale('log', base=10)
#     plt.xlabel('C (Regularization Parameter)')
#     plt.ylabel('Detection Cost Function (DCF)')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
###################################################################
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

def compute_dual_svm(DTR, LTR, C, K=1.0, factr=np.nan, pgtol=1e-5, maxiter=15000):
    print(f"for K: {K} and C:{C}")    
    Z = 2 * LTR - 1   # Labels in {-1, +1}

    # Extended training matrix with bias term scaled by K
    Db = np.vstack([DTR, np.ones(DTR.shape[1]) * K])
    Zb = Z

    # Compute Gram matrix Gb and H matrix
    Gb = np.dot(Db.T, Db)
    H = np.outer(Zb, Zb) * Gb

    # Dual objective (to minimize)
    def LbD(alpha):
        Ha = np.dot(H, alpha)
        obj = 0.5 * np.dot(alpha, Ha) - np.sum(alpha)
        grad = Ha - np.ones_like(alpha)
        return obj, grad

    n = DTR.shape[1]
    bounds = [(0, C)] * n
    x0 = np.zeros(n)

    alpha_opt, f, d = scipy.optimize.fmin_l_bfgs_b(
        func=LbD,
        x0=x0,
        bounds=bounds,
        factr=factr,
        pgtol=pgtol,
        maxiter=maxiter
    )

    # Compute primal solution
    wb_star = Db @ (alpha_opt * Z)
    w_star = wb_star[:-1]
    b_star = wb_star[-1] * K  # Rescale bias term

    # Primal objective
    def primal_objective(wb):
        scores = np.dot(wb.T, Db)
        hinge_losses = np.maximum(0, 1 - Zb * scores)
        return 0.5 * np.linalg.norm(wb)**2 + C * np.sum(hinge_losses)

    J_primal = primal_objective(wb_star)
    J_dual = -f
    duality_gap = J_primal - J_dual

    return w_star, b_star, alpha_opt, J_primal, J_dual, duality_gap

def plot_dcf_vs_C(C_values, minDCF_values, actDCF_values, pi_T=0.1):
    plt.figure(figsize=(8, 6))
    plt.plot(C_values, minDCF_values, label='minDCF', marker='o')
    plt.plot(C_values, actDCF_values, label='actDCF', marker='x')

    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel(f'DCF (πT = {pi_T})')
    plt.title('SVM: minDCF vs actDCF vs C')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
    

def polynomial_kernel(X1, X2, c=1, d=2):
    return (X1.T @ X2 + c) ** d

def rbf_kernel(X1, X2, gamma):
    X1_sq = np.sum(X1**2, axis=0).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=0).reshape(1, -1)
    dist_sq = X1_sq + X2_sq - 2 * (X1.T @ X2)
    return np.exp(-gamma * dist_sq)

# === Dual SVM with Kernel ===

def kernel_with_bias(kernel_func, kernel_params, bias_K):
    def biased_kernel(X1, X2):
        base = kernel_func(X1, X2, **kernel_params)
        return base + bias_K**2  # K squared as ξ
    return biased_kernel

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

# === Score computation ===

def compute_kernel_svm_scores(DTE, alpha, Z, DTR, kernel_func, kernel_params):
    # Only support vectors
    support = alpha > 1e-6
    alpha_sv = alpha[support]
    Z_sv = Z[support]
    DTR_sv = DTR[:, support]

    K = kernel_func(DTR_sv, DTE, **kernel_params)
    scores = np.sum((alpha_sv * Z_sv)[:, None] * K, axis=0)
    return scores

import matplotlib.pyplot as plt
import numpy as np

def plot_rbf_gridsearch_results(C_values, gamma_values, minDCF_results, actDCF_results, pi_T=0.1):

    plt.figure(figsize=(12, 5))

    # ---- minDCF Plot ----
    plt.subplot(1, 2, 1)
    for i, gamma in enumerate(gamma_values):
        plt.plot(C_values, minDCF_results[i], label=f'γ = {gamma:.0e}', marker='o')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel(f'minDCF (πT={pi_T})')
    plt.title('minDCF vs C (RBF Kernel)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # ---- actDCF Plot ----
    plt.subplot(1, 2, 2)
    for i, gamma in enumerate(gamma_values):
        plt.plot(C_values, actDCF_results[i], label=f'γ = {gamma:.0e}', marker='x')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel(f'actDCF (πT={pi_T})')
    plt.title('actDCF vs C (RBF Kernel)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

    
if __name__ == '__main__':

    file_path = 'Traindata.txt'  # Replace with your actual file path
    D, L = load_project_data(file_path)
    
    #10% of the data
    # nTotal = int(D.shape[1] * 0.1)
    # idx = np.random.permutation(D.shape[1])[:nTotal]
    # D = D[:, idx] 
    # L = L[idx]
    
    # for full data
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    # Define the range of C values (regularization parameter)
    C_values = np.logspace(-5, 0, 11)
    K=1
    min_dcf_values = []
    act_DCF_values = []
    # for c in C_values:
    #     w, b, alpha, Jp, Jd, gap = compute_dual_svm(DTR, LTR, c, K)
    
        
    #     print("Primal J(w*, b*):", Jp)
    #     print("Dual Jd(α*):", Jd)
    
    #     # Evaluate on validation data
    #     scores = w.T @ DTE + b
    #     predictions = (scores > 0).astype(int)
    #     accuracy = (predictions == LTE).mean()
    #     print(f"Error: {(1-accuracy)*100:.2f}%")
        
        
    #     pi_emp = (LTR == 1).mean()
    #     log_odds = np.log(pi_emp / (1 - pi_emp))
    #     S_llr = scores - log_odds
        
    #     prior = 0.1
    #     cfn = 1
    #     cpn = 1
    
    #     decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)

    #     cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
    #     # print(cm)

    #     dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
    #     act_DCF_values.append(dcf_norm)
    #     print(f"Actual DCF : {dcf_norm}")
        
    #     #Minimum Detection Cost
    #     min_cost = compute_min_DCF(S_llr, LTE, prior, cfn, cpn, dcf_norm)
    #     min_dcf_values.append(min_cost)
    #     print(f"min DCF: {min_cost}")
    #     print()
        
    # plot_dcf_vs_C(C_values, min_dcf_values, act_DCF_values)
    
    #polynomial kernel
    min_dcf_pol = []
    act_dcf_pol = []
    
    configs = [
        (0.0, 'poly', {'c': 1, 'd': 2}),
    ]
    # for C in C_values:
    #     print(f"C: {C}")
    #     for K, kernel_type, params in configs:
            
    #         base_kernel = polynomial_kernel
    #         kernel_params = {'c': params['c'], 'd': params['d']}
            
        
    #         kernel = kernel_with_bias(base_kernel, kernel_params, bias_K=K)
        
    #         # Train
    #         alpha_opt, Z, DTR_kernel, H = compute_kernel_svm_dual(DTR, LTR, C, kernel_func=kernel, kernel_params={})
    #         primal_loss, dual_loss, gap = compute_objectives(alpha_opt, H, Z, C)
    #         # Predict
    #         scores = compute_kernel_svm_scores(DTE, alpha_opt, Z, DTR_kernel, kernel_func=kernel, kernel_params={})
    #         preds = (scores > 0).astype(int)
    #         acc = np.mean(preds == LTE)
            
    #         pi_emp = (LTR == 1).mean()
    #         log_odds = np.log(pi_emp / (1 - pi_emp))
    #         S_llr = scores - log_odds
            
    #         prior = 0.1
    #         cfn = 1
    #         cpn = 1
            
           
    #         decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)
    
    #         cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
    #         # print(cm)
    #         # Display result
    #         kernel_desc = f"Poly(d={params.get('d')}, c={params.get('c')})" if kernel_type == 'poly' else f"RBF(γ={params.get('gamma')})"
    #         # print(f"K={K}, Kernel={kernel_desc} → Error Rate: {(1-acc) * 100:.2f}%")
    #         print(f"Primla loss: {primal_loss}")
    #         print(f"Dual loss:{dual_loss}")
    #         print(f"Error rate: {(1-acc)*100:.2f}")
            
    #         dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
    #         # act_dcf_pol.append(dcf_norm)
    #         print(f"Actual DCF : {dcf_norm}")
            
    #         #Minimum Detection Cost
    #         min_cost = compute_min_DCF(S_llr, LTE, prior, cfn, cpn, dcf_norm)
    #         # min_dcf_pol.append(min_cost)
    #         print(f"min DCF: {min_cost}")
    #         print()
            
            
    #RBF kernel
    gamma_values = np.exp(-np.array([4, 3, 2, 1]))
    C_values = np.logspace(-3, 2, 11)
    
    for gamma in gamma_values:
        min_dcf_gamma = []
        act_dcf_gamma = []
        
        for C in C_values:
            base_kernel = rbf_kernel
            kernel_params = {'gamma': gamma}
            K=1    
            kernel = kernel_with_bias(base_kernel, kernel_params, bias_K=K)
        
            # Train
            alpha_opt, Z, DTR_kernel, H = compute_kernel_svm_dual(DTR, LTR, C, kernel_func=kernel, kernel_params={})
            primal_loss, dual_loss, gap = compute_objectives(alpha_opt, H, Z, C)
            # Predict
            scores = compute_kernel_svm_scores(DTE, alpha_opt, Z, DTR_kernel, kernel_func=kernel, kernel_params={})
            preds = (scores > 0).astype(int)
            acc = np.mean(preds == LTE)
            
            pi_emp = (LTR == 1).mean()
            log_odds = np.log(pi_emp / (1 - pi_emp))
            S_llr = scores - log_odds
            
            prior = 0.1
            cfn = 1
            cpn = 1
            
           
            decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)
    
            cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)

            
            dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
            act_dcf_gamma.append(dcf_norm)

            
            #Minimum Detection Cost
            min_cost = compute_min_DCF(S_llr, LTE, prior, cfn, cpn, dcf_norm)
            min_dcf_gamma.append(min_cost)
            # print(f"min DCF: {min_cost}")
        min_dcf_pol.append(min_dcf_gamma)
        act_dcf_pol.append(act_dcf_gamma)
        
    plot_rbf_gridsearch_results(C_values, gamma_values, min_dcf_pol, act_dcf_pol)
        

   
   
    