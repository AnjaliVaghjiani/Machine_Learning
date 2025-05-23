import numpy as np
import scipy.special
import sklearn.datasets

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

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']    
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

###############################################################
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

#Kernal SVM
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

if __name__ == '__main__':

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    
    C = [0.1, 1.0, 10]
    for k in [1, 10]:
        for c in C:
            w, b, alpha, Jp, Jd, gap = compute_dual_svm(DTR, LTR, c, k)
        
            
            print("Primal J(w*, b*):", Jp)
            print("Dual Jd(α*):", Jd)
        
            # Evaluate on validation data
            scores = w.T @ DTE + b
            predictions = (scores > 0).astype(int)
            accuracy = (predictions == LTE).mean()
            print(f"Error: {(1-accuracy)*100:.2f}%")
            print()
            
            pi_emp = (LTR == 1).mean()
            log_odds = np.log(pi_emp / (1 - pi_emp))
            S_llr = scores - log_odds
            
            prior = 0.5
            cfn = 1
            cpn = 1
        
            decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)
    
            cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
            # print(cm)
    
            dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
            print(f"Actual DCF : {dcf_norm}")
            
            #Minimum Detection Cost
            min_cost = compute_min_DCF(S_llr, LTE, prior, cfn, cpn, dcf_norm)
            print(f"min DCF: {min_cost}")
            
    #Kernal SVM
    # Assume DTR, LTR, DTE, LTE are available

    # configs = [
    #     (0.0, 'poly', {'c': 0, 'd': 2}),
    #     (1.0, 'poly', {'c': 0, 'd': 2}),
    #     (0.0, 'poly', {'c': 1, 'd': 2}),
    #     (1.0, 'poly', {'c': 1, 'd': 2}),
    #     (0.0, 'rbf', {'gamma': 1.0}),
    #     (1.0, 'rbf', {'gamma': 1.0}),
    #     (0.0, 'rbf', {'gamma': 10.0}),
    #     (1.0, 'rbf', {'gamma': 10.0}),
    # ]
    
    # for K, kernel_type, params in configs:
    #     if kernel_type == 'poly':
    #         base_kernel = polynomial_kernel
    #         kernel_params = {'c': params['c'], 'd': params['d']}
    #     elif kernel_type == 'rbf':
    #         base_kernel = rbf_kernel
    #         kernel_params = {'gamma': params['gamma']}
    #     else:
    #         raise ValueError("Unknown kernel type")
    
    #     kernel = kernel_with_bias(base_kernel, kernel_params, bias_K=K)
    
    #     # Train
    #     alpha_opt, Z, DTR_kernel, H = compute_kernel_svm_dual(DTR, LTR, C=1.0, kernel_func=kernel, kernel_params={})
    #     primal_loss, dual_loss, gap = compute_objectives(alpha_opt, H, Z, C=1.0)
    #     # Predict
    #     scores = compute_kernel_svm_scores(DTE, alpha_opt, Z, DTR_kernel, kernel_func=kernel, kernel_params={})
    #     preds = (scores > 0).astype(int)
    #     acc = np.mean(preds == LTE)
        
    #     pi_emp = (LTR == 1).mean()
    #     log_odds = np.log(pi_emp / (1 - pi_emp))
    #     S_llr = scores - log_odds
        
    #     prior = 0.5
    #     cfn = 1
    #     cpn = 1
        
       
    #     decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)

    #     cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
    #     # print(cm)
    #     # Display result
    #     kernel_desc = f"Poly(d={params.get('d')}, c={params.get('c')})" if kernel_type == 'poly' else f"RBF(γ={params.get('gamma')})"
    #     print(f"K={K}, Kernel={kernel_desc} → Error Rate: {(1-acc) * 100:.2f}%")
    #     print(f"Primla loss: {primal_loss}")
    #     print(f"Dual loss:{dual_loss}")
       
        
    #     dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
    #     print(f"Actual DCF : {dcf_norm}")
        
    #     #Minimum Detection Cost
    #     min_cost = compute_min_DCF(S_llr, LTE, prior, cfn, cpn, dcf_norm)
    #     print(f"min DCF: {min_cost}")
    #     print()
            
        
            
        
    