
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

################################################################
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
    
    plot_bnr(lambdas, actual_dcf_list, min_dcf_list, labels)
    
def plot_bnr(lamdas, actual_dcf, min_dcf, label):
    plt.figure(figsize=(8, 5))
    plt.plot(lamdas, actual_dcf, label="Actual DCF", marker='o')
    plt.plot(lamdas, min_dcf, label="Min DCF", marker='x')
    plt.xscale('log')
    plt.xlabel("λ (log scale)")
    plt.ylabel("DCF")
    plt.title(f"Actual vs. Minimum DCF vs λ (πT = 0.1) ({label})")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
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

    
if __name__ == '__main__':
    
    file_path = 'Traindata.txt'  # Replace with your actual file path
    D, L = load_project_data(file_path)
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    lambdas = numpy.logspace(-4, 2, 13)
    labels = "Binary Logistic Regression model"
    evaluate_bnr(DTR, LTR, DTE, LTE, lambdas, labels)
    
    #results with fewwer training set
    DTR_sub = DTR[:, ::50]
    LTR_sub = LTR[::50]
    labels = "fewer samples"
    evaluate_bnr(DTR_sub, LTR_sub, DTE, LTE, lambdas, labels)
    
    #prior weighted version
    actual_dcf_list = []
    min_dcf_list = []
    objectives = []

    for lambda_val in lambdas:
        pi_T = 0.1
        w, b, grad = train_prior_weighted_logreg(DTR, LTR, lambda_val, pi_T)
        acc = compute_accuracy(DTE, LTE, w, b)
        objectives.append(grad)
        
        #score
        S = (w @ DTE + b).ravel()
        s_llr = S - np.log(pi_T / (1 - pi_T))
        prior = 0.1
        cfn = 1
        cpn = 1
        decisions = optimal_bayes_decision(s_llr ,pi_T , cfn, cpn)
        cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
        # print(cm)
        dcf_norm = empirical_bayes_risks(cm, pi_T, cfn, cpn, normalize=True)
        min_cost = compute_min_DCF(s_llr, LTE, pi_T, cfn, cpn, dcf_norm)
        actual_dcf_list.append(dcf_norm)
        min_dcf_list.append(min_cost)
    labels = "prior-weighted"
    plot_bnr(lambdas, actual_dcf_list, min_dcf_list, labels)
    
    #Quadratic logistic regression model
    DTR_quad = quadratic_feature_expansion(DTR)
    DTE_quad = quadratic_feature_expansion(DTE)
    labels = "quadratic logistic regression"
    evaluate_bnr(DTR_quad, LTR, DTE_quad, LTE, lambdas, labels)
    
  