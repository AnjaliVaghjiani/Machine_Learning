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


def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5*x.shape[0]*np.log(np.pi*2) - 0.5*np.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)

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
    
##################################################################
def ML_estimares_class(LTR, DTE):
    data = {}
    for i in set(LTR):
       DX = DTE[:, LTR==i]
       mu, C = compute_mu_C(DX)
       data[i] = compute_mu_C(DX)
       # print(mu,C) 
    return data 

def loglikelihood(DTE, data):
    S = np.zeros((len(data), DTE.shape[1]))
    for i in range(S.shape[0]):
        #data[i][0] is mean and data[i][1] is covaraince
        S[i, :] = logpdf_GAU_ND(DTE, data[i][0], data[i][1])
    return S

#computation of class posteriror probability by combining the score matrix with priory information
def compute_class_posterior(S, value):
    SJoint = S + vcol(np.log(value))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost

def compute_confusion_matrix(L_pred, L_true, num_classes):
    CM = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(L_true, L_pred):
        CM[pred_label, true_label] += 1  # Rows: predicted, Columns: actual
    return CM

 #cfn is false negative and cfp is false positive
def optimal_bayes_decision(LLR, prior, cfn, cfp):
    threshold = -np.log(prior * cfn / ((1-prior) * cfp))
  
    return (LLR > threshold).astype(int)

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

def tied_ML_estimation(LTR, DTE):
    data={}
    CGlobal = 0
    m = {}
    for i in set(LTR):
        DX = DTE[:, LTR==i]
        mu, C = compute_mu_C(DX)
        CGlobal += C * DX.shape[1]
        m[i] = mu
    CGlobal = CGlobal/DTE.shape[1]
    for i in set(LTR):
        data[i] = (m[i], CGlobal)
    return data

 
def naive_ML_estimation(LTR, DTE):
    data ={}
    for i in set(LTR):
        DX = DTE[:, LTR==i]
        mu, C = compute_mu_C(DX)
        data[i] = (mu, C * np.eye(DTE.shape[0]))
        
    return data 

def evaluate_model(applications, llr, LTE, labels):
    for prior, cfn, cpn in applications:
        eff_prior = compute_effective_prior(prior, cfn, cpn)
        decisions = optimal_bayes_decision(llr ,prior , cfn, cpn)
        print()
        print('Prior', prior, '- Cfn', cfn, '- Cfp', cpn)
        cm = compute_confusion_matrix(decisions, LTE, LTE.max()+1)
        print(cm)
        
        print()
        DCF = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=False)
        print(f"Unnormalized DCF values for Prior: {prior}, cfn: {cfn}, cpn: {cpn} : {DCF}\n")
        
        print()
        dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
        print(f"normalized DCF values for Prior: {prior}, cfn: {cfn}, cpn: {cpn} : {dcf_norm}\n")
        
        #Minimum Detection Cost
        print()
        min_cost = compute_min_DCF(llr, LTE, prior, cfn, cpn, dcf_norm)
        print(f"min DCF for : {prior}, cfn: {cfn}, cpn: {cpn} : {min_cost}\n")     
        
        
def compute_effective_prior(prior, cfn, cfp):
    return prior * cfn / (prior * cfn + (1 - prior) * cfp)

def bayes_error_plot(llrs, labels, cfn=1, cfp=1, model_name="Model"):
    prior_log_odds = np.linspace(-4, 4, 25)
    DCFs = []
    minDCFs = []

    for ptilde in prior_log_odds:
        prior = 1 / (1 + np.exp(-ptilde))
        eff_prior = compute_effective_prior(prior, cfn, cfp)

        # Actual DCF
        threshold = -np.log((eff_prior * cfn) / ((1 - eff_prior) * cfp))
        predictions = (llrs > threshold).astype(int)
        cm = compute_confusion_matrix(predictions, labels, 2)
        dcf = empirical_bayes_risks(cm, eff_prior, cfn, cfp, normalize=True)
        DCFs.append(dcf)

        # Minimum DCF
        min_dcf = compute_min_DCF(llrs, labels, eff_prior, cfn, cfp, dcf)
        minDCFs.append(min_dcf)

    plt.plot(prior_log_odds, DCFs, label=f"{model_name} - DCF", color="r")
    plt.plot(prior_log_odds, minDCFs, label=f"{model_name} - minDCF", color="b")
    plt.xlabel("Prior Log-Odds")
    plt.ylabel("DCF")
    plt.ylim([0, 1.1])
    plt.title(f"Bayes Error Plot - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()
        
def main():
    
    file_path = 'Traindata.txt'  # Replace with your actual file path
    llr, labels = load_project_data(file_path)
    (DTR, LTR), (DTE, LTE) = split_db_2tot(llr, labels)
    applications = [
        (0.5, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.1, 1.0, 1.0),
        (0.5, 1.0, 9.0),
        (0.5, 9.0, 1.0)
    ]
    
    
    data_MVG = ML_estimares_class(LTR, DTR);
    S_logLikelihood = loglikelihood(DTE, data_MVG);
    S_logPost = compute_class_posterior(S_logLikelihood, 1/2)
    llr_MVG = S_logPost[1, :] - S_logPost[0, :]  # Assuming shape (2, N)

    print("MVG classifire")
    # # print(compute_confusion_matrix(commedia_predictions, labels, labels.max()+1))
    # evaluate_model(applications, llr_MVG, LTE, labels)
    bayes_error_plot(llr_MVG, LTE, model_name="MVG")
    print()
    print("Tied Gaussian")
    data_tied = tied_ML_estimation(LTR, DTR)
    S_logLikelihood = loglikelihood(DTE, data_tied)
    S_logPost = compute_class_posterior(S_logLikelihood, 1/2)
    llr_tied = S_logPost[1, :] - S_logPost[0, :]  # Assuming shape (2, N)
    # evaluate_model(applications, llr_tied, LTE, labels)
    bayes_error_plot(llr_tied, LTE, model_name="Tied")
    print()
    print("Naive bayes classifier")
    
    #Naive Bayes classifier
    data_naive = naive_ML_estimation(LTR, DTR)
    S_logLikelihood = loglikelihood(DTE, data_naive)
    S_logPost = compute_class_posterior(S_logLikelihood, 1/2)
    llr_naive = S_logPost[1, :] - S_logPost[0, :]
    # evaluate_model(applications, llr_naive, LTE, labels)
    bayes_error_plot(llr_naive, LTE, model_name="Naive Bayes")
    

   

if __name__ == '__main__':
    main()
