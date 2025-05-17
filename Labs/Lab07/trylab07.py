# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:31:44 2025

@author: anjal
"""

import numpy as np
import scipy.special
import matplotlib
import matplotlib.pyplot as plt

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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
    threshold = -np.log((prior * cfn) / ((1-prior) * cfp))
  
    return np.int32(LLR > threshold)  

def empirical_bayes_risks(cf, prior, cfn, cpn, normalize=True):
    pfn = cf[0,1] / (cf[0,1] + cf[1,1])
    ppn = cf[1,0] / (cf[0,0] + cf[1,0])
    DCF = prior * cfn * pfn + (1-prior)*cpn*ppn
    if normalize:
        return DCF / np.minimum(prior * cfn, (1-prior)*cpn)
    return DCF



def compute_optimal_bayes_decision(posterior, cost_matrix):
    expected_costs = cost_matrix @ posterior  # Shape: (num_classes, num_samples)
    return np.argmin(expected_costs, axis=0)  # Choose class with minimum cost
    
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

def compute_confusion_matrix_binary(predictions, labels):
    # True positives, false negatives, etc.
    TP = np.sum((predictions == 1) & (labels == 1))
    FP = np.sum((predictions == 1) & (labels == 0))
    FN = np.sum((predictions == 0) & (labels == 1))
    TN = np.sum((predictions == 0) & (labels == 0))
    return TP, FP, FN, TN

def compute_roc(llr, labels):
    thresholds = np.concatenate([[-np.inf], np.sort(llr), [np.inf]])
    tpr = []
    fpr = []
    
    for t in thresholds:
        predictions = (llr > t).astype(int)
        TP, FP, FN, TN = compute_confusion_matrix_binary(predictions, labels)

        # Avoid divide-by-zero
        P = TP + FN
        N = FP + TN

        Ptp = TP / P if P > 0 else 0
        Pfp = FP / N if N > 0 else 0

        tpr.append(Ptp)
        fpr.append(Pfp)

    return np.array(fpr), np.array(tpr)

def compute_bayes_plot(llr, labels):
    effPriorLogOdds = np.linspace(-3, 3, 21)
    dcf = []
    min_dcf = []
    
    for p in effPriorLogOdds:
        eff_prior = 1 / (1 + np.exp(-p))
        #DCF using actual threshold
        decisions = optimal_bayes_decision(llr, eff_prior, 1, 1)
        cm = compute_confusion_matrix(decisions, labels, 2)
        dcf_val = empirical_bayes_risks(cm, eff_prior, 1, 1, normalize=True)
        dcf.append(dcf_val)
        
        #Min DCF
        min_dcf_val = compute_min_DCF(llr, labels, eff_prior, 1, 1, dcf_norm=True)
        min_dcf.append(min_dcf_val)
        
    
    return effPriorLogOdds, np.array(dcf), np.array(min_dcf)
 
#multiclass evaluation
def softmax(ll, priors):
    log_prior = np.log(priors).reshape(-1, 1)
    log_post = ll + log_prior
    log_post -= np.max(log_post, axis=0)  # for numerical stability
    exp_log_post = np.exp(log_post)
    return exp_log_post / np.sum(exp_log_post, axis=0)

# Multiclass solution that works also for binary problems
def compute_empirical_Bayes_risk(predictedLabels, classLabels, prior_array, costMatrix, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels, classLabels.max() + 1) # Confusion matrix
    errorRates = M / vrow(M.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior_array.ravel()).sum()
    if normalize:
        return bayesError / np.min(costMatrix @ vcol(prior_array))
    return bayesError

def multiclass_DCF(ll, labels, C, priors,  normalize=True,):
    # Posterior probabilities (3 x N)
    Pb = compute_class_posterior(ll, priors)
    Pb = np.exp(Pb)
    # prediction = Pb.argmax(0) 
    prediction = compute_optimal_bayes_decision(Pb, C)
    
    # Confusion matrix
    M = compute_confusion_matrix(prediction, labels, labels.max() + 1)
    if normalize:
        DCF_norm = compute_empirical_Bayes_risk(prediction, labels, priors, C, normalize=True)
        print('Emprical Bayes risk normalized: %.3f', DCF_norm)
    else:
        DCF_unnorm = compute_empirical_Bayes_risk(prediction, labels, priors, C, normalize=False)
        print('Emprical Bayes risk unnormalized: %.3f', DCF_unnorm)
    return M
       
if __name__ == '__main__':
    print()
    print("Multiclass - uniform priors and costs - confusion matrix")
    commedia_ll = np.load('commedia_ll.npy')
    commedia_labels = np.load('commedia_labels.npy')

    commedia_posteriors = compute_class_posterior(commedia_ll, 1/3)
    commedia_predictions = commedia_posteriors.argmax(0) 

    print(compute_confusion_matrix(commedia_predictions, commedia_labels, 3))
    
    #Optimal Bayes Decision
    #Binary decision
    print('Binary Tasks')
    llr = np.load('commedia_llr_infpar.npy')
    labels = np.load('commedia_labels_infpar.npy')
    for prior, cfn, cpn in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        decisions = optimal_bayes_decision(llr ,prior , cfn, cpn)
        print()
        print('Prior', prior, '- Cfn', cfn, '- Cfp', cpn)
        cm = compute_confusion_matrix(decisions, labels, labels.max()+1)
        print(cm)
        
        print()
        DCF = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=False)
        print(f"Unnormalized DCF values for Prior: {prior}, cfn: {cfn}, cpn: {cpn} : {DCF}\n")
        
        print()
        dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
        print(f"normalized DCF values for Prior: {prior}, cfn: {cfn}, cpn: {cpn} : {dcf_norm}\n")
        
        #Minimum Detection Cost
        print()
        min_cost = compute_min_DCF(llr, labels, prior, cfn, cpn, dcf_norm)
        print(f"min DCF for : {prior}, cfn: {cfn}, cpn: {cpn} : {min_cost}\n")     
    
    #ROC
    fpr, tpr = compute_roc(llr, labels)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #Bayes error plot
    p_tilde_range, dcf, min_dcf = compute_bayes_plot(llr, labels)

    plt.figure(figsize=(8, 6))
    plt.plot(p_tilde_range, dcf, label='DCF', color='r')
    plt.plot(p_tilde_range, min_dcf, label='min DCF', color='b')
    plt.xlabel('Prior log-odds (p̃)')
    plt.ylabel('Normalized DCF')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.title('Bayes Error Plot')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # #comparing recognizers
    llr_eps0001 = llr
    llr_eps1 = np.load("commedia_llr_infpar_eps1.npy")       # ε = 1
    for prior, cfn, cpn in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        decisions = optimal_bayes_decision(llr_eps1 ,prior , cfn, cpn)
        cm = compute_confusion_matrix(decisions, labels, labels.max()+1)
        print()
        dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
        print(f"for e =1 normalized DCF values for Prior: {prior}, cfn: {cfn}, cpn: {cpn} : {dcf_norm}\n")
        
        #Minimum Detection Cost
        print()
        min_cost = compute_min_DCF(llr_eps1, labels, prior, cfn, cpn, dcf_norm)
        print(f"for e=1 min DCF for : {prior}, cfn: {cfn}, cpn: {cpn} : {min_cost}\n")
        
    log_odds, dcf_eps0001, min_dcf_eps0001 = compute_bayes_plot(llr_eps0001, labels)
    _, dcf_eps1, min_dcf_eps1 = compute_bayes_plot(llr_eps1, labels)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(log_odds, dcf_eps0001, label='DCF ε=0.001', color='r', linestyle='--')
    plt.plot(log_odds, min_dcf_eps0001, label='minDCF ε=0.001', color='r')
    
    plt.plot(log_odds, dcf_eps1, label='DCF ε=1', color='b', linestyle='--')
    plt.plot(log_odds, min_dcf_eps1, label='minDCF ε=1', color='b')
    
    plt.xlabel('Prior Log-Odds')
    plt.ylabel('Normalized DCF')
    plt.title('Bayes Error Plot: Multinomial Models (ε=0.001 vs ε=1)')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #multiclass evaluation
    # Define cost matrix and priors
    commedia_ll = np.load('commedia_ll.npy')
    commedia_labels = np.load('commedia_labels.npy')
    C_custom = np.array([[0, 1, 2],
                         [1, 0, 1],
                         [2, 1, 0]])
    
    priors_custom = np.array([0.3, 0.4, 0.3])
    
    C_uniform = np.ones((3, 3)) - np.eye(3)
    priors_uniform = np.array([1/3, 1/3, 1/3])
    
    # Evaluate ε=0.001
    M1 = multiclass_DCF(commedia_ll, commedia_labels, C_custom, priors_custom, normalize=False)
    print("ε=0.001, custom cost/prior")
    print("Confusion matrix:\n", M1)
    multiclass_DCF(commedia_ll, commedia_labels, C_custom, priors_custom, normalize=True)

    commedia_ll = np.load('commedia_ll_eps1.npy')
    commedia_labels = np.load('commedia_labels_eps1.npy')
    # Evaluate ε=1.0
    M2 = multiclass_DCF(commedia_ll, commedia_labels, C_custom, priors_custom, normalize=False)
    multiclass_DCF(commedia_ll, commedia_labels, C_custom, priors_custom, normalize=True)
    print("ε=0.001, custom cost/prior")
    print("Confusion matrix:\n", M2)
    

    
    # # Evaluate uniform case
    print()
    print("uniform class")
    commedia_ll = np.load('commedia_ll.npy')
    commedia_labels = np.load('commedia_labels.npy')
    print("ε=0.001, custom cost/prior")
    M1 = multiclass_DCF(commedia_ll, commedia_labels, C_uniform, priors_uniform, normalize=False)
    multiclass_DCF(commedia_ll, commedia_labels, C_uniform, priors_uniform, normalize=True)

    # Evaluate ε=1.0
    commedia_ll = np.load('commedia_ll_eps1.npy')
    commedia_labels = np.load('commedia_labels_eps1.npy')
    print("ε=0.001, custom cost/prior")
    M2 = multiclass_DCF(commedia_ll, commedia_labels, C_uniform, priors_uniform, normalize=False)
    multiclass_DCF(commedia_ll, commedia_labels, C_uniform, priors_uniform, normalize=True)
