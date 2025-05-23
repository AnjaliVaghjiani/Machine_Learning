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

def evaluate_DCF(score, labels, prior):
    prior = 0.2
    predictions = (score > 0).astype(int)
    accuracy = (predictions == labels).mean()
    
    pi_emp = (labels == 1).mean()
    log_odds = np.log(pi_emp / (1 - pi_emp))
    S_llr = score - log_odds
    
    cfn = 1
    cpn = 1

    decisions = optimal_bayes_decision(S_llr ,prior , cfn, cpn)

    cm = compute_confusion_matrix(decisions, labels, labels.max()+1)
    # print(cm)

    dcf_norm = empirical_bayes_risks(cm, prior, cfn, cpn, normalize=True)
    # print(f"Actual DCF : {dcf_norm}")
    
    #Minimum Detection Cost
    min_cost = compute_min_DCF(S_llr, labels, prior, cfn, cpn, dcf_norm)
    # print(f"min DCF: {min_cost}")
    
    return dcf_norm, min_cost

def bayes_error_plot(scores1, scores2, labels, title='Bayes Error Plot'):
    effective_priors = np.linspace(-3, 3, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effective_priors))
    min_dcf1, act_dcf1 = [], []
    min_dcf2, act_dcf2 = [], []

    for pi in effPriors:
        dcf1, mindcf1 = evaluate_DCF(scores1, labels, pi)
        dcf2, mindcf2 = evaluate_DCF(scores2, labels, pi)
        min_dcf1.append(mindcf1)
        act_dcf1.append(dcf1)
        min_dcf2.append(mindcf2)
        act_dcf2.append(dcf2)

    plt.figure(figsize=(10, 6))
    plt.plot(effective_priors, min_dcf1, label='System 1 minDCF', linestyle='--')
    plt.plot(effective_priors, act_dcf1, label='System 1 actDCF')
    plt.plot(effective_priors, min_dcf2, label='System 2 minDCF', linestyle='--')
    plt.plot(effective_priors, act_dcf2, label='System 2 actDCF')
    plt.xlabel('Prior log-odd')
    plt.ylabel('DCF')
    plt.title(title)
    plt.ylim([0, 0.8])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    score_1 = np.load('scores_1.npy')
    score_2 = np.load('scores_2.npy')
    labels = np.load('labels.npy')
    
    prior = 0.2
    dcf_norm_1, min_cost_1 = evaluate_DCF(score_1, labels, prior)
    dcf_norm_2, min_cost_2 = evaluate_DCF(score_2, labels, prior)
    
    print(f'Score 1: min_cost = {min_cost_1} and act_dcf = {dcf_norm_1}')
    print(f'Score 2: min_cost = {min_cost_2} and act_dcf = {dcf_norm_2}')

    bayes_error_plot(score_1, score_2, labels, title='Bayes Error Plot - System 1')
