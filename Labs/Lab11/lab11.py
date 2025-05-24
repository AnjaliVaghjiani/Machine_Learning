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

    plt.figure()
    plt.plot(effective_priors, min_dcf1, label='System 1 minDCF', linestyle='--')
    plt.plot(effective_priors, act_dcf1, label='System 1 actDCF')
    plt.plot(effective_priors, min_dcf2, label='System 2 minDCF', linestyle='--')
    plt.plot(effective_priors, act_dcf2, label='System 2 actDCF')
    plt.xlabel('Prior log-odd')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
    plt.show()

def bayes_plot(score, labels, title='Bayes plot'):
    effective_priors = np.linspace(-3, 3, 21)
    effPriors = 1.0 / (1.0 + np.exp(-effective_priors))
    min_dcf = []
    act_dcf = []
    
    for pi in effPriors:
        dcf1, mindcf1 = evaluate_DCF(score, labels, pi)
        min_dcf.append(mindcf1)
        act_dcf.append(dcf1)
    
    plt.figure()
    plt.plot(effective_priors, min_dcf, label='minDCF', linestyle='--')
    plt.plot(effective_priors, act_dcf, label='actDCF')
    plt.xlabel('Prior log-odd')
    plt.ylabel('DCF')
    plt.title(title)
    plt.legend()
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
    
if __name__ == '__main__':
    
    score_1 = np.load('scores_1.npy')
    score_2 = np.load('scores_2.npy')
    labels = np.load('labels.npy')
    
    prior = 0.2
    # dcf_norm_1, min_cost_1 = evaluate_DCF(score_1, labels, prior)
    # dcf_norm_2, min_cost_2 = evaluate_DCF(score_2, labels, prior)
    
    # print(f'Score 1: min_cost = {min_cost_1} and act_dcf = {dcf_norm_1}')
    # print(f'Score 2: min_cost = {min_cost_2} and act_dcf = {dcf_norm_2}')

    # bayes_error_plot(score_1, score_2, labels, title='Bayes Error Plot - Full DataSet')
    
    #calibration- Single fold approch
    SCAL1, SVAL1 = score_1[::3], np.hstack([score_1[1::3], score_1[2::3]])
    SCAL2, SVAL2 = score_2[::3], np.hstack([score_2[1::3], score_2[2::3]])
    LCAL, LVAL = labels[::3], np.hstack([labels[1::3], labels[2::3]])
    
    # dcf_norm_1, min_cost_1 = evaluate_DCF(SVAL1, LVAL, prior)
    # dcf_norm_2, min_cost_2 = evaluate_DCF(SVAL2, LVAL, prior)
    
    # print('Raw score')
    # print(f'Score 1: min_cost = {min_cost_1} and act_dcf = {dcf_norm_1}')
    # print(f'Score 2: min_cost = {min_cost_2} and act_dcf = {dcf_norm_2}')

    # bayes_plot(SVAL1 ,LVAL, title='System 1 - Single fold approch')
    # bayes_plot(SVAL2 ,LVAL, title='System 1 - Single fold approch')
    
    # w, b, loss = train_prior_weighted_logreg(vrow(SCAL1), LCAL, 0, prior)
    # calibrated_SVAL1 = (w.T @ vrow(SVAL1) + b - np.log(prior / (1-prior))).ravel()
    
    # w, b, loss = train_prior_weighted_logreg(vrow(SCAL2), LCAL, 0, prior)
    # calibrated_SVAL2 = (w.T @ vrow(SVAL2) + b - np.log(prior / (1-prior))).ravel()
    
    # dcf_norm_1, min_cost_1 = evaluate_DCF(calibrated_SVAL1, LVAL, prior)
    # dcf_norm_2, min_cost_2 = evaluate_DCF(calibrated_SVAL2, LVAL, prior)
    
    # print('Calibrated training')
    # print(f'Score 1: min_cost = {min_cost_1} and act_dcf = {dcf_norm_1}')
    # print(f'Score 2: min_cost = {min_cost_2} and act_dcf = {dcf_norm_2}')

    # bayes_plot(calibrated_SVAL1 ,LVAL, title='System 1 - Single fold approch')
    # bayes_plot(calibrated_SVAL2 ,LVAL, title='System 2 - Single fold approch')
    
    
    eval_score1 = np.load('eval_scores_1.npy')
    eval_score2 = np.load('eval_scores_2.npy')
    eval_labels = np.load('eval_labels.npy')
    
    # dcf_norm_1, min_cost_1 = evaluate_DCF(eval_score1, eval_labels, prior)
    # dcf_norm_2, min_cost_2 = evaluate_DCF(eval_score2, eval_labels, prior)
    
    # print('Evaluation Raw score')
    # print(f'Score 1: min_cost = {min_cost_1} and act_dcf = {dcf_norm_1}')
    # print(f'Score 2: min_cost = {min_cost_2} and act_dcf = {dcf_norm_2}')

    # bayes_plot(eval_score1 ,eval_labels, title='System 1 - Single fold approch')
    # bayes_plot(eval_score2 ,eval_labels, title='System 2 - Single fold approch')
    
    print('Calibrated evaluation score')
    w, b, loss = train_prior_weighted_logreg(vrow(eval_score1), eval_labels, 0, prior)
    calibrated_EVAL1 = (w.T @ vrow(eval_score1) + b - np.log(prior / (1-prior))).ravel()
    
    w, b, loss = train_prior_weighted_logreg(vrow(eval_score2), eval_labels, 0, prior)
    calibrated_EVAL2 = (w.T @ vrow(eval_score2) + b - np.log(prior / (1-prior))).ravel()
    
    dcf_norm_1, min_cost_1 = evaluate_DCF(calibrated_EVAL1, eval_labels, prior)
    dcf_norm_2, min_cost_2 = evaluate_DCF(calibrated_EVAL2, eval_labels, prior)
    
    print('Calibrated training')
    print(f'Score 1: min_cost = {min_cost_1} and act_dcf = {dcf_norm_1}')
    print(f'Score 2: min_cost = {min_cost_2} and act_dcf = {dcf_norm_2}')

    bayes_plot(calibrated_EVAL1 ,eval_labels, title='System 1 - Single fold approch')
    bayes_plot(calibrated_EVAL2 ,eval_labels, title='System 2 - Single fold approch')
    
    
    