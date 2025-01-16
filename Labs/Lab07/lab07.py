import numpy
import scipy.special
import matplotlib
import matplotlib.pyplot

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

# compute matrix of posteriors from class-conditional log-likelihoods (each column represents a sample) and prior array
def compute_posteriors(log_clas_conditional_ll, prior_array):
    logJoint = log_clas_conditional_ll + vcol(numpy.log(prior_array))
    logPost = logJoint - scipy.special.logsumexp(logJoint, 0)
    return numpy.exp(logPost)

# Compute optimal Bayes decisions for the matrix of class posterior (each column refers to a sample)
def compute_optimal_Bayes(posterior, costMatrix):
    expectedCosts = costMatrix @ posterior
    return numpy.argmin(expectedCosts, 0)

# Build uniform cost matrix with cost 1 for all kinds of error, and cost 0 for correct assignments
def uniform_cost_matrix(nClasses):
    return numpy.ones((nClasses, nClasses)) - numpy.eye(nClasses)

# Assume that classes are labeled 0, 1, 2 ... (nClasses - 1)
def compute_confusion_matrix(predictedLabels, classLabels):
    nClasses = classLabels.max() + 1
    M = numpy.zeros((nClasses, nClasses), dtype=numpy.int32)
    for i in range(classLabels.size):
        M[predictedLabels[i], classLabels[i]] += 1
    return M

# Optimal Bayes deicsions for binary tasks with log-likelihood-ratio scores
def compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp):
    th = -numpy.log( (prior * Cfn) / ((1 - prior) * Cfp) )
    return numpy.int32(llr > th)

# Multiclass solution that works also for binary problems
def compute_empirical_Bayes_risk(predictedLabels, classLabels, prior_array, costMatrix, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    errorRates = M / vrow(M.sum(0))
    bayesError = ((errorRates * costMatrix).sum(0) * prior_array.ravel()).sum()
    if normalize:
        return bayesError / numpy.min(costMatrix @ vcol(prior_array))
    return bayesError

# Specialized function for binary problems (empirical_Bayes_risk is also called DCF or actDCF)
def compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=True):
    M = compute_confusion_matrix(predictedLabels, classLabels) # Confusion matrix
    Pfn = M[0,1] / (M[0,1] + M[1,1])
    Pfp = M[1,0] / (M[0,0] + M[1,0])
    bayesError = prior * Cfn * Pfn + (1-prior) * Cfp * Pfp
    if normalize:
        return bayesError / numpy.minimum(prior * Cfn, (1-prior)*Cfp)
    return bayesError

# Compute empirical Bayes (DCF or actDCF) risk from llr with optimal Bayes decisions
def compute_empirical_Bayes_risk_binary_llr_optimal_decisions(llr, classLabels, prior, Cfn, Cfp, normalize=True):
    predictedLabels = compute_optimal_Bayes_binary_llr(llr, prior, Cfn, Cfp)
    return compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp, normalize=normalize)

# Compute all combinations of Pfn, Pfp for all thresholds (sorted)
def compute_Pfn_Pfp_allThresholds_slow(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs

    Pfn = []
    Pfp = []
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llrSorted, numpy.array([numpy.inf])]) #The function returns a slightly different array than the fast version, which does not include -numpy.inf as threshold - see the fast function comment
    for th in thresholds:
        M = compute_confusion_matrix(th, classLabels) # Confusion matrix
        Pfn.append(M[0,1] / (M[0,1] + M[1,1]))
        Pfp.append(M[1,0] / (M[0,0] + M[1,0]))
    return Pfn, Pfp, thresholds
        
    
    
# Compute minDCF (slow version, loop over all thresholds recomputing the costs)
# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_slow(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):
    # llrSorter = numpy.argsort(llr) 
    # llrSorted = llr[llrSorter] # We sort the llrs
    # classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs
    # We can remove this part
    llrSorted = llr # In this function (slow version) sorting is not really necessary, since we re-compute the predictions and confusion matrices everytime
    
    thresholds = numpy.concatenate([numpy.array([-numpy.inf]), llrSorted, numpy.array([numpy.inf])])
    dcfMin = None
    dcfTh = None
    for th in thresholds:
        predictedLabels = numpy.int32(llr > th)
        dcf = compute_empirical_Bayes_risk_binary(predictedLabels, classLabels, prior, Cfn, Cfp)
        if dcfMin is None or dcf < dcfMin:
            dcfMin = dcf
            dcfTh = th
    if returnThreshold:
        return dcfMin, dcfTh
    else:
        return dcfMin

# Compute minDCF (fast version)
# If we sort the scores, then, as we sweep the scores, we can have that at most one prediction changes everytime. We can then keep a running confusion matrix (or simply the number of false positives and false negatives) that is updated everytime we move the threshold

# Auxiliary function, returns all combinations of Pfp, Pfn corresponding to all possible thresholds
# We do not consider -inf as threshld, since we use as assignment llr > th, so the left-most score corresponds to all samples assigned to class 1 already
def compute_Pfn_Pfp_allThresholds_fast(llr, classLabels):
    llrSorter = numpy.argsort(llr)
    llrSorted = llr[llrSorter] # We sort the llrs
    classLabelsSorted = classLabels[llrSorter] # we sort the labels so that they are aligned to the llrs

    Pfp = []
    Pfn = []
    
    nTrue = (classLabelsSorted==1).sum()
    nFalse = (classLabelsSorted==0).sum()
    nFalseNegative = 0 # With the left-most theshold all samples are assigned to class 1
    nFalsePositive = nFalse
    
    Pfn.append(nFalseNegative / nTrue)
    Pfp.append(nFalsePositive / nFalse)
    
    for idx in range(len(llrSorted)):
        if classLabelsSorted[idx] == 1:
            nFalseNegative += 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we increase the error rate
        if classLabelsSorted[idx] == 0:
            nFalsePositive -= 1 # Increasing the threshold we change the assignment for this llr from 1 to 0, so we decrease the error rate
        Pfn.append(nFalseNegative / nTrue)
        Pfp.append(nFalsePositive / nFalse)

    #The last values of Pfn and Pfp should be 1.0 and 0.0, respectively
    #Pfn.append(1.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    #Pfp.append(0.0) # Corresponds to the numpy.inf threshold, all samples are assigned to class 0
    llrSorted = numpy.concatenate([-numpy.array([numpy.inf]), llrSorted])

    # In case of repeated scores, we need to "compact" the Pfn and Pfp arrays (i.e., we need to keep only the value that corresponds to an actual change of the threshold
    PfnOut = []
    PfpOut = []
    thresholdsOut = []
    for idx in range(len(llrSorted)):
        if idx == len(llrSorted) - 1 or llrSorted[idx+1] != llrSorted[idx]: # We are indeed changing the threshold, or we have reached the end of the array of sorted scores
            PfnOut.append(Pfn[idx])
            PfpOut.append(Pfp[idx])
            thresholdsOut.append(llrSorted[idx])
            
    return numpy.array(PfnOut), numpy.array(PfpOut), numpy.array(thresholdsOut) # we return also the corresponding thresholds
    
# Note: for minDCF llrs can be arbitrary scores, since we are optimizing the threshold
# We can therefore directly pass the logistic regression scores, or the SVM scores
def compute_minDCF_binary_fast(llr, classLabels, prior, Cfn, Cfp, returnThreshold=False):

    Pfn, Pfp, th = compute_Pfn_Pfp_allThresholds_fast(llr, classLabels)
    minDCF = (prior * Cfn * Pfn + (1 - prior) * Cfp * Pfp) / numpy.minimum(prior * Cfn, (1-prior)*Cfp) # We exploit broadcasting to compute all DCFs for all thresholds
    idx = numpy.argmin(minDCF)
    if returnThreshold:
        return minDCF[idx], th[idx]
    else:
        return minDCF[idx]

compute_actDCF_binary_fast = compute_empirical_Bayes_risk_binary_llr_optimal_decisions # To have a function with a similar name to the minDCF one

    
if __name__ == '__main__':

    # Initial multiclass task

    print()
    print("Multiclass - uniform priors and costs - confusion matrix")
    commedia_ll = numpy.load('commedia_ll.npy')
    commedia_labels = numpy.load('commedia_labels.npy')

    commedia_posteriors = compute_posteriors(commedia_ll, numpy.ones(3)/3.0)
    commedia_predictions = compute_optimal_Bayes(commedia_posteriors, uniform_cost_matrix(3))

    print(compute_confusion_matrix(commedia_predictions, commedia_labels))

    

    # Binary task
    print()
    print("-"*40)
    print()
    print("Binary task")
    commedia_llr_binary = numpy.load('commedia_llr_infpar.npy')
    commedia_labels_binary = numpy.load('commedia_labels_infpar.npy')

    for prior, Cfn, Cfp in [(0.5, 1, 1), (0.8, 1, 1), (0.5, 10, 1), (0.8, 1, 10)]:
        print()
        print('Prior', prior, '- Cfn', Cfn, '- Cfp', Cfp)
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, prior, Cfn, Cfp)
        print(compute_confusion_matrix(commedia_predictions_binary, commedia_labels_binary))
        print('DCF (non-normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
            commedia_predictions_binary, commedia_labels_binary, prior, Cfn, Cfp, normalize=False)))
        print('DCF (non-normalized, multiclass code): %.3f' % (compute_empirical_Bayes_risk(
            commedia_predictions_binary,
            commedia_labels_binary,
            numpy.array([1-prior, prior]), # Class 1 is the second element for multiclass
            numpy.array([[0, Cfn], [Cfp, 0]]),
            normalize=False)))
        print('DCF (normalized): %.3f' % (compute_empirical_Bayes_risk_binary(
            commedia_predictions_binary, commedia_labels_binary, prior, Cfn, Cfp)))
        print('DCF (normalized, multiclass code): %.3f' % (compute_empirical_Bayes_risk(
            commedia_predictions_binary,
            commedia_labels_binary,
            numpy.array([1-prior, prior]), # Class 1 is the second element for multiclass
            numpy.array([[0, Cfn], [Cfp, 0]]))))
        minDCF, minDCFThreshold = compute_minDCF_binary_slow(commedia_llr_binary, commedia_labels_binary, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, slow): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))
        minDCF, minDCFThreshold = compute_minDCF_binary_fast(commedia_llr_binary, commedia_labels_binary, prior, Cfn, Cfp, returnThreshold=True)
        print('MinDCF (normalized, fast): %.3f (@ th = %e)' % (minDCF, minDCFThreshold))

    # ROC plot - uncomment the commented lines to see the plot
    Pfn, Pfp, _ = compute_Pfn_Pfp_allThresholds_fast(commedia_llr_binary, commedia_labels_binary)
    #matplotlib.pyplot.figure(0)
    #matplotlib.pyplot.plot(Pfp, 1-Pfn)
    #matplotlib.pyplot.show()

    # Bayes error plot
    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    effPriors = 1.0 / (1.0 + numpy.exp(-effPriorLogOdds))
    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, commedia_labels_binary, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0))
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF eps 0.001', color='r')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='DCF eps 0.001', color='b')
    matplotlib.pyplot.ylim([0, 1.1])
    #matplotlib.pyplot.show()
            
    commedia_llr_binary = numpy.load('commedia_llr_infpar_eps1.npy')
    commedia_labels_binary = numpy.load('commedia_labels_infpar_eps1.npy')

    actDCF = []
    minDCF = []
    for effPrior in effPriors:
        # Alternatively, we can compute actDCF directly from compute_empirical_Bayes_risk_binary_llr_optimal_decisions(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0)
        commedia_predictions_binary = compute_optimal_Bayes_binary_llr(commedia_llr_binary, effPrior, 1.0, 1.0)
        actDCF.append(compute_empirical_Bayes_risk_binary(commedia_predictions_binary, commedia_labels_binary, effPrior, 1.0, 1.0))
        minDCF.append(compute_minDCF_binary_fast(commedia_llr_binary, commedia_labels_binary, effPrior, 1.0, 1.0))

    matplotlib.pyplot.plot(effPriorLogOdds, actDCF, label='actDCF eps 1.0', color='y')
    matplotlib.pyplot.plot(effPriorLogOdds, minDCF, label='DCF eps 1.0', color='c')
    matplotlib.pyplot.ylim([0, 1.1])

    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


    
    # Multiclass task
    print()
    print("-"*40)
    print()
    print("Multiclass task")

    prior = numpy.array([0.3, 0.4, 0.3])
    #prior = numpy.ones(3)/3.0 # For the uniform cost solution
    costMatrix = numpy.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
    #costMatrix = uniform_cost_matrix(3) # For the uniform cost solution

    print()
    print('Eps 0.001')
    commedia_ll = numpy.load('commedia_ll.npy')
    commedia_labels = numpy.load('commedia_labels.npy')

    commedia_posteriors = compute_posteriors(commedia_ll, prior)
    commedia_predictions = compute_optimal_Bayes(commedia_posteriors, costMatrix)
    print(compute_confusion_matrix(commedia_predictions, commedia_labels))
    print('Emprical Bayes risk: %.3f' % compute_empirical_Bayes_risk(
        commedia_predictions, commedia_labels, prior, costMatrix, normalize=False))
    print('Normalized emprical Bayes risk: %.3f' % compute_empirical_Bayes_risk(
        commedia_predictions, commedia_labels, prior, costMatrix, normalize=True))

    print()
    print('Eps 1.0')
    commedia_ll = numpy.load('commedia_ll_eps1.npy')
    commedia_labels = numpy.load('commedia_labels_eps1.npy')
    

    commedia_posteriors = compute_posteriors(commedia_ll, prior)
    commedia_predictions = compute_optimal_Bayes(commedia_posteriors, costMatrix)
    print(compute_confusion_matrix(commedia_predictions, commedia_labels))
    print('Emprical Bayes risk: %.3f' % compute_empirical_Bayes_risk(
        commedia_predictions, commedia_labels, prior, costMatrix, normalize=False))
    print('Normalized emprical Bayes risk: %.3f' % compute_empirical_Bayes_risk(
        commedia_predictions, commedia_labels, prior, costMatrix, normalize=True))
    print()