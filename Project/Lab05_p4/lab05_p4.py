import numpy as np
import scipy.special

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D - mu) @ (D - mu).T) / D.shape[1]
    return mu, C

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    const = -0.5 * x.shape[0] * np.log(2 * np.pi)
    const -= 0.5 * np.linalg.slogdet(C)[1]
    diff = x - mu
    exponent = -0.5 * np.sum(diff * (P @ diff), axis=0)
    return const + exponent

def load_project_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    D = data[:, :-1].T
    L = data[:, -1].astype(int)
    return D, L

def Gau_MVG_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        hParams[lab] = compute_mu_C(DX)
    return hParams

def Gau_Naive_ML_estimates(D, L):
    labelSet = set(L)
    hParams = {}
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C = compute_mu_C(DX)
        hParams[lab] = (mu, C * np.eye(D.shape[0]))
    return hParams

def Gau_Tied_ML_estimates(D, L):
    labelSet = set(L)
    hMeans = {}
    CGlobal = np.zeros((D.shape[0], D.shape[0]))
    for lab in labelSet:
        DX = D[:, L == lab]
        mu, C = compute_mu_C(DX)
        hMeans[lab] = mu
        CGlobal += C * DX.shape[1]
    CGlobal /= D.shape[1]
    hParams = {lab: (hMeans[lab], CGlobal) for lab in labelSet}
    return hParams

def compute_log_likelihood_Gau(D, hParams):
    labelList = sorted(hParams.keys())
    S = np.zeros((len(labelList), D.shape[1]))
    for i, lab in enumerate(labelList):
        mu, C = hParams[lab]
        S[i, :] = logpdf_GAU_ND(D, mu, C)
    return S, labelList

def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    SMarginal = scipy.special.logsumexp(SJoint, axis=0)
    SPost = SJoint - SMarginal
    return SPost

def compute_error_rate(PRED, LTRUE):
    return (PRED != LTRUE).sum() / LTRUE.size * 100

def compute_pca(D, m):
    mu = np.mean(D, axis=1, keepdims=True)
    DC = D - mu
    U, s, Vh = np.linalg.svd(DC, full_matrices=False)
    P = U[:, :m]
    return mu, P

def apply_pca(D, mu, P):
    DC = D - mu
    DP = P.T @ DC
    return DP
def evaluate_model(DTR, LTR, DVAL, LVAL, model_estimator):
    """
    Train the model using the provided estimator and evaluate its performance on the validation set.
    Returns the error rate.
    """
    hParams = model_estimator(DTR, LTR)
    S_logLikelihood, labelList = compute_log_likelihood_Gau(DVAL, hParams)
    # Assuming uniform class priors
    S_logPost = compute_logPosterior(S_logLikelihood, np.ones(len(labelList)) / len(labelList))
    PVAL_indices = S_logPost.argmax(0)
    # Map predicted indices back to actual class labels
    PVAL = np.array([labelList[i] for i in PVAL_indices])
    error_rate = compute_error_rate(PVAL, LVAL)
    return error_rate

def compute_LDA_parameters(D, L):
    """
    Compute the LDA parameters: shared covariance matrix and class-specific means.
    """
    mu_classes = {}
    cov_shared = np.zeros((D.shape[0], D.shape[0]))

    for label in np.unique(L):
        D_class = D[:, L == label]
        mu_classes[label] = compute_mu_C(D_class)[0]
        cov_shared += compute_mu_C(D_class)[1] * D_class.shape[1]

    cov_shared /= D.shape[1]
    return mu_classes, cov_shared

def LDA_log_likelihood(D, mu_classes, cov_shared):
    """
    Compute the log likelihoods for LDA using the shared covariance matrix.
    """
    S = np.zeros((len(mu_classes), D.shape[1]))
    inv_cov_shared = np.linalg.inv(cov_shared)
    for label, mu in mu_classes.items():
        S[label, :] = logpdf_GAU_ND(D, mu, cov_shared)
    return S

def compute_covariance_matrix(D, L):
    """
    Compute the covariance matrix for each class in the dataset.
    Returns a dictionary where keys are class labels and values are the covariance matrices.
    """
    classes = np.unique(L)
    covariance_matrices = {}
    for label in classes:
        D_class = D[:, L == label]
        _, C = compute_mu_C(D_class)
        covariance_matrices[label] = C
    return covariance_matrices

def compute_correlation_matrix(C):
    """
    Compute the Pearson correlation matrix from the covariance matrix.
    """
    variances = np.diag(C)
    std_devs = np.sqrt(variances)
    correlation_matrix = C / (std_devs[:, None] * std_devs[None, :])
    return correlation_matrix

def evaluate_models_on_features(D, L, feature_indices, model_estimators):
    D_reduced = D[feature_indices, :]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D_reduced, L)
    
    results = {}
    for model_name, estimator in model_estimators.items():
        error_rate = evaluate_model(DTR, LTR, DVAL, LVAL, estimator)
        results[model_name] = error_rate
        print(f"{model_name} - Error rate: {error_rate:.2f}%")
    
    return results
if __name__ == '__main__':
    # Load the project data
    file_path = 'Traindata.txt'  # Replace with your actual file path
    D, L = load_project_data(file_path)

    # Split the data into training and validation sets
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # Define model estimators
    model_estimators = {
        "MVG": Gau_MVG_ML_estimates,
        "Tied MVG": Gau_Tied_ML_estimates,
        "Naive Bayes": Gau_Naive_ML_estimates
    }

    # # Evaluate without PCA
    # print("Evaluating without PCA:")
    # for model_name, estimator in model_estimators.items():
    #     error_rate = evaluate_model(DTR, LTR, DVAL, LVAL, estimator)
    #     print(f"{model_name} - Error rate: {error_rate:.2f}%")

    # # Apply PCA with different number of components
    # for n_components in range(1, DTR.shape[0] + 1):
    #     print(f"\nEvaluating with PCA (n_components={n_components}):")
    #     # Compute PCA on training data
    #     mu_pca, P = compute_pca(DTR, n_components)
    #     # Apply PCA transformation
    #     DTR_pca = apply_pca(DTR, mu_pca, P)
    #     DVAL_pca = apply_pca(DVAL, mu_pca, P)
    #     for model_name, estimator in model_estimators.items():
    #         error_rate = evaluate_model(DTR_pca, LTR, DVAL_pca, LVAL, estimator)
    #         print(f"{model_name} with PCA - Error rate: {error_rate:.2f}% (n_components={n_components})")

    # Apply MVG Model
    hParams_MVG = Gau_MVG_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_MVG)
    S_logPost = compute_logPosterior(S_logLikelihood, np.ones(2) / 2.)
    PVAL = S_logPost.argmax(0)
    print("MVG - Error rate: %.2f%%" % compute_error_rate(PVAL, LVAL))


    
    mu_classes, cov_shared = compute_LDA_parameters(DTR, LTR)
    S_logLikelihood_LDA = LDA_log_likelihood(DVAL, mu_classes, cov_shared)
    S_logPost_LDA = compute_logPosterior(S_logLikelihood_LDA, np.ones(2) / 2.)
    PVAL_LDA = S_logPost_LDA.argmax(0)
    error_rate_LDA = compute_error_rate(PVAL_LDA, LVAL)
    print("LDA - Error rate: %.2f%%" % error_rate_LDA)
    
    
    # Apply Naive Bayes Model
    hParams_Naive = Gau_Naive_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Naive)
    S_logPost = compute_logPosterior(S_logLikelihood, np.ones(2) / 2.)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.2f%%" % compute_error_rate(PVAL, LVAL))

    # Apply Tied Gaussian Model
    hParams_Tied = Gau_Tied_ML_estimates(DTR, LTR)
    S_logLikelihood = compute_log_likelihood_Gau(DVAL, hParams_Tied)
    S_logPost = compute_logPosterior(S_logLikelihood, np.ones(2) / 2.)
    PVAL = S_logPost.argmax(0)
    print("Tied Gaussian - Error rate: %.2f%%" % compute_error_rate(PVAL, LVAL))

     # Compute covariance matrices for each class
    covariance_matrices = compute_covariance_matrix(DTR, LTR)

    # Print covariance matrices
    print("Covariance Matrices for Each Class:")
    for label, C in covariance_matrices.items():
        print(f"Class {label} Covariance Matrix:\n{C}\n")

    # Compute and print correlation matrices for each class
    print("Correlation Matrices for Each Class:")
    for label, C in covariance_matrices.items():
        correlation_matrix = compute_correlation_matrix(C)
        print(f"Class {label} Correlation Matrix:\n{correlation_matrix}\n")
     # Test with all features
    print("Testing with all features:")
    mvg_error = evaluate_model(DTR, LTR, DVAL, LVAL, Gau_MVG_ML_estimates)
    naive_error = evaluate_model(DTR, LTR, DVAL, LVAL, Gau_Naive_ML_estimates)
    tied_error = evaluate_model(DTR, LTR, DVAL, LVAL, Gau_Tied_ML_estimates)
    print(f"MVG - Error rate: {mvg_error:.2f}%")
    print(f"Naive Bayes Gaussian - Error rate: {naive_error:.2f}%")
    print(f"Tied Gaussian - Error rate: {tied_error:.2f}%")
    
    # Test with only Features 1-4
    DTR_reduced = DTR[:4, :]
    DVAL_reduced = DVAL[:4, :]
    
    print("\nTesting with Features 1-4:")
    mvg_error_reduced = evaluate_model(DTR_reduced, LTR, DVAL_reduced, LVAL, Gau_MVG_ML_estimates)
    naive_error_reduced = evaluate_model(DTR_reduced, LTR, DVAL_reduced, LVAL, Gau_Naive_ML_estimates)
    tied_error_reduced = evaluate_model(DTR_reduced, LTR, DVAL_reduced, LVAL, Gau_Tied_ML_estimates)
    print(f"MVG - Error rate: {mvg_error_reduced:.2f}%")
    print(f"Naive Bayes Gaussian - Error rate: {naive_error_reduced:.2f}%")
    print(f"Tied Gaussian - Error rate: {tied_error_reduced:.2f}%")
    
   
    # Evaluate using Features 1-2
    print("Testing with Features 1-2:")
    features_1_2_results = evaluate_models_on_features(D, L, [0, 1], model_estimators)
   
    # Evaluate using Features 3-4
    print("\nTesting with Features 3-4:")
    features_3_4_results = evaluate_models_on_features(D, L, [2, 3], model_estimators)
   
    D = D[:, L != 0]
    L = L[L != 0]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    # Evaluate without PCA
    print("Evaluating without PCA:")
    for model_name, estimator in model_estimators.items():
        error_rate = evaluate_model(DTR, LTR, DVAL, LVAL, estimator)
        print(f"{model_name} - Error rate: {error_rate:.2f}%")
        
  
    
    # Apply PCA with different number of components
    for n_components in range(1, DTR.shape[0] + 1):
        print(f"\nEvaluating with PCA (n_components={n_components}):")
        
        # Apply PCA on training and validation sets
        DTR_pca = apply_pca(DTR, n_components)
        DVAL_pca = apply_pca(DVAL, n_components)
        
        for model_name, estimator in model_estimators.items():
            error_rate = evaluate_model(DTR_pca, LTR, DVAL_pca, LVAL, estimator)
            print(f"{model_name} with PCA - Error rate: {error_rate:.2f}% (n_components={n_components})")
    