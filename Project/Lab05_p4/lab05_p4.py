import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy.stats import norm

def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))

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

def ML_estimares_class(LTR, DTE):
    data = {}
    for i in set(LTR):
       DX = DTE[:, LTR==i]
       mu, C = compute_mu_C(DX)
       data[i] = compute_mu_C(DX)
       # print(mu,C) 
    return data 


def naive_ML_estimation(LTR, DTE):
    data ={}
    for i in set(LTR):
        DX = DTE[:, LTR==i]
        mu, C = compute_mu_C(DX)
        data[i] = (mu, C * np.eye(DTE.shape[0]))
        
    return data

def tied_ML_estimation(LTR, DTE):
    data={}
    CGlobal = 0
    m = {}
    for i in set(LTR):
        DX = DTE[:, LTR==i]
        mu, C = compute_mu_C(DX)
        CGlobal += C * DX.shape[1]
        m[i] = mu
    CGlobal = CGlobal/D.shape[1]
    for i in set(LTR):
        data[i] = (m[i], CGlobal)
    return data

def compute_log_likelihood_Gau(D, hParams):
    labelList = sorted(hParams.keys())
    S = np.zeros((len(labelList), D.shape[1]))
    for i, lab in enumerate(labelList):
        mu, C = hParams[lab]
        S[i, :] = logpdf_GAU_ND(D, mu, C)
    return S, labelList

def loglikelihood(DTE, data):
    S = np.zeros((len(data), DTE.shape[1]))
    for i in range(S.shape[0]):
        #data[i][0] is mean and data[i][1] is covaraince
        S[i, :] = logpdf_GAU_ND(DTE, data[i][0], data[i][1])
    return S

def compute_class_posterior(S, value):
    SJoint = S + vcol(np.log(value))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
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

def PCA(D, m):
    mu = D.mean(axis=1, keepdims=True)
    DC = D - mu  # Center data
    C = np.dot(DC, DC.T) / D.shape[1]  # Covariance matrix
    s, U = np.linalg.eigh(C)  # Eigen-decomposition
    P = U[:, ::-1][:, :m]  # Take top-m eigenvectors
    return np.dot(P.T, D), P, mu

def evaluate_model(DTR, LTR, DVAL, LVAL, model_estimator):
    """
    Train the model using the provided estimator and evaluate its performance on the validation set.
    Returns the error rate.
    """
    hParams = model_estimator(LTR, DTR)
    S_logLikelihood, labelList = compute_log_likelihood_Gau(DVAL, hParams)
    # Assuming uniform class priors
    S_logPost = compute_class_posterior(S_logLikelihood, np.ones(len(labelList)) / len(labelList))
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
    variances = np.diag(C)
    std_devs = np.sqrt(variances)
    correlation_matrix = C / (std_devs[:, None] * std_devs[None, :])
    return correlation_matrix

def evaluate_models_on_features(D, L, feature_indices, model_estimators):
    D_reduced = D[feature_indices, :]
    (DTR, LTR), (DVAL, LVAL) = split_db_2tot(D_reduced, L)
    
    results = {}
    for model_name, estimator in model_estimators.items():
        error_rate = evaluate_model(DTR, LTR, DVAL, LVAL, estimator)
        results[model_name] = error_rate
        print(f"{model_name} - Error rate: {error_rate:.2f}%")
    
    return results

def plot_feature_gaussian_fit(D, L, class_labels, feature_names=None):
    num_features = D.shape[0]
    num_classes = len(class_labels)

    for i in range(num_features):
        plt.figure(figsize=(12, 4))
        for j, label in enumerate(class_labels):
            plt.subplot(1, num_classes, j+1)
            # Select data for class and feature
            D_class = D[:, L == label]
            feature_data = D_class[i, :]

            # Estimate mean and std
            mu = np.mean(feature_data)
            sigma = np.std(feature_data)

            # Plot histogram
            count, bins, _ = plt.hist(feature_data, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')

            # Plot Gaussian fit
            x = np.linspace(min(bins), max(bins), 1000)
            plt.plot(x, norm.pdf(x, mu, sigma), color='red', label='Gaussian PDF')

            plt.title(f'Class {label} - Feature {i}' if not feature_names else f'Class {label} - {feature_names[i]}')
            plt.legend()

        plt.tight_layout()
        plt.show()
        
if __name__ == '__main__':
    # Load the project data
    file_path = 'Traindata.txt'  # Replace with your actual file path
    D, L = load_project_data(file_path)

    # Split the data into training and validation sets
    (DTR, LTR), (DTE, LTE) = split_db_2tot(D, L)

    # Define model estimators
    model_estimators = {
        "MVG": ML_estimares_class,
        "Tied MVG": tied_ML_estimation,
        "Naive Bayes": naive_ML_estimation
    }
    # Apply MVG Model
    data_MVG = ML_estimares_class(LTR, DTR)
    S_logLikelihood = loglikelihood(DTE, data_MVG)
    S_logPost = compute_class_posterior(S_logLikelihood, 1 / 2)
    PVAL = S_logPost.argmax(0)
    print("MVG - Error rate: %.2f%%" % compute_error_rate(PVAL, LTE))

    #tied model
    data_tied = tied_ML_estimation(LTR, DTR)
    S_logLikelihood = loglikelihood(DTE, data_tied)
    S_logPost = compute_class_posterior(S_logLikelihood, 1/2)
    PVAL = S_logPost.argmax(0)
    print("Tied Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))
        
    print()
    
    #Naive Bayes model
    data_naive = naive_ML_estimation(LTR, DTR)
    S_logLikelihood = loglikelihood(DTE, data_naive)
    S_logPost = compute_class_posterior(S_logLikelihood, 1/2)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.1f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))
        
    print()
    
    # Compute covariance matrices for each class
    covariance_matrices = compute_covariance_matrix(DTR, LTR)

    # Print covariance matrices
    print("Covariance Matrices for Each Class:")
    for label, C in covariance_matrices.items():
        print(f"Class {label} Covariance Matrix:\n{C}\n")
        print(f"Variance for class {label} is: {np.diag(C)}\n")

    # Compute and print correlation matrices for each class
    print("Correlation Matrices for Each Class:")
    for label, C in covariance_matrices.items():
        correlation_matrix = compute_correlation_matrix(C)
        print(f"Class {label} Correlation Matrix:\n{correlation_matrix}\n")
    
      # Test with all features
    class_labels = np.unique(LTR)
    plot_feature_gaussian_fit(DTR, LTR, class_labels)
    
    # Test with only Features 1-4
    DTR_reduced = DTR[:4, :]
    DVAL_reduced = DTE[:4, :]
    
    print("\nTesting with Features 1-4:")
    data_MVG = ML_estimares_class(LTR, DTR_reduced)
    S_logLikelihood = loglikelihood(DVAL_reduced, data_MVG)
    S_logPost = compute_class_posterior(S_logLikelihood, 1 / 2)
    PVAL = S_logPost.argmax(0)
    print("MVG - Error rate: %.2f%%" % compute_error_rate(PVAL, LTE))

    #tied model
    data_tied = tied_ML_estimation(LTR, DTR_reduced)
    S_logLikelihood = loglikelihood(DVAL_reduced, data_tied)
    S_logPost = compute_class_posterior(S_logLikelihood, 1/2)
    PVAL = S_logPost.argmax(0)
    print("Tied Bayes Gaussian - Error rate: %.2f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))
        
    print()
    
    #Naive Bayes model
    data_naive = naive_ML_estimation(LTR, DTR_reduced)
    S_logLikelihood = loglikelihood(DVAL_reduced, data_naive)
    S_logPost = compute_class_posterior(S_logLikelihood, 1/2)
    PVAL = S_logPost.argmax(0)
    print("Naive Bayes Gaussian - Error rate: %.2f%%" % ((PVAL != LTE).sum() / float(LTE.size) * 100))
        
    print()
    
    
   
    # Evaluate using Features 1-2
    print("Testing with Features 1-2:")
    features_1_2_results = evaluate_models_on_features(D, L, [0, 1], model_estimators)
   
    # Evaluate using Features 3-4
    print("\nTesting with Features 3-4:")
    features_3_4_results = evaluate_models_on_features(D, L, [2, 3], model_estimators)
   

    # Apply PCA with different number of components
    for n_components in range(1, DTR.shape[0] + 1):
        print(f"\nEvaluating with PCA (n_components={n_components}):")
        
        DTR_PCA, P, mu = PCA(DTR, n_components)            # Reduce training data
        DTE_PCA = np.dot(P.T, DTE - mu)         # Project test data

        
        for model_name, estimator in model_estimators.items():
            error_rate = evaluate_model(DTR_PCA, LTR, DTE_PCA, LTE, estimator)
            print(f"{model_name} with PCA - Error rate: {error_rate:.2f}% (n_components={n_components})")
    