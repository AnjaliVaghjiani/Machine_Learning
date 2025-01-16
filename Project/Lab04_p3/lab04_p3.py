import numpy as np
import matplotlib.pyplot as plt

def load(f):
    label = []
    x = []
    with open(f, 'r') as myData:
        for line in myData:
            features, bool_value = line.strip().split(",")[:-1], line.strip().split(",")[-1]
            features = np.array([float(i) for i in features])
            x.append(features)
            label.append(bool_value)
    x = np.array(x)
    return x.T, np.array(label, dtype=np.int32)

def vcol(x):
    return x.reshape((x.size, 1))

def logpdf_GAU_ND(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    logdetC = np.linalg.slogdet(C)[1]
    invC = np.linalg.inv(C)
    v = np.einsum('ij,ji->i', XC.T, np.dot(invC, XC))
    return -0.5 * (M * np.log(2 * np.pi) + logdetC + v)

def plot_feature_gaussian(D, L, feature_index):
    plt.figure(figsize=(10, 6))

    # Map class labels to descriptive names
    class_names = {0: 'Counterfeit', 1: 'Genuine'}

    # Plot histograms for each class
    for class_label in np.unique(L):
        D_class = D[feature_index, L == class_label]
        plt.hist(D_class.ravel(), bins=50, density=True, alpha=0.6, label=f'{class_names[class_label]}')

        # Compute Gaussian parameters
        mu_ML = D_class.mean()
        sigma_ML = D_class.std()

        # Plot Gaussian density
        XPlot = np.linspace(D_class.min(), D_class.max(), 1000)
        gaussian_density = np.exp(logpdf_GAU_ND(vrow(XPlot), vcol(mu_ML), vcol(sigma_ML**2)))
        plt.plot(XPlot, gaussian_density, label=f'Gaussian Density {class_names[class_label]}')

    plt.title(f'Feature {feature_index + 1}')
    plt.xlabel('Feature Value')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def analyze_features(D, L):
    for feature_index in range(D.shape[0]):
        plot_feature_gaussian(D, L, feature_index)

def vrow(x):
    return x.reshape((1, x.size))

if __name__ == '__main__':
    # Load the data
    file_path = 'trainData.txt'  # Change this to your actual data file path
    D, L = load(file_path)
    
    # Analyze features for each class
    analyze_features(D, L)
