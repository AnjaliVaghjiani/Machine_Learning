import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.shape((1, v.size))

def load(fname):
    labels = []
    features_list = []
    #classes representation in vector form
    iris_to_label = {
        'Iris-setosa': 0, #iris- setosa is considered as 0
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            values = line.strip().split(',') #splitting the data because in the file it is represented as n1,n2,n3,n4,class, where n1,...n4 are the features of the flower
            features = np.array([float(val) for val in values[:4]]) #using slicing method to access first 4 element in the list so [n1, n2, n3, n4]
            iris_type = values[4] # getting the element on 4th index which is the class
            features_list.append(features) #we are stacking the array list one on top of another
            labels.append(iris_to_label[iris_type])#list of all classes as a vectory array i.g [0, 0, 1, 2, 0 , 1....N]
        
        features_array = np.array(features_list)
        features_array = features_array.T
        labels_array = np.array(labels) #np.array() stores the array of same data type avoiding the probelems related to the datatype
        
            

    return features_array, labels_array

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx in range(4):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Setosa')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Versicolor')
        plt.hist(D2[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Virginica')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()

def plot_scatter(D, L):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }

    for dIdx1 in range(4):
        for dIdx2 in range(4):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Setosa')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Versicolor')
            plt.scatter(D2[dIdx1, :], D2[dIdx2, :], label = 'Virginica')
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()
        
def compute_mean(D):
    #we want the mean to be a coloum vector, so we first reshape it
    mu = mcol(D.mean(axis=1)) #computation of mean using numpy .mean where axis = 0 allow computation of mean over row and axis = 1 allow computation of mean over coloum
    return mu
 
def covariance_matrix(D, DC):
    mu = compute_mean(D)
    return ((D - mu) @ (D - mu).T) / float(D.shape[1])
        

            
def plot_centered_data(D, L):
    feature_names = ['Centered Sepal Length', 'Centered Sepal Width', 
                     'Centered Petal Length', 'Centered Petal Width']
    class_names = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    colors = ['blue', 'orange', 'green']
    markers = ['o', 's', '^']
    
    # Create all possible pairs of features
    feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    
    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot each pair
    for i, (feature_i, feature_j) in enumerate(feature_pairs):
        ax = axes[i]
        
        # For each class
        for class_idx in range(3):
            # Get indices for samples of this class
            class_indices = np.where(L == class_idx)[0]
            
            # Plot the scatter for this feature pair and this class
            ax.scatter(
                D[feature_i, class_indices],
                D[feature_j, class_indices],
                c=colors[class_idx],
                marker=markers[class_idx],
                s=50,
                alpha=0.7,
                label=class_names[class_idx]
            )
        
        # Add labels and grid
        ax.set_xlabel(feature_names[feature_i])
        ax.set_ylabel(feature_names[feature_j])
        ax.set_title(f'{feature_names[feature_i]} vs {feature_names[feature_j]}')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Add legend only to the first plot
        if i == 0:
            ax.legend()
    
    # Add a common legend
    fig.legend(
        labels=class_names,
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.01)
    )
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.suptitle('Centered Data Visualization', fontsize=16)
    plt.show()            
            

    
if __name__ == '__main__':

    D, L = load('iris.csv')
    # plot_hist(D, L)
    # plot_scatter(D, L)
    
    feature_names = ['Sepal length', 'Sepal Width', 'Petal Lenght', 'Petal Width']    
    class_name = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
    
    results = {}
    
    if L is not None:
        for i, class_name in enumerate(class_name):
            class_indices = L == i
            D_class = D[:, class_indices]
            
            mu = compute_mean(D_class)
            DC = D_class - mu #centering the data (to remove the mean from all points)
            C = covariance_matrix(D_class, DC)
            
            var = mcol(D_class.var(axis = 1))
            std = mcol(D_class.std(axis = 1))
            
            # Store results
            results[class_name] = {
                'mean': mu,
                'covariance': C,
                'variance': var,
                'std': std,
            }
    # plot_centered_data(DC, L)

    # var = D.var(1)
    # std = D.std(1)
    # print('Variance:', var)
    # print('Std. dev.:', std)
    # print()
    
    # for cls in [0,1,2]:
    #     print('Class', cls)
    #     DCls = D[:, L==cls]
    #     mu = DCls.mean(1).reshape(DCls.shape[0], 1)
    #     print('Mean:')
    #     print(mu)
    #     C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
    #     print('Covariance:')
    #     print(C)
    #     var = DCls.var(1)
    #     std = DCls.std(1)
    #     print('Variance:', var)
    #     print('Std. dev.:', std)
    #     print()
        
    