# # -*- coding: utf-8 -*-
# """
# Created on Fri Mar 15 13:13:07 2024

# @author: anjal
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from itertools import combinations



# #C:/Users/anjal/OneDrive/Desktop/Machine Learning/Labs/Lab02/iris.csv

# # converting the row into coloumn
# def mcol(v):
#     # print("the v is: " , v.reshape(v.size, 1))
#     return v.reshape((v.size, 1))

# def load(f):
#     l = []
#     myData = open(f, 'r')
#     x=[]
#     for line in myData:
        
#         x1, x2, x3, x4, name = line.strip().split(",")
#         attrs = line.split(',')[0:4]
        
#         attrs = mcol(np.array([float(i) for i in attrs]))
#         # x.append([float(v) for v in line.strip().split(',')[:-1]])
#         x.append(attrs)
#         if name == 'Iris-setosa':
#             l.append(0)
           
#         elif name=='Iris-virginica':
#             l.append(2) 
#         elif name=='Iris-versicolor':
#             l.append(1)
#     return np.hstack(x), np.array(l, dtype=np.int32)




# def visualize_histograms(x, l):
#     f_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
    
    
#     D0 = x[:, l==0]
#     D1 = x[:, l==1]
#     D2 = x[:, l==2]
    

   
#     for i in range(4):
#         plt.figure()
#         plt.hist(D0[i, :], bins=10, density=True, alpha=0.4, label='Setosa')
#         plt.hist(D1[i, :], bins=10, density=True, alpha=0.4, label='Versicolor')
#         plt.hist(D2[i, :], bins=10, density=True, alpha=0.4, label='Virginica')
#         plt.xlabel(f_names[i])
#         plt.ylabel('Density')
#         plt.legend()
#         plt.show()
        
#     # Plot scatter plots for all combinations of feature pairs
#         for  j in range(i+1, 4):
#             plt.figure()
#             plt.scatter(D0[i, :], D0[j, :], color='blue', label='Setosa')
#             plt.scatter(D1[i, :], D1[j, :], color='orange', label='Versicolor')
#             plt.scatter(D2[i, :], D2[j, :], color='green', label='Virginica')
#             plt.xlabel(f_names[i])
#             plt.ylabel(f_names[j])
#             plt.legend()
#             plt.show()
            
#         for k in range(i+1):
#             if i<3:
#                 plt.figure()
#                 plt.scatter(D0[i+1, :], D0[k, :], color='blue', label='Setosa')
#                 plt.scatter(D1[i+1, :], D1[k, :], color='orange', label='Versicolor')
#                 plt.scatter(D2[i+1, :], D2[k, :], color='green', label='Virginica')
#                 plt.xlabel(f_names[i+1])
#                 plt.ylabel(f_names[k])
#                 plt.legend()
#                 plt.show()
                
#             # print(f_names[i], f_names[j])
           
        
# def main():
#     x , l =load('iris.csv')   
#     visualize_histograms(x, l)     
            
    
# if __name__ == "__main__":
#      main()

import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


def load2():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

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

    
if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load('iris.csv')
    plot_hist(D, L)
    plot_scatter(D, L)

    mu = D.mean(1).reshape((D.shape[0], 1))
    print('Mean:')
    print(mu)
    print()

    DC = D - mu
    
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    print('Covariance:')
    print(C)
    print()

    var = D.var(1)
    std = D.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()
    
    for cls in [0,1,2]:
        print('Class', cls)
        DCls = D[:, L==cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print('Mean:')
        print(mu)
        C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
        print('Covariance:')
        print(C)
        var = DCls.var(1)
        std = DCls.std(1)
        print('Variance:', var)
        print('Std. dev.:', std)
        print()
        
    