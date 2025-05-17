# -*- coding: utf-8 -*-
"""
Created on Sat May 11 09:50:40 2024

@author: anjal
"""
import numpy as np
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def mrow(v):
    return v.reshape((1, v.size))

def load(f):
    label = []
    myData = open(f, 'r')
    x=[]
    for line in myData:
        
        x1, x2, x3, x4, x5, x6, bool_value = line.strip().split(",")
        features = line.split(',')[0:6]
        
        features = (np.array([float(i) for i in features]))
        # x.append([float(v) for v in line.strip().split(',')[:-1]])
        x.append(features)
        label.append(bool_value)
       
    x = np.array(x)
    return x, np.array(label, dtype=np.int32)
    


def visualize_histograms(x, l):
    
    f_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6']
    
    true_features = x[l== 1]
    false_features = x[l==0]
    
    
    
    for i in range(6):
        plt.figure()
        plt.hist(true_features[:, i], bins=10, density=True, alpha=0.4, label='Genuine')
        plt.hist(false_features[:, i], bins=10, density=True, alpha=0.4, label='Counterfeit')
       
        plt.xlabel(f_names[i])
        plt.ylabel('Density')
        plt.legend()
        plt.show()
        
    # Plot scatter plots for all combinations of feature pairs
  
    plt.figure()
    plt.scatter(true_features[:, 0], true_features[:, 1], color='blue', label='Genuine')
    plt.scatter(false_features[:, 0], false_features[:, 1], color='orange', label='Counterfeit')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.scatter(true_features[:, 2], true_features[:, 3], color='blue', label='Genuine')
    plt.scatter(false_features[:, 2], false_features[:, 3], color='orange', label='Counterfeit')
    plt.xlabel('Feature 3')
    plt.ylabel('Feature 4')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.scatter(true_features[:, 4], true_features[:, 5], color='blue', label='Genuine')
    plt.scatter(false_features[:, 4], false_features[:, 5], color='orange', label='Counterfeit')
    plt.xlabel('Feature 5')
    plt.ylabel('Feature 6')
    plt.legend()
    plt.show()
    
def compute_statistics(features, labels):
    true_features = features[labels == 1]
    false_features = features[labels == 0]
    
    true_means = np.mean(true_features, axis=0)
    false_means = np.mean(false_features, axis=0)
    true_vars = np.var(true_features, axis=0)
    false_vars = np.var(false_features, axis=0)
    
    return true_means, false_means, true_vars, false_vars


    
  
       
def main():
    x , l =load('trainData.txt')   
    visualize_histograms(x, l)    
    
    
    true_means, false_means, true_vars, false_vars = compute_statistics(x, l)

    f_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6']
    
    print("Genuine class statistics:")
    for i in range(6):
        print(f"{f_names[i]}: mean = {true_means[i]:f}, variance = {true_vars[i]:f}")
    
    print("\nCounterfeit class statistics:")
    for i in range(6):
        print(f"{f_names[i]}: mean = {false_means[i]:f}, variance = {false_vars[i]:f}")

    
    
   
            
    
if __name__ == "__main__":
     main()

