# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:56:40 2024

@author: anjal
"""

class details:
    def __init__(self, name, surname, country, score):
        self.name = name
        self.surname = surname
        self.country = country
        self.score = score
        
f =  open ('C:/Users/anjal/OneDrive/Desktop/Machine Learning/Labs/Lab01/ex1_data.txt', 'r')
#print(f.readlines())

list_data = []

for line in f:
    # for word in line.split():
        w= line.split(" ")
        high = 'null'
        lower = 'null'
        score = 0.0
        for i in range(3,8):
            # print(w[i])
            if high=='null' or high < float(w[i]) :
                high = float(w[i])
            if lower =='null' or lower > float(w[i]):
                lower = float(w[i])
            
            score += float(w[i])
            
        score = score - high - lower
        l= details(w[0], w[1], w[2], score)
        list_data.append(l)
        if len(list_data) > 4:
            list_data = sorted(list_data, key = lambda x: x.score, reverse=True)
highest_score = 0
country_name = ' '

for i in list_data:
    if i.score>highest_score:
        highest_score = score
        country_name = i.country
 
Total_score =0
for i in list_data:
    if i.country==country_name:
       Total_score += i.score
       
       
print('final ranking:')
j=1
for i in list_data:
    if(j<=3):
        print(j, ':', i.name, ' ', i.surname, '- Score: ',i.score)  
        j+=1
        
print()        
print('Best Country:')       
print(country_name, " - Total score: ", Total_score)       