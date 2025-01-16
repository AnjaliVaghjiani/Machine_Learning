# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:51:19 2024

@author: anjal
"""
import math


class public_transport:
    def __init__(self, ID, bus_server, x, y, time):
        self.ID = ID
        self.bus_server = bus_server
        self.x = x
        self.y = y
        self.time = time
        
def distance( x1, x2,  y1, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2) 

def speed(data,server):
    tot_distance =0 
    tot_time=0
    tmp_ser= 0
    x1=0
    x2=y1=y2=counter=0
    Bus_data = sorted(data, key = lambda x:x.bus_server)
    for d in Bus_data:
        
            
            if int(d.bus_server) == int(server):
                if tmp_ser==0 or tmp_ser!=d.ID:
                    tmp_ser = d.ID
                    dis,t = compute_Distance(data, d.ID)
                    print(t)
                    tot_distance+=dis
                    tot_time += t
                    if counter>=2:
                        tot_distance += distance(x1, x2, y1, y2)
    # print(tot_time)
    print(tot_distance/tot_time)
    return tot_distance/tot_time
            
def compute_Distance(data,bus_id):
    Bus_data = sorted(data, key=lambda x:x.ID)
    x1=0
    x2=0
    y1=0
    y2=0
    t1=t2=time=0
    count=0
    total_distance =0
    for d in Bus_data:
        
        if int(d.ID)==int(bus_id):
            count +=1
            if x1==0 or y1==0:
                x1=d.x
                y1 = d.y
                t1 = d.time
            elif (x1 != 0 and x2==0) or (y1!=0 and y2==0):
                x2=d.x
                y2 = d.y
                t2 =d.time
            elif (x1!= 0 and x2!=0) or (y1!=0 and y2!=0):
                x1 = x2
                x2 = d.x
                y1=y2 
                t1=t2
                t2=d.time
            # for a, b in zip(Bus_data[:-1], Bus_data[1:]):
            if count>=2:   
                total_distance += distance(x1, x2, y1, y2)
                time += t2-t1
    return total_distance, time                 
        
f = open('ex2_data.txt', 'r')

bus_data =[]
for line in f:
        w=line.split(" ")
        l = public_transport(int(w[0]), int(w[1]), int(w[2]), int(w[3]), int(w[4]))
        bus_data.append(l)
        bus_data = sorted(bus_data, key = lambda x:x.ID)
        
       

                
               
compute_Distance(bus_data, 1976)
speed(bus_data, 4)


#comments to continue the exercise
#to compute average speed we need to do x2-x1/t2-t1
#same procedure for multiple time
#find a way to write a function
                