# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 20:35:11 2024

@author: vasou
"""
import numpy as np

def interp1d(x, y, x_spikes):
    if len(x) != len(y):
        raise ValueError('Inputs x and t must be vectors of the same length')
    s=len(x_spikes)
    y_spikes=[0]*len(x_spikes)
    for i in range(s):
        
        dy = y[i+1]-y[i-1]
        k=(x[i]-x[i-1])/(x[i+1]-x[i-1]) 
        y_spikes[i]=y[i-1]+k*dy
        

    return y_spikes  


      
    
       
           
         