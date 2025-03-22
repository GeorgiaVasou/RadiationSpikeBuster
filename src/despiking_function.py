# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:20:43 2024

@author: vasou
"""
import numpy as np
import pandas as pd

def despiking_function(x, window_size, multiplier):
    t = np.arange(len(x))
    df = pd.DataFrame(data=np.vstack((t,x)).T, columns=['time', 'flux'])
    df.set_index('time')

    m = df['flux'].rolling(window = window_size, min_periods=int(window_size*0.8), center=True).mean()
    s = df['flux'].rolling(window = window_size, min_periods=int(window_size*0.8), center=True).std()
   
    detected_inds = np.where(x > m +  multiplier * s)[0]
   
    return detected_inds