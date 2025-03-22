# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 20:57:22 2024

@author: vasou
"""

import numpy as np

def despiking_interpolation(t,x,detected_inds):
    t_clean = np.delete(t, detected_inds)
    x_clean = np.delete(x, detected_inds)
    
    interpolated_x = np.interp(t,t_clean, x_clean)
    
    return interpolated_x
    
    
    