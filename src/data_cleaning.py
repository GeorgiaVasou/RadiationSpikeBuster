# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 10:10:37 2024

@author: vasou
"""
import numpy as np
import matplotlib as plt
import scipy.optimize as sc
 

def despiking_rolling_window(x,window_length,multiplier):
 """
  Parameters
  ----------
  x : x values.
  window_lenght :Lenght of the rolling window that the method is applied.
  multiplier : parameter used to costomize the threshold.

  Returns
  -------
  d_spikes : detected spikes.

  """
 N=len(x)
 data_groups = len(x) // window_length 
 m1 = [0] * data_groups
 s2 = [0] * data_groups

 for i in range(data_groups):
    group = x[i*window_length: (i+1)*window_length]
    m1[i] = np.mean(group)
    s2[i] = np.std(group)


 detected_inds=[]
 for i in range(N):
    group_index = i // window_length
    if group_index < data_groups:
        threshold = m1[group_index] + multiplier* s2[group_index]
        if x[i]>threshold:
            detected_inds.append(i)
        
 return detected_inds

 
def despiking_overlapping_window(x, window_length, multiplier):
    """
    Parameters
    ----------
    x : x values.
    window_lenght :Lenght of the overlapping window that the method is applied.
    multiplier : parameter used to costomize the threshold.

    Returns
    -------
    d_spikes : detected spikes.

    """
    N = len(x)

    mean_new = []
    std_new = []
    detected_spikes = []

    for j in range(N):
        w_start = max(0, j - window_length // 2)
        w_finish = min(N, j + window_length // 2)
        g_window = x[w_start:w_finish]
        
        mean_new1 = np.mean(g_window)
        std_new1 = np.std(g_window)
        mean_new.append(mean_new1)
        std_new.append(std_new1)
        
        if x[j] > mean_new1 + multiplier * std_new1:
            detected_spikes.append(j)

       
        
    return detected_spikes

def despiking_median_iqr(x,window_lenght,multiplier):
    """
    Parameters
    ----------
    x : x values.
    window_lenght :Lenght of the overlapping window that the method is applied.
    multiplier : parameter used to costomize the threshold.

    Returns
    -------
    d_spikes : detected spikes.

    """
    median=[]
    interquantile_range=[]
    d_spikes=[]
    N=len(x)

    for l in range(N):
        w_start=max(0,l-window_lenght//2)
        w_finish=min(N,l+window_lenght//2)
        g_windows=x[w_start:w_finish]
       
        median_value=np.median(g_windows)
        median.append(median_value)
       
        q=np.percentile(g_windows,[25,75])
        iqr=q[1]-q[0]
        interquantile_range.append(iqr)
        
        if x[l]>np.array(median_value)+4*np.array(iqr):
            d_spikes.append(l)
    
    return d_spikes


def interpolation(t,x,detected_inds):
    """

    Parameters
    ----------
    t : time values.
    x : x values.
    detected_inds : Detected spikes.

    Returns
    -------
    interpolated_x : interpolated time series.

    """
    t_clean = np.delete(t, detected_inds)
    x_clean = np.delete(x, detected_inds)
    
    interpolated_x = np.interp(t,t_clean, x_clean)
    
    return interpolated_x
    
def data_fitting(t,x,fun,w,initial_guesses,upper_bounds,lower_bounds):
 """
    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    fun : function used for the fitting of the data.
    w : parameter initializing the range of the fitting.
    initial_guesses : initial guesses of the parameter.
    upper_bounds :  Specifies the maximum allowable values for the parameters being optimized during the curve fitting.
    lower_bounds :  Specifies the minimum allowable values for the parameters being optimized during the curve fitting.

    Returns
    -------
    None.

    """
    


 N= len(x)
 smoothed_x = np.copy(x)

 for i in range(w,N-w):
   x_w= x[i-w:i+w]
   t_w= t[i-w:i+w]
   fit_params1 = sc.curve_fit(fun, t_w, x_w, 
                     initial_guesses, check_finite=True, 
                     bounds=(lower_bounds, upper_bounds))[0]
   smoothed_x[i]=fun(t[i],*fit_params1)
 plt.subplot(2, 1, 1)  
 plt.plot(t, smoothed_x, '-b')
 plt.subplot(2, 1, 2)
 plt.plot(t, x-smoothed_x,'-r')

def z_score_filter(df, window_length, threshold ,  threshold2):
   """
    Parameters
    ----------
    df : DataFrame.
    window_length :  Lenght of the rolling window that the method is applied.
    
    threshold: maximum accepted values.
    threshold2: minimum accepted values.

    Returns
    -------
    df_invalid : DataFrame icluding the detected errors.

     """
  
   
   m = df['x1_new'].rolling(window=window_length, min_periods=int(window_length*0.8), center=True).mean()
   s = df['x1_new'].rolling(window=window_length, min_periods=int(window_length*0.8), center=True).std()
    
   s = s.replace(0, np.nan)
    
   z_scores = (df['x1_new'] - m) / s
    
   invalid_data_u = z_scores > threshold
   invalid_data_d = z_scores < -threshold2
   
    
   df_invalid = df.copy()
   df_invalid.loc[invalid_data_u, 'x1_new'] = np.nan
   df_invalid.loc[invalid_data_d, 'x1_new'] = np.nan
    
   return df_invalid

def data_visualization(df, x1, x2, x1_new=None, x2_new=None, plot_type='all'):
    """
    Parameters
    ----------
    df : DataFrame.
    x1 : str,x1 values.
    x2 : str,x2 values.
    x1_new : str, optional
        interpolated data . The default is None.
    x2_new : str, optional
        interpolated data . The default is None.
    plot_type : str, optional
        Type of plot to generate ('time_series', 'scatter', 'histogram', 'boxplot', or 'all'). The default is 'all'.

    Returns
    -------
    None.

    """
   
   
    if plot_type == 'time_series' or plot_type == 'all':
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[x1], marker='o')
        if x1_new:
            plt.plot(df.index, df[x1_new], linestyle='--', color='red')
       
        plt.title('Time Series: x1')
    

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[x2], marker='o')
        if x2_new:
            plt.plot(df.index, df[x2_new], linestyle='--', color='red')
       
        plt.title('Time Series x2')
        
   
    if plot_type == 'scatter' or plot_type == 'all':
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x1], df[x2], c='blue', alpha=0.5)
        plt.xlabel(x1)
        plt.ylabel(x2)
        plt.title('Scatter Plot: x1 and x2')
       
    
    if plot_type == 'histogram' or plot_type == 'all':
        plt.figure(figsize=(10, 6))
        plt.hist(df[x1].dropna(), bins=30, alpha=0.5, color='blue')
        if x1_new:
            plt.hist(df[x1_new], bins=30, alpha=0.5, color='red')
       
        plt.title(' Histogram for x1')
     

        plt.figure(figsize=(10, 6))
        plt.hist(df[x2].dropna(), bins=30, alpha=0.5,color='blue')
        if x2_new:
            plt.hist(df[x2_new], bins=30, alpha=0.5, color='red')
        
        plt.title(' Histogram for x2')
       

   
    if plot_type == 'boxplot' or plot_type == 'all':
       plt.figure(figsize=(12, 6))

       plt.subplot(1, 2, 1)
       plt.boxplot(df[x1].dropna(), labels=[f'{x1}'])
       plt.title(f'Box Plot: {x1}')
    
       plt.subplot(1, 2, 2)        
       plt.boxplot(df[x2].dropna(), labels=[f'{x2}'])
       plt.title(f'Box Plot: {x2}')
       

