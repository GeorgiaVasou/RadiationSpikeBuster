# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:00:39 2024

@author: vasou
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_cleaning as dc

filename ="C:/Users/vasou/Desktop/PYTHON SAT/dataset_ace_epam_rt.csv"

data = np.genfromtxt(fname = filename, delimiter = ',', skip_header=1)

t = np.datetime64("0000-01-01") + data[:,0] * np.timedelta64(1, 'ms')

# Create a DataFrame for the data
x = {'x1': data[:,3], 'x2': data[:,4], 'x_quality': data[:,2]}
df = pd.DataFrame(x)

# Set x1 and x2 to NaN where x_quality is not 0
df.loc[df['x_quality'] != 0, ['x1', 'x2']] = np.nan

# Detect errors where x_quality is not 0
detected_errors = np.where(df['x_quality'] != 0)[0]

# Perform interpolation to fix the errors in x1
df['x1_new'] = dc.interpolation(t.astype('int64'), df['x1'], detected_errors)
df['x2_new'] = dc.interpolation(t.astype('int64'), df['x2'], detected_errors)

#%% despiking d
window_length =5

m = df['x1_new'].rolling(window=window_length, center=True).median()

df['Q1'] = df['x1_new'].rolling(window=window_length, center=True).quantile(0.25)
df['Q3'] = df['x1_new'].rolling(window=window_length, center=True).quantile(0.75)
IQR = df['Q3'] - df['Q1']
df_filtered = df[df['x1_new'] > m-0.5*IQR ]
t_filtered = t[df['x1_new'] >  m-0.5*IQR ]
plt.figure()
plt.semilogy(t_filtered,df_filtered['x1_new'], '--b')


m_filtered = df_filtered['x1_new'].rolling(window=window_length, center=True).median()

df_filtered['Q1'] = df_filtered['x1_new'].rolling(window=window_length, center=True).quantile(0.25)
df_filtered['Q3'] = df_filtered['x1_new'].rolling(window=window_length, center=True).quantile(0.75)
IQR_filtered = df_filtered['Q3'] - df_filtered['Q1']

df_filtered1 = df_filtered[df_filtered['x1_new'] > m_filtered - 0.5 * IQR_filtered]
t_filtered1 = t_filtered[df_filtered['x1_new'] > m_filtered - 0.5 * IQR_filtered]

plt.figure()
plt.semilogy(t_filtered1, df_filtered1['x1_new'], '--b')
plt.show()


m1 =df_filtered1['x1_new'].rolling(window = window_length, min_periods=int(window_length*0.8), center=True).mean()
s1 = df_filtered1['x1_new'].rolling(window = window_length, min_periods=int(window_length*0.8), center=True).std()
df_filtered2 = df_filtered1[df_filtered1['x1_new'] > m1 - 0.5 * s1]
t_filtered2= t_filtered1[df_filtered1['x1_new'] > m1- 0.5 * s1]
plt.figure()
plt.semilogy(t_filtered2, df_filtered2['x1_new'], '--b')
plt.show()

m2 =df_filtered2['x1_new'].rolling(window = window_length, min_periods=int(window_length*0.8), center=True).mean()
s2 = df_filtered2['x1_new'].rolling(window = window_length, min_periods=int(window_length*0.8), center=True).std()
df_filtered3= df_filtered2[df_filtered2['x1_new'] > m2 - 0.5 * s2]
t_filtered3= t_filtered2[df_filtered2['x1_new'] > m2- 0.5 * s2]
plt.figure()
plt.semilogy(t_filtered3, df_filtered3['x1_new'], '--b')
plt.show()

m3 =df_filtered3['x1_new'].rolling(window = window_length, min_periods=int(window_length*0.8), center=True).mean()
s3 = df_filtered3['x1_new'].rolling(window = window_length, min_periods=int(window_length*0.8), center=True).std()
df_filtered4= df_filtered3[df_filtered3['x1_new'] > m3 - 0.5 * s3]
t_filtered4= t_filtered3[df_filtered3['x1_new'] > m3- 0.5 * s3]
plt.figure()
plt.semilogy(t_filtered4, df_filtered4['x1_new'], '--b')
plt.show()


m4 =df_filtered4['x1_new'].rolling(window = window_length, min_periods=int(window_length*0.8), center=True).mean()
s4 = df_filtered4['x1_new'].rolling(window = window_length, min_periods=int(window_length*0.8), center=True).std()
df_filtered5= df_filtered4[df_filtered4['x1_new'] > m4- 0.5 * s4]
t_filtered5= t_filtered4[df_filtered4['x1_new'] > m4- 0.5 * s4]
plt.figure()
plt.semilogy(t_filtered5, df_filtered5['x1_new'], '--b')
plt.show()


#%% despiking u

df_filtered5['x1_new']=np.where(df_filtered5['x1_new']<10**(-2) , 0 ,df_filtered5['x1_new'])
df_filtered6 = dc.despiking_median_iqr(df_filtered5['x1_new'].to_numpy(),20,3)
t_filtered6 = dc.despiking_median_iqr(df_filtered5['x1_new'].to_numpy(),20,3)
plt.figure(1)
plt.semilogy(t_filtered6,df_filtered6,'--b')

#%%
# Plotting the results
plt.figure()
plt.subplot(2,1,1)
plt.semilogy(t,df['x1'],'-b')
plt.semilogy(t, df['x1_new'], '-b')
plt.ylim([1E+2, 1E+7])
plt.subplot(2,1,2)
plt.semilogy(t, df['x2'], '-r',)
plt.semilogy(t, df['x2_new'], '-r')
plt.ylim([1, 1E+7])
plt.show()

