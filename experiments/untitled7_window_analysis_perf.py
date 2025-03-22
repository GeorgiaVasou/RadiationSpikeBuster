# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 13:42:40 2024

@author: vasou
"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from interp1d import interp1d
import time



N = 1000
nSpikes = 20

# Δημιουργία αρχικού σήματος
t = np.arange(N)
x = np.sin(2*np.pi*t/500)  + 0.2*np.random.randn(N)

plt.figure()
plt.plot(t, x, '-k')
plt.grid(True)
plt.title('Original Signal')

# Προσθήκη spikes
spike_inds = np.random.permutation(np.arange(N))[:nSpikes]
x[spike_inds] = x[spike_inds] + 3

plt.figure()
plt.plot(t, x)
plt.grid(True)
plt.title('Signal with Spikes')

# Windowed Analysis
window_length = 20
data_groups = len(x) // window_length 

m1 = [0] * data_groups
s2 = [0] * data_groups

for i in range(data_groups):
    group = x[i*window_length: (i+1)*window_length]
    m1[i] = np.mean(group)
    s2[i] = np.std(group)
t_n = np.arange(window_length/2, N, window_length)

plt.figure()
plt.plot(t, x, '-b')
plt.plot(t_n, m1, '-r')
plt.plot(t_n, np.array(m1) + 3 * np.array(s2), '--r')
plt.grid(True)
plt.title('Windowed Analysis')

# Εντοπισμός indices
detected_inds = []
for i in range(N):
    group_index = i // window_length
    if group_index < data_groups:
        threshold = m1[group_index] + 3 * s2[group_index]
        if x[i] > threshold:
            detected_inds.append(i)

# Διαγραφή των ανιχνευμένων indices από t και x
t_clean = np.delete(t, detected_inds)
x_clean = np.delete(x, detected_inds)
start = time.time()
interp= interp1d(t_clean, x_clean, detected_inds)
end = time.time()

print('time=' , end-start)
print(interp)
x[detected_inds]=interp
plt.figure()
plt.plot(t, x, '-b', label='Interpolated Signal')
plt.legend()
plt.grid(True)
plt.title('Signal with Interpolation')
plt.show()


start2 = time.time()
interpolator = np.interp(t,t_clean, x_clean)
end2 = time.time()
print('time=' , end2-start2)


plt.figure()
plt.plot(t, x, '-b', label='Original with Spikes')
plt.plot(t, interpolator, '-r', label='Interpolated')
plt.legend()
plt.grid(True)
plt.title('Signal with Interpolation')
plt.show()


