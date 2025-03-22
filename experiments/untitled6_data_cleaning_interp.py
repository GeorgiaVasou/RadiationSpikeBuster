# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:56:44 2024

@author: vasou
"""
import numpy as np
import matplotlib.pyplot as plt
import interp1d as inte

# Create test signal
N = 1000
nSpikes = 20
t = np.arange(N)
x = np.sin(2 * np.pi * t / 500) + 0.2 * np.random.randn(N)

plt.figure()
plt.plot(t, x, '-k', label='Original Signal')
plt.grid(True)
plt.title('Original Signal')
plt.legend()

# Add spikes
spike_inds = np.random.permutation(np.arange(N))[:nSpikes]
x[spike_inds] = x[spike_inds] + 3

plt.figure()
plt.plot(t, x, '-b', label='Signal with Spikes')
plt.grid(True)
plt.title('Signal with Spikes')
plt.legend()

# Spike detection
window_length = 20
multiplier = 3
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

# Time values at detected spikes
tq = t[detected_spikes]
print("Detected spikes:", detected_spikes)

# Interpolation of detected spikes using custom interp1d
S = inte.interp1d(t, x, tq)
print("Interpolated values at detected spikes:", S)

# Replace the spikes with interpolated values
x_new = np.copy(x)
for i, spike in enumerate(detected_spikes):
    x_new[spike] = S[i]

# Plot the new signal
plt.figure()
plt.plot(t, x, '-b', label='Signal with Spikes')
plt.plot(t, x_new, '-r', label='Signal after Interpolation')
plt.scatter(tq, S, color='green', label='Interpolated Points')
plt.grid(True)
plt.title('Signal after Interpolation')
plt.legend()
plt.show()