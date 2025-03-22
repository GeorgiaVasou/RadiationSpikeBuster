# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:11:35 2024

@author: vasou
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:21:12 2024

@author: vasou
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import data_cleaning

import time
import scipy.optimize as sc

start = time.time()

N = 1000
nSpikes = 20

# Δημιουργία αρχικού σήματος
t = np.arange(N)
x = np.sin(2 * np.pi * t / 500) + 0.2 * np.random.randn(N)

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

# Ανίχνευση spikes και διαγραφή
detected_inds = data_cleaning.despiking_median_iqr(x, 20, 1.5)

# Διαγραφή των ανιχνευμένων indices από t και x
t_clean = np.delete(t, detected_inds)
x_clean = np.delete(x, detected_inds)

# Εκτέλεση interpolation
interpolator = np.interp(t, t_clean, x_clean)

plt.figure()
plt.plot(t, x, '-b', label='Original with Spikes')
plt.plot(t, interpolator, '-r', label='Interpolated')
plt.legend()
plt.grid(True)
plt.title('Signal with Interpolation')
plt.show()

end = time.time()
print('time=', end - start)

#%%

plt.figure()
plt.plot(t, interpolator, '.k')

# Ορισμός συνάρτησης για το fitting
def fun(x, a, b):
    return a * x + b

# Πλοκή παραβολής
y_par = fun(t, 10, 1)

plt.plot(t, y_par, '-b')

#%% Fitting
# Καθορισμός αρχικών τιμών και ορίων για το fitting
initial_guesses = [0.01, 0.01]
lower_bounds = [-100, -100]
upper_bounds = [150, 100]

# Εκτέλεση της προσαρμογής καμπύλης (curve fitting)
fit_params = sc.curve_fit(fun, t, interpolator, 
                          initial_guesses, check_finite=True, 
                          bounds=(lower_bounds, upper_bounds))[0]

# Πλοκή της προσαρμοσμένης καμπύλης
y_fit = fun(t, *fit_params)
plt.plot(t, y_fit, '-r')

# Υπολογισμός του σφάλματος για τις δύο καμπύλες
MAE_blue = np.mean(np.abs(y_par - interpolator))
MAE_red = np.mean(np.abs(y_fit - interpolator))
print(f'The Mean Absolute Error of the new result (red line) is {MAE_red:.3f} which has to be lower than the previous one (blue line) which was {MAE_blue:.3f}')

#%% Task 2 - Smoothing με παράθυρο και fitting

N = len(interpolator)
smoothed_x = np.copy(interpolator)

w = 15

for i in range(w, N - w):
    x_w = interpolator[i - w:i + w]
    t_w = t[i - w:i + w]
    fit_params1 = sc.curve_fit(fun, t_w, x_w, 
                               initial_guesses, check_finite=True, 
                               bounds=(lower_bounds, upper_bounds))[0]
    smoothed_x[i] = fun(t[i], *fit_params1)

# Plot with better formatting and no titles

plt.figure(figsize=(10, 6))

# Plot the smoothed data
plt.subplot(2, 1, 1)
plt.plot(t, smoothed_x, '-b', label='Smoothed Data')

plt.grid(True)

# Plot the residuals
plt.subplot(2, 1, 2)
plt.plot(t, interpolator - smoothed_x, '-r', label='Residuals')

plt.grid(True)

# Adjust the layout
plt.tight_layout()
plt.show()
