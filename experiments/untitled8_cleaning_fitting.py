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

start = time.time()

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

detected_inds=data_cleaning.despiking_median_iqr(x,20,1.5)

# Διαγραφή των ανιχνευμένων indices από t και x
t_clean = np.delete(t, detected_inds)
x_clean = np.delete(x, detected_inds)


# Εκτέλεση interpolation
interpolator = np.interp(t,t_clean, x_clean)

print(interpolator)

plt.figure()
plt.plot(t, x, '-b', label='Original with Spikes')
plt.plot(t, interpolator, '-r', label='Interpolated')
plt.legend()
plt.grid(True)
plt.title('Signal with Interpolation')
plt.show()

end = time.time()
print('time=' , end-start)
#%%

plt.figure()
plt.plot(t,interpolator, '.k')

# And now let's define a function, e.g. a parabola like y = y0 + a * (x - x0)**2
def fun(x,a,b):
    return  a*x+b

# We can plot the points of this, giving to the three parameters the values 
# x0 = 10, y0=1.1 and a=-0.015 (the minus is because we want it to extend 
# downwards, to match the data points of x)

y_par = fun(t, 10, 1)

plt.plot(t, y_par, '-b')

#%% Fitting
# But now let's say that we want to fit this parabolic function to the data so
# we can derive exactly the best values for the parameters x0, y0 and a, so 
# that the result of fun(t) will be as close as possible to the data points
#
# For this we will need the Optimize module from SciPy

import scipy.optimize as sc

# we must also define some initial values for the parameters (best guesses)
initial_guesses = [0.01, 0.01]
# the lower bound for each parameter
lower_bounds = [-100, -100]
# and the upper bound for each parameter
upper_bounds = [150, 100]
# e.g. if we want 'a' to be strictly negative, we will set its upper bound to 
# be 0 and use some large negative value as its lower bound, e.g. -100

# then we do the fit like so
fit_params = sc.curve_fit(fun, t, interpolator, 
                   initial_guesses, check_finite=True, 
                   bounds=(lower_bounds, upper_bounds))[0]

# The best parameters are saved in 'fit_params'. We can now plot the function 
# outputs, produced with these three parameters and see if they are indeed 
# better than our guess

y_fit = fun(t, *fit_params)
plt.plot(t, y_fit, '-r')

# Is this better? Let's calculate the mean absolute error to check
MAE_blue = np.mean(np.abs(y_par - interpolator))
MAE_red = np.mean(np.abs(y_fit - interpolator))
print('The Mean Absolute Error of the new result (red line) is %.3f which has to be lower than the previous one (blue line) which was %.3f'%(MAE_red, MAE_blue))



#%% Task 2
# We can use this method to "smooth" time series. Let's say we have a series 
# 'x'. We can read the elements of 'x' with a window of e.g. 5 points. So for 
# each x(i) we read data from x(i-2) up to x(i+2), then take these 5 points, fit 
# a parabola and replace the middle point x(i) with the one predicted by the 
# parabola that best fits these 5 data points. We can use this to find something
# like a moving average. 
# Can you try it in the previous example that we used for the despiking, to see
# what you'll get?
import scipy.optimize as sc

N= len(interpolator)
smoothed_x = np.copy(interpolator)
def fun(x,a,b):
    return  a*x+b
initial_guesses = [0.01, 0.01]

lower_bounds = [-100, -100]

upper_bounds = [150, 100]

w=15

for i in range(w,N-w):
  x_w= interpolator[i-w:i+w]
  t_w= t[i-w:i+w]
  fit_params1 = sc.curve_fit(fun, t_w, x_w, 
                     initial_guesses, check_finite=True, 
                     bounds=(lower_bounds, upper_bounds))[0]
  smoothed_x[i]=fun(t[i],*fit_params1)
plt.subplot(2, 1, 1)  
plt.plot(t, smoothed_x, '-b')
plt.subplot(2, 1, 2)
plt.plot(t, interpolator-smoothed_x,'-r')


print("Original x:", interpolator)
print("Smoothed x:", smoothed_x)

  
  
  
  



