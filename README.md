
# SpikeBuster: Preprocessing & Cleaning Satellite Particle Radiation Data

## Overview

This project was developed as part of a **3-month internship** focused on the preprocessing and cleaning of **particle radiation data from satellites operating in near-Earth geospace**.

The goal was to remove noise, spikes, and anomalies in satellite radiation measurements using a combination of:
- **Statistical spike detection methods:** Rolling window statistics, Median & IQR filtering, Z-score analysis.
- **Interpolation & curve fitting:** To reconstruct smooth, reliable radiation signals.
- **Visualization tools:** For comparison of raw, cleaned, and smoothed datasets.

---

## Features

### Signal Despiking Techniques:
- Rolling window mean and standard deviation.
- Overlapping window analysis.
- Median and Interquartile Range (IQR) detection.
- Z-score based outlier filtering.

### Interpolation Methods:
- NumPy interpolation.
- Custom linear interpolation (`interp1d`).

### Smoothing and Curve Fitting:
- Local curve fitting using sliding windows.
- Mean Absolute Error (MAE) comparison for fit evaluation.



## About This Project
This project was completed during my **3-month internship**, where I focused on cleaning and analyzing **particle radiation data from satellites in near-Earth geospace**, 

