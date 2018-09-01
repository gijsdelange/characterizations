# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 21:37:16 2018

@author: gidelang
"""
import characterizations
import characterizations.resonators as res

import matplotlib.pyplot as plt
import numpy as np

# generate_fake_data
hang_s21 = res.hanger_S21_sloped
npoints = 1001
units = 1e9 #Hz
f = np.linspace(5,5.2, npoints)*units
f0 = 5.085*units
Q = 5000.
Qe = 6000.
A = 10. 
snr = 500000.
theta = np.pi/6
phi_v = 2000*np.pi/1001/units
phi_0 = -np.pi*4./5
df = 5./units
noise = A/np.sqrt(snr)
p0 = (f, f0, Q, Qe, A, theta, phi_v, phi_0, df)
s21meas = hang_s21(*p0) + noise*(np.random.randn(len(f)) + 1.j*np.random.randn(len(f))) 

# the actual fit
pars, results, hanger_model, fit_report = res.fit_hanger(f, s21meas)

# plot report
res.plot_results(s21meas, pars, hanger_model, fit_report, results)

