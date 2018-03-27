# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 16:18:47 2018

@author: gidelang

Mobility fitting 
requires numpy 1.13

"""
import numpy as np
from .fitter import fit



e_ = 1.60217662e-19
G0 = 7.7480917310e-5

def step(x, rate):
    return 1-1./(np.exp((x)/rate)+ 1)


def pinch_off(V_g, V_th, R_series, L, C_g, mu):
    switch = np.heaviside(V_g - V_th, 0.)#
    G = 1e-4*mu*C_g*(V_g - V_th)/L**2*switch
    
    G_series = 1./R_series
    
    return G*G_series/(G + G_series)/G0

def find_V_th_and_R_series(V_gs, Gs):
    '''
    Find V_th as the point at which conductance (Gs) > 5*noise 
    estimated from the first 10 points of the pinch-off trace
    '''
    if np.argmin(V_gs) > 0:
        V_gs = V_gs[::-1] # reverse sweep
        Gs = Gs[::-1]
    noise = np.std(Gs[:10])
    return V_gs[Gs > 5*noise][0], 1./(G0*Gs[-1])

def fit_pinchoff(V_gs, Gs, C_g, L, mu_init = 5e3):
    '''
    Fits pinch off curves using standard mobility 
    args: 
        V_gs = Gate voltages
        Gs = conductance in units of G0
        C_g = estimated gate capacitance in F
        L = Length of the wire in m
    
    returns:
        fit_report
    '''
    V_th, R_series = find_V_th_and_R_series(V_gs, Gs)
    p0 = (V_gs, V_th, R_series, L, C_g, mu_init)
    pinch_off_model, pars = fit.make_model(pinch_off,  p0 = p0)
    pars['L'].vary = False
    pars['C_g'].vary = False
    #pars['mu'].
    result = fit.fit(pars, Gs, pinch_off_model)
    return pars, result, pinch_off_model
    
    
def generate_sample_data(V_gs, V_th, R_series, L, C_g, mu):    
    return pinch_off(V_gs, V_th, R_series, L, C_g, mu) + 0.02*np.random.randn(len(V_gs))
    

       
def get_peak_dG_dV(V_gs, Gs):
    return np.max(np.diff(Gs)/np.diff(V_gs))




 