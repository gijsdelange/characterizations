# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:01:55 2018

@author: gidelang
"""

import numpy as np
from .fitter import fit
import matplotlib.pyplot as plt

def BCS(E, Delta):
    return np.sign(E)*((E+0.j)/np.sqrt(E**2 - Delta**2+0.j)).real

def Dynes(E, Delta, Gamma):
    newE = E+1.j*Gamma
    
    return BCS(newE, Delta)

def dIdV(Vbias, V_Delta, V_Gamma, GN):
    return GN*Dynes(Vbias, V_Delta, V_Gamma).real


def estimate_dIdV_pars(Vbiass, ys):
    GN = np.average(ys[:10])
    ind = int(len(Vbiass)/4)
    print('index: ', ind, Vbiass[ind] )
    V_Delta = np.abs(Vbiass[ind])
    V_Gamma = 0.01*V_Delta
    return [Vbiass, V_Delta, V_Gamma, GN]

def fit_dIdV(Vbiass, Gs):
    '''
    Vbias in Volts and ys in conductance
    '''
    
    p0 = estimate_dIdV_pars(Vbiass, Gs)
    dIdV_model, pars = fit.make_model(dIdV,  p0 = tuple(p0))    
    pars['Vbias'].vary = False
    pars['V_Delta'].min = 0
    pars['V_Gamma'].min = 1e-6 # minimum of 1 uV
    pars.add('Hardness', expr = 'V_Gamma/sqrt(V_Gamma**2+V_Delta**2)')
    
    result,  fitted_values= fit.fit(pars, Gs, dIdV_model)
    pars['Hardness'].vary = True
    fit_report = fit.print_fitres(pars)
    return pars, result, dIdV_model, fit_report


    
        
def test_fit():    
    Vs = np.linspace(-400e-6, 400e-6, 201)+1e-10
    V_Delta = 150e-6
    V_Gamma = 10e-6
    
    simys = dIdV(Vs, V_Delta, V_Gamma, 2.) + 0.05*np.random.randn(len(Vs))
    
    plt.figure('DOS')
    plt.clf()
    
    plt.plot(Vs, simys, 'o')
    pars, result, dIdV_model, fit_report = fit_dIdV(Vs, simys)
    simfit = dIdV_model(pars)
    plt.plot(Vs, simfit)
