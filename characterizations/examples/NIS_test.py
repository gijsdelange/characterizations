from characterizations import NIS
import matplotlib.pyplot as plt
import numpy as np

def test_fit():    
    Vs = np.linspace(-400e-6, 400e-6, 201)+1e-10
    V_Delta = 150e-6
    V_Gamma = 10e-6
    
    simys = dIdV(Vs, V_Delta, V_Gamma, 2.) + 0.1*np.random.randn(len(Vs))
    
    plt.figure('DOS')
    plt.clf()
    
    plt.plot(Vs, simys, 'o')
    pars, result, dIdV_model, fit_report = NIS.fit_dIdV(Vs, simys)
    simfit = dIdV_model(pars)
    plt.plot(Vs, simfit)
    plt.yscale('log')