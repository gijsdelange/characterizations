import numpy as np
from .fitter import fit

def hanger_S21_sloped(f, f0, Q, Qe, A, theta, phi_v, phi_0, df):
    '''
    ''' 
    S21=(1.+df*(f-f0)/f0)*np.exp(1.j*(phi_v*f+phi_0-phi_v*f[0]))*hanger_S21(f,f0,Q,Qe,A,theta)
    return S21
    
def hanger_S21(f, f0, Q, Qe, A, theta):
    '''
    THis is THE hanger fitting function for fitting normalized hanger  (no offset)
    x,x0,Q,Qe,A,theta
    S21 = A*(1-Q/Q_c*exp(1.j*theta)/(1+2.j*Q(x-x0)/x0)
    where 1/Qi = 1/Q - Re{exp(1.j*theta)/Qc}
    '''

    
    S21 = A*(1.-Q/np.abs(Qe)*np.exp(1.j*theta)/(1.+2.j*Q*(f-f0)/f0))
    return S21
    
def estimate_hanger_pars(xdat, ydat):
    s21 = np.abs(ydat)
    A = (np.abs(s21[0]) + np.abs(s21[-1]))/2.
    f0 = xdat[np.argmin(s21)]
    s21min = np.min(s21)/A
    FWHM = np.abs(xdat[-1] - xdat[0])/50.
    Ql = f0/FWHM
    Qi = Ql/s21min
    Qc = Qi/(1-s21min)
    phi_0 = np.angle(ydat[0])
    phi_v = np.average( np.diff(np.angle(ydat[:11])) )/(np.diff(xdat)[0])
        
    p0 = (xdat, f0, Ql, Qc, A, 0., phi_v, phi_0, 0.)
    print(p0, A, s21min)
    return p0

def fit_hanger(xdat, ydat, slope = True):
    # initial guess
    p0 = estimate_hanger_pars(xdat, ydat)
    hanger_model, pars = fit.make_model(hanger_S21_sloped,  p0 = p0)
    pars['f'].vary = False
    if not slope:
        pars['df'].vary = False
    result = fit.fit(pars, ydat, hanger_model)
    return pars, result, hanger_model
    