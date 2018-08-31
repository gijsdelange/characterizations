import numpy as np
from .fitter import fit
import matplotlib.pyplot as plt

def hanger_S21_sloped(f, f0, Q, Qe, A, theta, phi_v, phi_0, df):
    '''
    ''' 
    S21=(1.+df*(f-f0)/f0)*np.exp(1.j*(phi_v*(f-f[0])+phi_0))*hanger_S21(f,f0,Q,Qe,A,theta)
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

def plot_results(s21m, params, hanger_model, fit_report, resolution = 0.05):
    pars = params.copy()
    to_MHz = 10**((np.floor(np.log10(pars['f0'].value)))-3)
    
    #plt.close('all')
    fig = plt.figure('Hanger')
    plt.clf()
    plt.subplot(241)
    plt.plot(s21m.real, s21m.imag, '.')
    df = resolution*pars['f0'].value/ pars['Q'].value
    fs = pars['f'].value
    f_ran = (fs[-1] - fs[0])
    fs_dat_MHz = pars['f'].value/to_MHz
    pars['f'].value = np.linspace(fs[0], fs[-1], f_ran/df)
    f_MHz = pars['f'].value/to_MHz
    
    s21fit = hanger_model(pars)
    plt.plot(s21fit.real, s21fit.imag, '-r')
    max_val = np.max(np.abs(s21m))
    xy_ran = -1.5*max_val, 1.5*max_val
    plt.xlim(xy_ran)
    plt.ylim(xy_ran)
    plt.xlabel(r'Re[$S_{21}$]')
    plt.ylabel(r'Im[$S_{21}$]')
    
    
    plt.subplot(242)
    plt.plot(fs_dat_MHz, np.abs(s21m)**2, '.')
    plt.plot(f_MHz, np.abs(s21fit)**2, '-r')
    plt.ylabel('$|S_{21}|^2$')
    plt.xlabel('frequency (MHz)')
    
    
    plt.subplot(245)
    plt.plot(fs_dat_MHz, s21m.real, '.b')
    plt.plot(fs_dat_MHz, s21m.imag, '.r')
    plt.plot(f_MHz, s21fit.real, '-b', label = 'Re[$S_21$]')
    plt.plot(f_MHz, s21fit.imag, '-r', label = 'Im[$S_21$]')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('amplitude (V/V)')
    
    plt.subplot(246)
    plt.plot(fs_dat_MHz, np.angle(s21m, deg = True), '.')
    plt.plot(f_MHz, np.angle(s21fit, deg = True), '-r')
    plt.xlabel('frequency (MHz)')
    plt.ylabel('pahse (deg.)')
    
    fig.text(0.52,0.15, fit_report)
    plt.tight_layout()

def fit_hanger(xdat, ydat, slope = True):
    # initial guess
    p0 = estimate_hanger_pars(xdat, ydat)
    hanger_model, pars = fit.make_model(hanger_S21_sloped,  p0 = p0)
    pars['f'].vary = False
    if not slope:
        pars['df'].vary = False
    result,  fitted_values= fit.fit(pars, ydat, hanger_model)
    print(result)
    print(fitted_values)
    fit_report = fit.print_fitres(pars)
    return pars, result, hanger_model, fit_report