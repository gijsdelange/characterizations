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

    
    S21 = A*(1.- np.abs(Q)/np.abs(Qe)*np.exp(1.j*theta)/(1.+2.j*np.abs(Q)*(f-f0)/f0))
    return S21
def Qc_calc(Qeext, theta_ext):
    return (np.real(1./(Qe*np.exp(1.j*theta))))**-1
    
        
def estimate_hanger_pars(xdat, ydat):
    s21 = np.abs(ydat)
    A = (np.abs(s21[0]) + np.abs(s21[-1]))/2.
    f0 = xdat[np.argmin(s21)]
    s21min = np.abs(np.min(s21)/A)
    #print('s21min: ',s21min)
    FWHM = np.abs(xdat[-1] - xdat[0])/50.
    Ql = f0/FWHM
    Qi = Ql/(s21min)
    Qc = Ql/np.abs(1-s21min)
    
    #print('Qc: ', Qc)
    phi_0 = np.angle(ydat[0])
    avg_ang = np.average( np.diff(np.angle(ydat)) )
    avg_ang = np.diff(np.angle(ydat))
    
    phi_v = -np.average((avg_ang[np.abs(avg_ang)<np.pi/4]))/(np.diff(xdat)[0])
        
    p0 = [xdat, f0, Ql, Qc, A, 0., phi_v, phi_0, 0.]
    #print(p0, A, s21min)
    return p0

def fit_hanger(xdat, ydat, slope = True, p0 = None, **kw):
    # initial guess
    if p0 is None:
        p0 = estimate_hanger_pars(xdat, ydat)
    hanger_model, pars = fit.make_model(hanger_S21_sloped,  p0 = tuple(p0))
    pars['f'].vary = False
    if not slope:
        pars['df'].vary = False
    pars['Q'].min = 0
    pars['Qe'].min = 0
    pars.add('Qi', expr = '1/(1/Q-abs(1/Qe*cos(theta)))')
    pars.add('Qc', expr = 'abs(Qe/cos(theta))')
    
    result,  fitted_values= fit.fit(pars, ydat, hanger_model)
        
    #print(result)
    #print(fitted_values)
    pars['Qc'].vary = True
    pars['Qi'].vary = True
    fit_report = fit.print_fitres(pars, **kw)
    return pars, result, hanger_model, fit_report

def plot_results(s21m, params, hanger_model, fit_report, results, resolution = 0.05, figlab = ''):
    
    pars = params#.copy()
    
    results.minimize()
    to_MHz = 10**((np.floor(np.log10(pars['f0'].value)))-3)
    #print(pars)
    #plt.close('all')
    fig = plt.figure('Hanger_fit: ' + figlab, figsize = (10,4))
    plt.clf()
    plt.subplot(241)
    plt.plot(s21m.real, s21m.imag, '.')
    df = resolution*pars['f0'].value/ pars['Q'].value
    dfdat = np.diff(pars['f'].value)[0]
    if df > dfdat:
        df = dfdat
        
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