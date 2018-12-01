import numpy as np
import scipy as sp
from .fitter import fit
import matplotlib.pyplot as plt

def demod_noref(sig, dIQ, kernel):    
    IF_per = len(dIQ)
    shp = list(sig.shape)
    nw_shp = tuple(shp[:-1] + [int(shp[-1]/IF_per), IF_per])
    rd = sig.reshape(nw_shp)
    rd =  rd * dIQ
    demod_dat = np.average(rd, axis=-1) 
     
    return 2 * average_data(demod_dat, kernel)

def demod_ref(sig, reference, dIQ, kernel): 
    '''
    sig is ch A on Alazar (through fridge)
    reference is ch B on Alazar (fridge bypassed)
    '''
    
    IF_per = len(dIQ)
    shp = list(sig.shape)
    nw_shp = tuple(shp[:-1] + [int(shp[-1]/IF_per), IF_per])
    rd = sig.reshape(nw_shp)
    ref = reference.reshape(nw_shp)
    ref_dem = ref * dIQ
    ref_angles = np.angle(np.average(ref_dem, axis=-1))  
    rd =  rd * dIQ
    demod_dat = np.average(rd,axis = -1)*np.exp(-1.j*ref_angles) # only correcting for phase noise/drifts
   
    
    return 2 * average_data(demod_dat, kernel)


def demod(sig, dIQ, kernel, reference=None):
    if reference is not None:
        return demod_ref(sig, reference, dIQ, kernel)
    else:
        return demod_noref(sig, dIQ, kernel)

def average_data(data, window):
    if window ==1:
        return data
    else:
        shp = list(data.shape)
        nw_shp = shp[:-1] + [int(shp[-1]/window), window]
        return np.average(data.reshape(nw_shp), axis=-1)

def demod_ref_gen(sig, reference, dIQ, kernel): 
    '''
    sig is ch A on Alazar (through fridge)
    reference is ch B on Alazar (fridge bypassed)
    '''
    
    IF_per = len(dIQ)
    shp = list(sig.shape)
    nw_shp = tuple(shp[:-1] + [int(shp[-1]/IF_per), IF_per])
    rd = sig.reshape(nw_shp)
    ref = reference.reshape(nw_shp)
    ref_dem = ref * dIQ
    ref_angles = np.angle(np.average(ref_dem,axis=-1))  
    rd =  rd * dIQ
    demod_dat = np.average(rd,axis=-1)*np.exp(-1.j*ref_angles) # only correcting for phase noise/drifts
   
    
    return 2 * average_data(demod_dat, kernel)


def IQangle(data):
    I = np.real(data)
    Q = np.imag(data)
    Cov = np.cov(I,Q)
    A = sp.linalg.eig(Cov)
    eigvecs = A[1]
    if A[0][1]>A[0][0]:
        eigvec1 = eigvecs[:,0]
    else:
        eigvec1 = eigvecs[:,1]
    theta = np.arctan(eigvec1[0]/eigvec1[1])
    return theta

def IQrotate(data, theta):
    return data*np.exp(1.j*theta)

def PSD(ts, ys):
    '''
    returns the PSD of a timeseries and freuency axis
    '''
    ys_fft = np.abs(np.fft.fft(ys))**2
    
    dt = np.diff(ts)[0]
    fmax = 1. / (dt)
    df = 1. / (len(ts)*dt)
    pts_fft = len(ts) / 2
    fs1 = np.linspace(0, fmax / 2.-df, pts_fft)
    fs2 = np.linspace(fmax / 2., df, pts_fft)
    fs = np.append(fs1, -fs2)
    #print(len(fs), len(ys_fft))
    return fs, ys_fft

def PSDs(ts, ys_array):
    '''
    returns the averaged PSD of an array of timeseries with freuency axis
    '''
    shp = ys_array.shape
    ys_fft = 0.*ys_array[0,:].real
    for ys in ys_array:
        
        fs, ys_fft_i = PSD(ts, ys) 
        ys_fft += ys_fft_i/shp[0]
    return fs, ys_fft

def threshold_shots(sorted_shots, xvals, xlab='', title='',iq_centers=None, max_dist=np.inf):
    
    # sorted shot data = [sweep_value, rep]
    shp = np.shape(sorted_shots)
    print('npoints: ',shp[1])
    IQ_pairs = sorted_shots.flatten()
    if iq_centers == None:
        nstates = 2
        theta = IQrotate_hist(IQ_pairs )
        print('theta: ',theta)
        IQ_pairs_rot = np.exp(1.0j*theta)*IQ_pairs
        avg_IQ = np.average(IQ_pairs_rot)
        
        states = IQ_pairs_rot.real > avg_IQ 
        selector = states == states #(select all values)
        
        
    else:
        nstates = len(iq_centers)
        distances = np.zeros([len(iq_centers) , len(IQ_pairs)])
        centers = iq_centers
        for kk, center in enumerate(centers):
            distances[kk,:] = np.abs(IQ_pairs - center)
        states = np.argmin(distances, axis = 0)
        mindistances = np.min(distances, axis = 0) 
        selector = mindistances < max_dist #only select those that are at some distance from the blob

    states.shape = shp   
    selector.shape = shp   
        
    Ps = np.zeros([nstates, shp[1]])
    labels = [str(kk) for kk in range(nstates)]
    plt.figure()
    for state in range(nstates):
        #Ps[state,:] = np.average((states == state), axis = 0)
        Ps[state,:] = np.average((states==state) , axis=0)#/np.sum(selector, axis = 0)
        plt.plot(xvals, Ps[state,:], label=labels[state])
    
    plt.ylabel('Probability')
    plt.xlabel(xlab)
    plt.legend()
    plt.title(title)
    
    
    return Ps.transpose(), states
    
def QPP_Lorentzian(f, Gamma, a, b):
    return a*(4*Gamma/((2*Gamma)**2+(2*np.pi*f)**2))+b
    
def estimate_QPP_pars(fs, ys):
    may = np.max(ys)
    miy = np.min(ys)
    roll_off = fs[ys - miy < (may-miy)/2.][0]
    a = may - miy
    b = miy
    Gamma = np.pi*roll_off
    return fs, Gamma, a, b



def fit_QPP(fs, PSD):
    '''
    fit PSD to QPP spectrum  
    '''
    
    p0 = estimate_QPP_pars(fs, PSD)
    QPP_model, pars = fit.make_model(QPP_Lorentzian,  p0=tuple(p0))    
    pars['f'].vary = False
    pars['Gamma'].min = 0
    pars.add('T_qp', expr='1/Gamma')
    result,  fitted_values = fit.fit(pars, PSD, QPP_model)
    pars['T_qp'].vary = True
    fit_report = fit.print_fitres(pars)
    return pars, result, QPP_model, fit_report
    
    
