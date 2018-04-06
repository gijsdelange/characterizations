import numpy  as np
import sys
import utils.fitter.fit as fit
import utils.dic_data.data as dd
#import utils.plotting as plot
import time
import matplotlib.pyplot as plt
from PyQt5 import QtGui
import qcodes as qc
from qcodes import Instrument
from utils.dic_data import dic2hdf5


ERASE_LINE = 500*'\b'
SENSES = [0.5, 1, 0.2, 2e-09, 5e-05, 1e-08, 2e-08, 0.0001, 0.05, 0.0002, 5e-06, 5e-08, 1e-06, 0.1, 1e-09, 5e-09, 2e-05, 0.02, 5e-07, 1e-05, 0.01, 2e-07, 0.005, 1e-07, 2e-06, 0.002, 0.001, 0.0005]
SENSES = np.array(SENSES)
SENSES.sort()
DEFAULT_Fs = [1000, 500e3]

class dummy_lockin(Instrument):
    def __init__(self, name,  address):
        super().__init__(name)
        self.name = name
        self.address = address
        self._amplitude = 1e-3
        self._frequency = 100
        self._sine_outdc = -5
        self._time_constant = 1e-3
        self._sensitivity = 0.001
        self.C_l = 4e-12
        self.C_h = 20e-12
        self.V_th = -0.3
        self.dV_th = 0.4
        self.args = (1.25e8, 0, 0, 1e3, 500e-12)
    
    def call_parameter(self, par, arg):
        if arg == ():
            ret = eval('self._{}'.format(par))
            return ret
        else:
            exec('self._{} = arg[0]'.format(par))
            return True
            
    def sensitivity(self, *arg):
        return self.call_parameter('sensitivity', arg)
    def frequency(self, *arg):
        return self.call_parameter('frequency', arg)
    def amplitude(self, *arg):
        return self.call_parameter('amplitude', arg)
    def time_constant(self, *arg):
        return self.call_parameter('time_constant' , arg)
    def sine_outdc(self, *arg):
        return self.call_parameter('sine_outdc', arg)
    def c_val(self, Vg, f):
        
        V_th = -0.3+0*0.1*f/500e3
        C_out = self.C_l + (self.C_h-self.C_l)/(1+np.exp(-(Vg - V_th)/0.4))
        return C_out
    def X(self):
        f = self.frequency()
        Z = V_out_cap_model(f, self.amplitude(), self.c_val(self.sine_outdc(), f),*self.args)
        return Z.real
    def Y(self):
        f = self.frequency()
        Z = V_out_cap_model(f, self.amplitude(), self.c_val(self.sine_outdc(), f),*self.args)
        return Z.imag
    def R(self):
        f = self.frequency()
        Z = V_out_cap_model(f, self.amplitude(), self.c_val(self.sine_outdc(), f),*self.args)
        return np.abs(Z)    
    
        
s= dummy_lockin('dummy', 'addr')
s.frequency(1)
station = qc.Station(s)

def frange(fstart, fstop, npoints):
    return np.append(fstart, np.linspace(fstop/(npoints-1), fstop, (npoints-1)))

class CV_measurement:
    def __init__(self, name, Vgs, lockin, fs = DEFAULT_Fs, npoints = 6, station = qc.Station()):
        self.Vac = 1e-3
        self.lockin = lockin
        self.name = name
        self.Vgs = Vgs
        self.fs = np.linspace(1e3,501e3,11)#frange(fs[0], fs[1], npoints)
        self.delay = 5*lockin.time_constant()
        self.C_p = 500e-12
        self.lockin.amplitude(self.Vac)
        self.Rm = 1e3
        self.cm = plt.cm.Spectral_r
        self.cmap = plt.cm.Spectral_r
        self.recorded_values = ['C_Dp (pF)', 
                                'R_Dp (MOhm)',
                                'tand',
                                'C_Ds (pF)',
                                'R_Ds (kOhm)',
                                ]
        self.init_data()
        
    def init_data(self):
        self.data = dd.init_dic_data(self.name)
        ds = {'name': self.name,
               'V_ac': self.Vac,
               'settings': station.snapshot(),
               'data': {'V_g': self.Vgs,
                       'V_m (V)':  [],
                       'f (Hz)': self.fs,
                       }
               }
        for v in self.recorded_values:
            ds['data'][v] = []
        for d in ds.keys():
            self.data[d] = ds[d]
            
    def set_sens(self):
        sens = SENSES[(SENSES > 1.4*self.lockin.R()).tolist()][0]
        self.lockin.sensitivity(sens)
    def measure_Vout_vs_f(self, fs, Vg = 0):
        lockin = self.lockin
        ramp_lockin_dc(Vg, lockin)
        time.sleep(0.5)
        self.set_sens()
        lockin.frequency(fs[0])
        time.sleep(4*self.delay)
        Vs = np.zeros(len(fs), dtype = np.complex128)
        for kk,f in enumerate(fs):
            lockin.frequency(f)
            time.sleep(self.delay)
            Vs[kk] = lockin.X() + 1.j*lockin.Y()
        self.set_sens()
        return np.array(Vs)
    def measure_CV(self, Vg):
        Vouts = self.measure_Vout_vs_f(self.fs, Vg)
        self.Vouts[self.sweep_ind, :] = Vouts
        C_Ds, Rs, tand = get_cap(Vouts, self.fs, self.C_p, self.Vac, self.Rm, model = 'Rs') 
        C_Dp, Rp, tand = get_cap(Vouts, self.fs, self.C_p, self.Vac, self.Rm, model = 'Rp')
        return C_Dp, Rp, tand, C_Ds, Rs
    def calibrate(self):
        fs = np.linspace(100,500e3, 101)
        Vcals = self.measure_Vout_vs_f(fs, min(self.Vgs))
        pars, cap_model = get_C_p(Vcals, fs, self.Vac)
        Vfit = cap_model(pars)
        self.data['calibration'] = {'data': Vcals,
                                     'fs': fs,
                                     'fit': Vfit,
                                     'fitpars': [pars['C_D'].value, pars['C_p'].value],
                                     'fiterrs': [pars['C_D'].stderr, pars['C_p'].stderr]}
        self.C_p = pars['C_p'].value
        C_D = pars['C_D'].value
        plt.figure('calibration')
        plt.clf()
        plt.plot(Vcals.real, Vcals.imag, '.')
        plt.plot(Vfit.real, Vfit.imag, label = 'fit, C_p = {:.2f} pF, C_D = {:.2f} pF'.format(self.C_p*1e12,C_D*1e12 ))
        plt.legend()
        mes = '%s @ V_g = %s'%(self.name, self.lockin.sine_outdc())
        plt.title(mes)
    
    
    def measure_CVs(self, monitor = True):
        
        d = self.data['data']
        self.figname = self.name + ': '+ self.data['filepath']
        if monitor:
            fig = plt.figure(self.figname, figsize = (8,7))
            plt.clf()
            axs =[]
            for kk, key in enumerate(self.recorded_values):
                ax = plt.subplot(321+kk)
                axs += [ax]
                ax.set_ylabel(key)
                plt.tight_layout()
            self.axs = axs
        self.Vouts = np.zeros([len(self.Vgs), len(self.fs)], dtype = np.complex128)
        for ll,Vg in enumerate(self.Vgs):
            self.sweep_ind = ll
            vals = self.measure_CV(Vg) #C_Dp, Rp, tand, C_Ds, Rs
            
            msg = 'V_gate = {} V, {} of {}'.format(Vg, ll, len(self.Vgs))
            msg += ' '*(150-len(msg))
            sys.stdout.write(ERASE_LINE + msg)
            
            for kk, key in enumerate(self.recorded_values):
                d[key] += [vals[kk]]
            if monitor:    
                for kk, key in enumerate(self.recorded_values):                  
                    dat = np.array(d[key])
                    for nn, f in enumerate(self.fs):
                        #print(dat.shape)
                        kw = {'color': self.cm(f/max(self.fs))}
                        if ll ==0 and kk == 3:
                            kw ['label'] = '{:.1f} kHz'.format(f/1000)
                        else:
                            pass
                            
                        axs[kk].plot(self.Vgs[:ll+1], dat[:ll+1, nn], 
                                       **kw
                                       )                        
                    plt.gca().relim(visible_only=True)
                    plt.gca().autoscale_view();
                axs[3].legend()
                axs[0].set_title(self.name +': '+ self.data['filepath'])
                plt.tight_layout()
                fig.canvas.draw()
                QtGui.QGuiApplication.processEvents()
        for key in self.recorded_values:
            d[key] = np.array(d[key])           
        dic2hdf5.save_dict_to_hdf5(self.data, self.data['filepath']) 

def ramp_lockin_dc(val, lockin):
    curval = lockin.sine_outdc()
    dv = 0.005
    nvals = np.abs((val - curval))/dv 
    vals = np.linspace(curval, val, int(nvals))
    for nval in vals:
        lockin.sine_outdc(nval)

def get_C_p(Vcals, fs, V_ac, C_D = 1e-12, C_off = 0, R_m = 1e3, C_p = 5e-10, ax = None):
    
    R_Dp = np.real(Z_D(V_ac, Vcals[fs.argmin()].real, R_m))
    
    R_Ds = 0.0
    cap_model, pars = fit.make_model(V_out_cap_model, 
                                     p0 = (fs, V_ac, C_D, R_Dp, R_Ds, C_off, R_m, C_p))
    pars['f'].vary = 0
    pars['V_ac'].vary = 0
    pars['R_Ds'].vary = 0
    pars['R_Dp'].vary = 1
    pars['C_D'].vary = 1
    pars['R_m'].vary = 0
    pars['C_off'].vary = 0
    #pars['L_i'].vary = False
    pars['C_p'].vary = 1
    fit.fit(pars, Vcals, cap_model)
    fit.print_fitres(pars)
    return pars, cap_model

def V_out_cap_model(f, V_ac, C_D, R_Dp, R_Ds, C_off, R_m, C_p):
    Z_p = 1/(2.j*np.pi*f*C_p)
    Z_C_D = 1./(2.j*np.pi*f*(C_D+C_off))
    Z_D = R_Dp*Z_C_D/(R_Dp + Z_C_D)+R_Ds
    return V_out(V_ac, Z_D, R_m, Z_p)

def V_out(V_ac, Z_D, R_m, Z_p):    
    Z_r = R_m*Z_p/(R_m + Z_p)
    V_out = Z_r/(Z_D + Z_r)*V_ac
    return V_out

def Z_p(R_m, C_p, f):
    return R_m/(1+1.j*2*np.pi*f*C_p*R_m)

def Z_D(V_ac, Vout, Z_p):
    return Z_p*(V_ac - Vout)/Vout
    
def get_cap(Vout, f, C_p, V_ac = 1e-3, R_m= 1000., model = 'Rs'):
    '''    
    '''
    w = 2*np.pi*f
    Z_detection = Z_p(R_m, C_p, f)
    Z_Dm = Z_D(V_ac, Vout, Z_detection)
    X,Y = Z_Dm.real, Z_Dm.imag
    if model == 'Rs':
        Rs = X
        C_D = -1/(2*np.pi*f*Y)
        return C_D*1e12, Rs/1e3, -X/Y
    else:
        C_D = -Y/(X**2 + Y**2)/w
        Rp = np.abs((X**2+Y**2)/X)
        return C_D*1e12, Rp/1e6, -X/Y

if __name__ == '__main__':
    Vgs_sim = np.linspace(-5,5,41)  
    
    cvm = CV_measurement('test1', Vgs_sim, s) 
    cvm.calibrate()
    #stop
    cvm.measure_CVs()
    