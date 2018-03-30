import numpy  as np
import utils.fitter.fit as fit
import sys
import os
import matplotlib.pyplot as plt
from scipy.optimize import fsolve






def V_out_cap_model(f, V_ac, C_D, R_Dp, R_Ds, C_off, R_m, C_p):
    Z_p = 1/(2.j*np.pi*f*C_p)
    Z_C_D = 1./(2.j*np.pi*f*(C_D+C_off))
    Z_D = R_Dp*Z_C_D/(R_Dp + Z_C_D)+R_Ds
    return V_out(V_ac, Z_D, R_m, Z_p)

def cmplx2array(val):
    
    return np.array([np.real(val), np.imag(val)])

def zero_func_Rs(x, f, V_ac, R_Dp, C_off, R_m, C_p, Vout):
    V_out_test = V_out_cap_model(f, V_ac, x[0], R_Dp, np.abs(x[1]), C_off, R_m, C_p)
    return cmplx2array(V_out_test - Vout)

def zero_func_Rp(x, f, V_ac, R_Ds, C_off, R_m, C_p, Vout):
    V_out_test = V_out_cap_model(f, V_ac, x[0], np.abs(x[1]), R_Ds, C_off, R_m, C_p)
    return cmplx2array(V_out_test - Vout)

def I_detector(V_ac, Z_D, Z_i, Z_s):
    Z_2 = Z_i*Z_s/(Z_i + Z_s)
    Im = Z_2/(Z_D+Z_2)/Z_i*V_ac
    return Im


def I_d_cap_model(f, V_ac, R_D, C_D, R_i, L_i, C_s):
    
    return I_detector(V_ac, (1./R_D+(2.j*np.pi*f*C_D))**-1, R_i+2.j*np.pi*f*L_i, 1./(2.j*np.pi*f*C_s))

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
    Z_Dm = Z_D(V_ac, Vout, Z_p(R_m, C_p, f))
    X,Y = Z_Dm.real, Z_Dm.imag
    if model == 'Rs':
        Rs = np.abs(np.real(Z_Dm))
        C_D = -1/(2*np.pi*f*np.imag(Z_Dm))
        return C_D, Rs, -X/Y
    else:
        
        
        C_D = -Y/(X**2 + Y**2)/w
        Rp = np.abs((X**2+Y**2)/X)
        return C_D, Rp, -X/Y
    
    
#    w = 2*np.pi*f
#    Zm = R_m/(1+1.j*w*C_p*R_m)
#    #print(Vout,V_ac,Zm,w)
#    C_D0 = np.imag(Vout/V_ac/Zm/w)
#    
#    R_s0 = 1.
#    R_p0 = 1e9
#    
#    
#    pars = fit.Parameters()
#    
#    
#    if method == 'Rs':
#        func = lambda  x:  zero_func_Rs(x, f, V_ac, R_Dp, C_off, R_m, C_p, Vout)
#        X0 = np.array([C_D0,R_s0])
#    else:
#        func = lambda  x:  zero_func_Rp(x, f, V_ac, R_Ds, C_off, R_m, C_p, Vout)
#        X0 = np.array([C_D0,R_p0])
#        
#    res = fsolve(func, X0 , full_output = False)
#    pars.add('R', value = (np.abs(res[1])))
#    pars.add('f', value = f)
#    pars.add('C_D', value = res[0])
#    zero = func(res)
#    pars.add('Rezero', value = zero[0])
#    pars.add('Imzero', value = zero[1])
#    pars.add('Vout', value = Vout)
#    pars.add('Vout_found', value = zero[0]+1.j*zero[1]+Vout)
#    
#    
#    
#    fit.print_fitres(pars)
#    return pars


