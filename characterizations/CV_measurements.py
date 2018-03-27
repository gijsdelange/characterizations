import numpy  as np

import sys
import os
import matplotlib.pyplot as plt



def V_out(V_ac, Z_D, R_m, Z_p):
    
    Z_r = R_m*Z_p/(R_m + Z_p)
    V_out = Z_r/(Z_D + Z_r)*V_ac
    return V_out

def V_out_cap_model(f, V_ac, C_D, R_Dp, R_Ds, C_off, R_m, C_p):
    Z_p = 1/(2.j*np.pi*f*C_p)
    Z_C_D = 1./(2.j*np.pi*f*(C_D+C_off))
    Z_D = R_Dp*Z_C_D/(R_Dp + Z_C_D)+R_Ds
    return V_out(V_ac, Z_D, R_m, Z_p)


def I_detector(V_ac, Z_D, Z_i, Z_s):
    Z_2 = Z_i*Z_s/(Z_i + Z_s)
    Im = Z_2/(Z_D+Z_2)/Z_i*V_ac
    return Im


def I_d_cap_model(f, V_ac, R_D, C_D, R_i, L_i, C_s):
    
    return I_detector(V_ac, (1./R_D+(2.j*np.pi*f*C_D))**-1, R_i+2.j*np.pi*f*L_i, 1./(2.j*np.pi*f*C_s))


 
