import numpy as np
import inspect
from ._legacy import minimize, Parameters, Parameter, report_fit

def make_model(function, p0 = None):
    args = inspect.getargspec(function).args
    pars = Parameters()
    for arg in args:
        pars.add(arg)
    def model(pars):
        margs=()
        for arg in args:
            #print(arg)
            margs += (pars[arg].value,)
            #print(margs)
        return function(*margs)
    if p0:

        for kk,par in enumerate(pars):
            
            pars[par].value = p0[kk]
            try:
                len(p0[kk])
                pars[par].vary = False
            except:
                pass
    return model, pars

def residuals_lmfit(pars, fit_func, data):  
    res = fit_func(pars)-data
    #print type(data[0])
    #print type(1.j)
    if type(data[0]) == np.complex128:
        res = np.append(res.real,res.imag)
        #print res
        
    return res
    
def fit(pars,data, fit_func, **kw):
    '''
    Fits pars to data
    set independent pars.vary = False
    '''
    tol = kw.pop('xtol',1.e-10)
    #print 'tol: ',tol
    result = minimize(residuals_lmfit, pars, args=(fit_func,data),xtol=tol,ftol=tol)
    #print result
    #rep = report_fit(result)
    if kw.pop('ret_fit',True):
        return result, fit_func(pars)
    else:
        return result

def print_fitres(pars, **kw):
    p = pars.copy()
    keys = list(p.keys())
    for key in keys:
        if not p[key].vary:
            p.pop(key)
    return report_fit(p, **kw)




