#'GdL 20180119'
'''
usage:
    p0 = (x1,x2, p0, p1,p2,....)
    model_function, pars = fit.make_model(function,  p0 = p0)
    result = fit.fit(pars, ydata, model_function)
    
if pars['xn'].value is an array it is assumed to be independent values so make_model sets p['xn'].vary = False
'''