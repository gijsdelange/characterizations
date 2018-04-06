from characterizations import CV_measurements as cvm
from characterizations.CV_measurements import dummy_lockin
import qcodes as qc
try:
    s= cvm.dummy_lockin('dummy_lockin', 'addr')
    station = qc.Station(s)
except:
    pass # instrument already created
#s.frequency(1)


print('default fmin, fmax = %s'%cvm.FREQUENCIES)
print('default npoints = %s'%cvm.NPOINTS)
print('default C_p  = %s pF'%(cvm.C_P*1e12))


Vgs_sim = np.linspace(-5,5,11)  
cv_dev = cvm.CV_measurement('dummy_CV_mmt', Vgs_sim, station, 'dummy_lockin', datapath = cvm.DATAPATH) 

cv_dev.calibrate() # do this only when you change the setup
cvm.C_P = cv_dev.C_p # this sets the dfault value of the module to the measured C_p for measurement of subsequent devices, this will be lost when the module is reloaded and set back to 500 pF


cv_dev.measure_CVs() # sweeps gate and frequency and plots and saves in D:\data\'date'\file.hdf5