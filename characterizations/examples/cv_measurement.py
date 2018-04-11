import numpy as np
from characterizations import CV_measurements as cvm
from characterizations.CV_measurements import dummy_lockin
import qcodes as qc


from qcodes.instrument_drivers.stanford_research.SR865 import SR865


try:
   lockin = SR865("lockin", "USB0::0xB506::0x2000::003225::INSTR")
   #cvm.DATAPATH(r"C:\Users\jowat\data")
   station = qc.Station(lockin)
   
    
except:
    pass # instrument already created
#s.frequency(1)
 

print('default fmin, fmax = %s'%cvm.FREQUENCIES)
print('default npoints = %s'%cvm.NPOINTS)
print('default C_p  = %s pF'%(cvm.C_P*1e12))


Vgs = np.linspace(-5,5,11)  
cv_dev = cvm.CV_measurement('MOSCAP003', Vgs, station, 'lockin', datapath = r"C:\Users\jowat\data")

#****USER INPUTS**********
cv_dev.data['diameter'] = 50e-6
cv_dev.data['column'] = 3
cv_dev.data['thickness'] = 41e-9
#*****END USER INPUTS*******

cv_dev.calibrate() # do this only when you change the setup
cvm.C_P = cv_dev.C_p # this sets the dfault value of the module to the measured C_p for measurement of subsequent devices, this will be lost when the module is reloaded and set back to 500 pF


cv_dev.measure_CVs() # sweeps gate and frequency and plots and saves in D:\data\'date'\file.hdf5