from characterizations import CV_measurements as cvm
from characterizations.CV_measurements import dummy_lockin
import qcodes as qc
Vgs_sim = np.linspace(-5,5,41)  
s= cvm.dummy_lockin('dummy', 'addr')
s.frequency(1)
station = qc.Station(s)
cvm = cvm.CV_measurement('test1', Vgs_sim, s) 
cvm.calibrate()
cvm.measure_CVs()