from unittest import TestCase
import numpy as np

from characterizations.mobility import find_V_th_and_R_series


class TestFind_V_th_and_R_series(TestCase):
    def setUp(self):
        # Example data was generated as following:
        # std = 1e-6
        # mean = 1e-3
        # dydx = 1e-3
        # x = np.linspace(0,1,100)
        # threshold = 0.5
        # example_Gs = np.piecewise(x,
        #                           [x < threshold, x >= threshold],
        #                           [lambda x: np.random.randn(len(x))*std+mean,
        #                            lambda x: dydx*(x-threshold)+np.random.randn(len(x))*std+mean])

        self.example_Gs = np.array([0.00100118, 0.00099965, 0.00099964, 0.00100076, 0.00099916,
                                    0.00100126, 0.00099908, 0.00100143, 0.00100035, 0.00099924,
                                    0.00100178, 0.00100193, 0.00100122, 0.00100093, 0.00099785,
                                    0.00099842, 0.00099945, 0.00099968, 0.00100109, 0.000998,
                                    0.0010009, 0.00100015, 0.00100008, 0.00100131, 0.00099897,
                                    0.00099917, 0.00099967, 0.00100005, 0.00100058, 0.00099802,
                                    0.00099988, 0.00099949, 0.00100043, 0.00100131, 0.00100034,
                                    0.00099914, 0.00100063, 0.00100069, 0.0009996, 0.00100091,
                                    0.00100013, 0.00099913, 0.00099976, 0.00100068, 0.00099981,
                                    0.00100122, 0.00100138, 0.00099946, 0.00100054, 0.00099923,
                                    0.00100413, 0.00101588, 0.00102476, 0.00103719, 0.0010453,
                                    0.00105612, 0.00106387, 0.00107549, 0.00108631, 0.00109535,
                                    0.00110636, 0.00111496, 0.00112648, 0.00113619, 0.00114693,
                                    0.00115796, 0.00116721, 0.00117684, 0.00118746, 0.00119733,
                                    0.00120828, 0.00121762, 0.00122692, 0.00123777, 0.00124658,
                                    0.00125575, 0.00126908, 0.00127574, 0.00128914, 0.00129673,
                                    0.00130862, 0.00131943, 0.00132787, 0.00133613, 0.00134895,
                                    0.0013592, 0.00136859, 0.00137749, 0.00138957, 0.00139845,
                                    0.0014085, 0.00141967, 0.00142929, 0.00144029, 0.00144912,
                                    0.00146003, 0.00146897, 0.00147999, 0.00149143, 0.00150205])

        self.example_Vgs = np.linspace(0, 1, 100)

    def test_find_V_th_and_R_series(self):
        Vth, R_series = find_V_th_and_R_series(self.example_Vgs, self.example_Gs)

        self.assertAlmostEqual(Vth, 0.5, 1)
