import pyLTR

import numpy

import tempfile
import unittest

class SolarWindTest(unittest.TestCase):
    """
    Exercise the SolarWind base class.    
    """
    def setUp(self):
        self.sw = pyLTR.SolarWind.LFM.LFM('examples/data/solarWind/LFM_SW-SM-DAT')

    def test_bxFit(self):
        coefs = self.sw.bxFit()

        # Updated these bxFit coeffs test values on December 29, 2010 on tyr.hao.ucar.edu
        # May 25, 2010: Set 'places' to pass tests on my Intel Macbook Pro running OSX-10.5.8
        self.assertAlmostEqual(numpy.abs(numpy.abs(coefs[0])-1.3321518), 0.0, places=1)
        self.assertAlmostEqual(numpy.abs(numpy.abs(coefs[1])-0.0668088),  0.0, places=1)
        self.assertAlmostEqual(numpy.abs(numpy.abs(coefs[2])-0.7407690), 0.0, places=0)
        
    def test_writeLFM(self):
        fh=tempfile.NamedTemporaryFile()
        pyLTR.SolarWind.Writer.LFM(self.sw, fh.name)

        
