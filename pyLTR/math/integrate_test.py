from . import integrate
import pyLTR.Models.MIX

import datetime
import numpy
import unittest

class TestIntegrate(unittest.TestCase):
    def setUp(self):
        # Test runner (nosetests) may fail without this silly try/except block:
        try:
            data = pyLTR.Models.MIX('examples/data/models', 'LMs_UTIO')
        except TypeError:
            data = pyLTR.Models.MIX.MIX('examples/data/models', 'LMs_UTIO')

        self.x=data.read('Grid X', datetime.datetime(year=2008, month=1, day=1, hour=4, minute=0, second=0))        
        self.y=data.read('Grid Y', datetime.datetime(year=2008, month=1, day=1, hour=4, minute=0, second=0))
        
        # Remove singularity
        self.x[:,0] = 0.0
        self.y[:,0] = 0.0

    def testAreaCircle(self):
        """ Compute area of a circle in a 2d plane. """
        analyticArea = numpy.pi * (self.y.max()**2)
        
        areaCircle = integrate.calcFaceAreas(self.x, self.y, numpy.zeros(self.x.shape))        
        self.assertAlmostEqual(analyticArea, areaCircle.sum(), delta=1e-4)

    def testAreaDisk(self):
        """ Compute area of a disk on the surface of a sphere. """
        analyticArea=2.0*numpy.pi*(1.0-numpy.cos(44.0*numpy.pi/180.0))

        # Assume we're working with a unit sphere
        z = numpy.sqrt(1.0-(self.x**2)-(self.y**2))  
        areaSphere = integrate.calcFaceAreas(self.x,self.y,z)

        self.assertAlmostEqual(analyticArea, areaSphere.sum(), delta=1e-3)

if __name__ == "__main__":
    unittest.main()
