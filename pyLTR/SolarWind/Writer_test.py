import pyLTR

import datetime
import tempfile
import unittest

class LFMWriterTest(unittest.TestCase):
    """ Exercise the LFM writer class"""
    def setUp(self):
        sw=pyLTR.SolarWind.LFM.LFM('examples/data/solarWind/LFM_SW-SM-DAT')
        self.f = pyLTR.SolarWind.Writer.LFM(sw, tempfile.NamedTemporaryFile().name) #doctest: +NORMALIZE_WHITESPACE

    def test_getWriteLFMFormatString(self):        
        formatStr = self.f._LFM__getWriteLFMFormatString([1],[1], [1],[1],[1], [1], [1],[1],[1],[1], [1])

        self.assertEqual(formatStr, '%4.2f %5.1f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %6.3f\n')


class TIEGCMWriterTest(unittest.TestCase):
    """ Exercise the TIEGCM writer class"""
    def setUp(self):
        sw = pyLTR.SolarWind.LFM.LFM('examples/data/solarWind/LFM_SW-SM-DAT')
        self.f  = pyLTR.SolarWind.Writer.TIEGCM(sw, tempfile.NamedTemporaryFile().name)

    def test_getTime(self):
        list = self.f._TIEGCM__getTime(datetime.datetime(2010,12,29,0,0,0), [0,15,30,60])

        self.assertAlmostEqual(2010363.0,          list[0][0], places=3)
        self.assertAlmostEqual(2010363.0104166667, list[0][1], places=3)
        self.assertAlmostEqual(2010363.0208333333, list[0][2], places=3)
        self.assertAlmostEqual(2010363.0416666667, list[0][3], places=3)
        self.assertEqual(['2010363'], list[1])
        
    def test_getFirstTimeIndex(self):
        firstIndex = self.f._TIEGCM__getFirstTimeIndex([0, 15, 30, 60])
        self.assertEqual(2, firstIndex)

    def test_getRealDataMask(self):
        """ 
        This solar wind file was hand-edited to contain invalid data
        at a few indices.  This exercises the code in
        TIEGCM::__getRealDataMask(...)
        """

        sw = pyLTR.SolarWind.OMNI.OMNI('examples/data/solarWind/OMNI_testInvalidIndex.txt1')
        f  = pyLTR.SolarWind.Writer.TIEGCM(sw, tempfile.NamedTemporaryFile().name)
        mask = f._TIEGCM__getRealDataMask(sw)

        # Should be 9 time elements in the file: File has 60 minutes
        # of data, but the 15-minute average means the output will be
        # for 45 minutes of data (from time t=20 to t=65).  We sample
        # the output at 5 minute intervals, so there should be 45/5=9
        # intervals of valid data.
        self.assertEqual( len(mask[0]), 9) 

        # Make sure the mask exists for 5 varialbes: bx, by, bz, V, Density
        self.assertEqual( len(mask), 5 )
        
        # Magnetic field has valid data everywhere:
        self.assertEqual( mask[0].sum(), 8) # bx has 8 valid data points after coarse filtering.
        self.assertEqual( mask[1].sum(), 9) # by has 9 valid data points.
        self.assertEqual( mask[2].sum(), 9) # bz has 9 valid data points.

        print((sw.data.getData('time_min')))
        print((f.time_min))

        print((mask[3]))
        print((mask[4]))

        # Clock starts at 1:32.  Vx, Vy, or Vz have invalid data at
        # 1:35, 1:36 and 1:45.  Therefore, the 45 minutes of
        # "averaged" data sampled at 5 minutes will be derived from
        # invalid/interpolated values at 3 indices
        #
        # Put another way: Velocity has 9 valid data points, 6 of
        # which are derived from original/unmodified data.
        self.assertEqual( mask[3].sum(), 6) 
        
        # Density has invalid data at 2:00.  Therefore, the 5-minute
        # averaged data will be invalid at three time indices.
        #
        # Put another way:
        # Density has 9 valid data points, 6 of which are derived from
        # original/unmodified data.
        self.assertEqual( mask[4].sum(), 6) 
        
    
        
