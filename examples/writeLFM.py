import pyLTR

import pylab

#
# Read an OMNI solar wind file from CDAWeb & write an LFM solar wind file
#
OMNI  = pyLTR.SolarWind.OMNI('data/solarWind/OMNI_HRO_1MIN_14180.txt1')

OMNI.writeLFM('LFM_SW_DATFILE')
print('Created LFM Solar Wind File at ./LFM_SW_DATFILE.')
