import pyLTR

import pylab

#
# Read & Plot solar wind data in a variety of formats
#
LFM  = pyLTR.SolarWind.LFM('data/solarWind/LFM_SW-SM-DAT')
CCMC = pyLTR.SolarWind.CCMC('data/solarWind/CCMC_wsa-enlil.dat')
OMNI  = pyLTR.SolarWind.OMNI('data/solarWind/OMNI_HRO_1MIN_14180.txt1')
ENLIL  = pyLTR.SolarWind.ENLIL('data/solarWind/ENLIL-cr2068-a3b2.Earth.dat')
pyLTR.Graphics.TimeSeries.MultiPlotN([LFM.data, CCMC.data, OMNI.data, ENLIL.data], 'time_doy', ['b','vx','n'],
                                      ['b','r','g', 'm'], ['LFM','CCMC', 'OMNI', 'ENLIL'])
pylab.show()

