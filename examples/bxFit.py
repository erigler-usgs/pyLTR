import pyLTR

import pylab

#
# Read a solar wind file & compute the Bx fit
#
OMNI  = pyLTR.SolarWind.OMNI('data/solarWind/OMNI_HRO_1MIN_14180.txt1')

coef = OMNI.bxFit()

# Let's analyze the results:
print('The coefficients are ', coef)
by = OMNI.data.getData('by')
bz = OMNI.data.getData('bz')
bx_fit = coef[0] + coef[1] * by + coef[2] * bz                        

pyLTR.Graphics.TimeSeries.BasicPlot(OMNI.data, 'time_doy', 'bx', color='k')
pylab.plot(OMNI.data.getData('time_doy'), bx_fit, 'g')
pylab.title('Bx Fit Coefficients:\n$Bx_{fit}(0)$=%f      $By_{coef}$=%f      $Bz_{coef}$=%f' % (coef[0], coef[1], coef[2]) )
pylab.legend(('$Bx$','$Bx_{fit}$'))
pylab.show()
