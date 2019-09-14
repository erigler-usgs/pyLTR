"""
This example shows how to generate simple line plots using the pyLTR
'TimeSeries' class to store data.
"""

# Standard Python imports
import numpy
import pylab

# Custom imports
import pyLTR

if __name__ == '__main__':
    # Generate some arrays
    x = numpy.deg2rad(numpy.arange(0,360,1,float))
    y = numpy.sin(x)
    y1 = numpy.cos(x)
    y2 = y+y1
    z = 2.0*y
    z1 = 2.0*y1
    z2 = 2.0*y2

    # Store arrays as pyLTR.TimeSeries objects
    Dict = pyLTR.TimeSeries()
    Dict.append('Xval','Time','s',x)
    Dict.append('Yval','Amp','m',y)
    Dict.append('Y1val','Amp 1','ft',y1)
    Dict.append('Y2val','Amp 2','cm',y2)
    Dict['MetaData']="This is the metadata"

    Dict2 = pyLTR.TimeSeries()
    Dict2.append('Xval','Time','s',x)
    Dict2.append('Yval','Amp','m',z)
    Dict2.append('Y1val','Amp 1','ft',z1)
    Dict2.append('Y2val','Amp 2','cm',z2)
    Dict2['MetaData']="This is the metadata"

    # Make simple Time Series plots via pyLTR.Graphics.TimeSeries module:
    pylab.figure(1)
    pyLTR.Graphics.TimeSeries.BasicPlot(Dict,"Xval","Yval")
    pylab.title('BasicPlot example')

    pylab.figure(2)
    # Summary plot is designed to plot EVERY variable in a dictionary.
    # Note: SummaryPlot has not been extensively tested.
    pyLTR.Graphics.TimeSeries.SummaryPlot(Dict,"Xval") 
    pylab.title('SummaryPlot example')

    pylab.figure(3)
    items=["Y2val","Yval"]
    pyLTR.Graphics.TimeSeries.MultiPlot(Dict,"Xval",items) 
    pylab.title('MultiPlot example')

    pylab.figure(4)
    pyLTR.Graphics.TimeSeries.MultiPlot2(Dict,Dict2,'Xval',items,color1='b',color2='r')
    pylab.title('MultiPlot2 example')

    pylab.figure(5)
    # MultiPlotN is the most generic way to plot lots of TimeSeries objects against one another.
    pyLTR.Graphics.TimeSeries.MultiPlotN([Dict, Dict2], 'Xval', items, ['b', 'r'], ['Dict 1 data','Dict 2 data'])
    pylab.title('MultiPlotN example')

    pylab.show()
