"""
Generate Cut planes of data stored in a dictionary
"""
import pylab as p
import numpy as n

def CutPlaneDict(x,y,variable,plotOpts=None,userAxes=None):
    """
    Create a well labeled Cut Plane
    """

    if plotOpts == None:
       plotOpts = {}
    # if colormap is supplied use it
    if 'colormap' in plotOpts:
        cmap=eval('p.cm.'+plotOpts['colormap'])
    else:
        cmap=None # default is used

    # if limits are given use them, if not use the variables min/max values
    if 'min' in plotOpts:
        lower = plotOpts['min']
    else:
        lower = variable['data'].min()

    if 'max' in plotOpts:
        upper = plotOpts['max']
    else:
        upper = variable['data'].max()

    # if format string for max/min is given use otherwise do default
    if 'format_str' in plotOpts:
       format_str = plotOpts['format_str']
    else:
       format_str='%.2f'

    # if number of contours is given use it otherwise do 51
    if 'numContours' in plotOpts:
      ncontours = plotOpts['numContours']
    else:
      ncontours = 51

    # if number of ticks is given use it otherwise do 51
    if 'numTicks' in plotOpts:
      nticks = plotOpts['numTicks']
    else:
      nticks = 11

    # if user supplies axes use them other wise create our own polar axes
    if userAxes == None:
      ax = p.axes()
    else:
      ax = userAxes
    #Now onto the plotting
    contours = n.linspace(lower,upper,ncontours)
    ticks = n.linspace(lower,upper,nticks)
    var=variable['data']
    p.contourf(x,y,var,contours,extend='both',cmap=cmap)
    p.axis('equal')
    cb=p.colorbar(pad=0.075,ticks=ticks,format='%.0f')
    cb.set_label(variable['name']+' ['+variable['units']+']')
   
