"""
Generate "polar" plots of from data stored in dictionaries. By "polar", we mean
plots of a 2D spherical shell centered on the north or south pole, rotated by
90 degrees so that 0 longitude points up in northern hemisphere, and properly
projected to a pole-centered orthographic POV where radius == sin(colatitude).
"""

import pylab as p
import numpy as n

def BasicPlotDict(longitude,colatitude,variable,plotOpts=None,
                  userAxes=None,useMesh=False,plotColorBar=True,colatLabel=True,axis_limits=None):
    """
    Wrapper to produce well-labeled polar plot of a color-coded scalar field.
 
    Inputs
      - first is a dictionary holding a 2D meshgrid of longitudes
      - second is a dictionary holding a 2D meshgrid of colatitudes
      - third is dictionary holding a 2D array of a scalar field whose elements 
        are located at coordinates specified in the first and second arguments;
        as per MIX convention: dim1 = longitudes, dim2=colatitudes
 
    Keywords
      - plotOpts1 - dictionary holding various optional parameters for tweaking
                    the scalar field appearance
                    'colormap': string specifying a colormap for surface or 
                                contourf plots
                                FIXME: this should be an actual colormap object
                    'min': minimum data value mapped to colormap
                    'max': maximum data value mapped to colormap
                    'format_str': format string for min/max labels
                    'numContours': number of contours between min and max
                    'numTicks': number of ticks in colorbar
                    
      - userAxes    - FIXME: not clear from context what this was intended to do,
                             but it currently does nothing;
      - useMesh     - if True, plot scalar field as a surface plot, not filled contours
      - plotColorBar- if True, plot a colorbar
      - colatLabel  - if True, label colatitudes, otherwise label latitudes

    Outputs
      Reference to a matplotlib.axes.PolarAxesSubplot object



    """
   
    # if user supplies axes use them otherwise create our own polar axes
    if userAxes == None:
       ax = p.axes(polar=True)
    else:
       ax = userAxes
    
     
    # convert longitude to polar coordinate theta, with 0 pointing north
    theta = longitude['data'] + p.pi/2.
  
    # convert colatitude to polar coordinate radius
    # FIXME: is this really what we want? ...perhaps it is more visually
    #        compelling, if not physically realistic, to plot colatitude 
    #        on a linear scale
    r = p.sin(colatitude['data'])
    
    
    # DEFINE GRID LINES AND LABELS
    # a list of circles in degrees co-latitude
    circle_list = [10,20,30,40]
    # circles in radii, NOT degrees or radians
    circles = [p.sin(elem*n.pi/180.) for elem in circle_list]
    # convert to string and add degree symbol
    if colatLabel:
       lbls=[str(elem)+'\xb0' for elem in circle_list]
    else:
       lbls=[str(elem)+'\xb0' for elem in (90-n.array(circle_list))]
    hour_labels = ['06','12','18','00']
    
    
    # DRAW POLAR GRID
    ax.set_rgrids(circles,lbls)
    ax.set_thetagrids((0.0,90.0,180.0,270.0),hour_labels)
    if not (axis_limits):
        ax.axis([0,2.0*n.pi,0,r.max()],'tight')
    else:
        ax.axis(axis_limits,'tight')
    
    # Process plotOpts
    if plotOpts == None:
       plotOpts = {}
    # if colormap is supplied use it
    if 'colormap' in plotOpts:
	#if string is passed assume it is a colormap name otherwise assume a
        #cmap object has been passed
        if isinstance(plotOpts['colormap'],str):
           cmap=eval('p.cm.'+plotOpts['colormap'])
        else:
           cmap = plotOpts['colormap']
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
    
    
    
    #Now onto the plotting
    contours = n.linspace(lower,upper,ncontours)
    ticks = n.linspace(lower,upper,nticks)
    var=variable['data']
    
    if (useMesh):
      p1 = ax.pcolor(theta,r,var,cmap=cmap,vmin=lower,vmax=upper,
                     rasterized=True)
    else:
      p1 = ax.contourf(theta,r,var,contours,extend='both',cmap=cmap)
    
    if (plotColorBar):
      cb=p.colorbar(p1,pad=0.075,ticks=ticks)
      cb.set_label(variable['name']+' ['+variable['units']+']')
      cb.solids.set_rasterized(True)

    ax.text(-75.*n.pi/180.,1.25*r.max(),('min: '+format_str+'\nmax: ' +format_str) % (var.min() ,var.max()))
    #ax.annotate(('min: '+format_str+'\nmax: ' + format_str) %
    #          (var.min() ,var.max()), (0.65, .0),
    #           textcoords='axes fraction',annotation_clip=False)
    
    return p1


def OverPlotDict(longitude,colatitude,variable1,variable2,plotOpts1=None,plotOpts2=None,
                 userAxes=None,useMesh=False,plotColorBar=True,colatLabel=True,axis_limits=None):
    """
    Wrapper to produce well-labeled polar plot of two co-located scalar fields.
 
    Inputs
      - first is a dictionary holding a 2D meshgrid of longitudes
      - second is a dictionary holding a 2D meshgrid of colatitudes
      - third is dictionary holding a 2D array of a scalar field whose elements 
        are located at coordinates specified in the first and second arguments;
        as per MIX convention: dim1 = longitudes, dim2=colatitudes
      - fourth is dictionary holding a 2D array of a scalar field whose elements 
        are located at coordinates specified in the first and second arguments;
        as per MIX convention: dim1 = longitudes, dim2=colatitudes
 
    Keywords
      - plotOpts1 - dictionary holding various optional parameters for tweaking
                    the scalar field appearance
                    'colormap': string specifying a colormap for surface or 
                                contourf plots
                                FIXME: this should be an actual colormap object
                    'min': minimum data value mapped to colormap
                    'max': maximum data value mapped to colormap
                    'format_str': format string for min/max labels
                    'numContours': number of contours between min and max
                    'numTicks': number of ticks in colorbar
                    
      - plotOpts2 - dictionary holding various optional parameters for tweaking
                    the scalar field appearance
                    'colormap': string specifying a colormap for contour plots
                                FIXME: this should be an actual colormap object
                    'colors': a recognized color, or list of colors
                    'min': minimum data value mapped to colormap
                    'max': maximum data value mapped to colormap
                    'format_str': format string for min/max labels
                    'numContours': number of contours between min and max
                    'numTicks': number of ticks in colorbar
      
      - userAxes    - FIXME: not clear from context what this was intended to do,
                            but it currently does nothing;
      - useMesh     - if True, plot scalar field as a surface plot, not filled contours
      - plotColorBar- if True, plot a colorbar
                      FIXME: does not currently plot a second colorbar
      - colatLabel  - if True, label colatitudes, otherwise label latitudes
 
    Outputs
      Reference to a matplotlib.axes.PolarAxesSubplot object
    """
   
    # if user supplies axes use them otherwise create our own polar axes
    if userAxes == None:
       ax = p.axes(polar=True)
    else:
       ax = userAxes
    
     
    # convert longitude to polar coordinate theta, with 0 pointing north
    theta = longitude['data'] + p.pi/2.
  
    # convert colatitude to polar coordinate radius
    # FIXME: is this really what we want? ...perhaps it is more visually
    #        compelling, if not physically realistic, to plot colatitude 
    #        on a linear scale
    r = p.sin(colatitude['data'])
    
    
    # DEFINE GRID LINES AND LABELS
    # a list of circles in degrees co-latitude
    circle_list = [10,20,30,40]
    # circles in radii, NOT degrees or radians
    circles = [p.sin(elem*n.pi/180.) for elem in circle_list]
    # convert to string and add degree symbol
    if colatLabel:
       lbls=[str(elem)+'\xb0' for elem in circle_list]
    else:
       lbls=[str(elem)+'\xb0' for elem in (90-n.array(circle_list))]
    hour_labels = ['06','12','18','00']
    
    # DRAW POLAR GRID
    ax.set_rgrids(circles,lbls)
    ax.set_thetagrids((0.0,90.0,180.0,270.0),hour_labels)
    if not (axis_limits):
        ax.axis([0,2.0*n.pi,0,r.max()],'tight')
    else:
        ax.axis(axis_limits,'tight')
    
    
    if plotOpts1 == None:
       plotOpts1 = {}
    # if colormap is supplied use it
    if 'colormap' in plotOpts1:
	#if string is passed assume it is a colormap name otherwise assume a
        #cmap object has been passed
        if isinstance(plotOpts1['colormap'],str):
           cmap1=eval('p.cm.'+plotOpts1['colormap'])
        else:
           cmap1 = plotOpts['colormap']
    else:
        cmap1=None # default is used

    # if limits are given use them, if not use the variables min/max values
    if 'min' in plotOpts1:
        lower1 = plotOpts1['min']
    else:
        lower1 = variable1['data'].min()

    if 'max' in plotOpts1:
        upper1 = plotOpts1['max']
    else:
        upper1 = variable1['data'].max()

    # if format string for max/min is given use otherwise do default
    if 'format_str' in plotOpts1:
       format_str1 = plotOpts1['format_str']
    else:
       format_str1='%.2f'

    # if number of contours is given use it otherwise do 51
    if 'numContours' in plotOpts1:
      ncontours1 = plotOpts1['numContours']
    else:
      ncontours1 = 51

    # if number of ticks is given use it otherwise do 51
    if 'numTicks' in plotOpts1:
      nticks1 = plotOpts1['numTicks']
    else:
      nticks1 = 11

    if plotOpts2 == None:
       plotOpts2 = {}

    # if colormap is supplied use it
    if 'colormap' in plotOpts2:
	#if string is passed assume it is a colormap name otherwise assume a
        #cmap object has been passed
        if isinstance(plotOpts2['colormap'],str):
           cmap2=eval('p.cm.'+plotOpts2['colormap'])
        else:
           cmap2 = plotOpts['colormap']
    else:
        cmap2=None # default is used

    # if colors is supplied use it
    if 'colors' in plotOpts2:
        colors2=plotOpts2['colors']
    else:
        colors2=None # default is used
  
    # can only have cmap or colors so if both use cmap
    if cmap2!=None and colors2!=None:
       colors2=None
   
    # if neither cmap or colors is supplied, default to black contours on overlay
    if cmap2==None and colors2==None:
       colors2='k'
   
    # if limits are given use them, if not use the variables min/max values
    if 'min' in plotOpts2:
        lower2 = plotOpts2['min']
    else:
        lower2 = variable2['data'].min()

    if 'max' in plotOpts2:
        upper2 = plotOpts2['max']
    else:
        upper2 = variable2['data'].max()

    # if format string for max/min is given use otherwise do default
    if 'format_str' in plotOpts2:
       format_str2 = plotOpts2['format_str']
    else:
       format_str2='%.2f'

    # if number of contours is given use it otherwise do 51
    if 'numContours' in plotOpts2:
      ncontours2 = plotOpts2['numContours']
    else:
      ncontours2 = 51

    # if number of ticks is given use it otherwise do 51
    if 'numTicks' in plotOpts2:
      nticks2 = plotOpts2['numTicks']
    else:
      nticks2 = 11
    
    
    #Now onto the plotting
    contours1 = n.linspace(lower1,upper1,ncontours1)
    ticks1 = n.linspace(lower1,upper1,nticks1)
    contours2 = n.linspace(lower2,upper2,ncontours2)
    ticks2 = n.linspace(lower2,upper2,nticks2)
    var1=variable1['data']
    var2=variable2['data']
    if (useMesh):
      if colors2:
         p1 = ax.contour(theta,r,var2,contours2,extend='both',colors=colors2)
      else:
         p1 = ax.contour(theta,r,var2,contours2,extend='both',cmap=cmap2)
      
      # call pcolor last, or color bar won't work correctly
      ax.pcolor(theta,r,var1,cmap=cmap1,vmin=lower1,vmax=upper1)
    
    else:
      if colors2:
         p1 = ax.contour(theta,r,var2,contours2,extend='both',colors=colors2)
      else:   
         p1 = ax.contour(theta,r,var2,contours2,extend='both',cmap=cmap2)
      
    p2 = ax.contourf(theta,r,var1,contours1,extend='both',cmap=cmap1)
    
    if (plotColorBar):
      cb1=p.colorbar(p2,pad=0.075,ticks=ticks1)
      cb1.set_label(variable1['name']+' ['+variable1['units']+']')
      cb1.solids.set_rasterized(True)
      

    ax.text(-75.*n.pi/180.,1.25*ax.get_ylim()[1],('min: '+format_str1+'\nmax: ' +
           format_str1) % (var1.min() ,var1.max()))
    ax.text(n.deg2rad(255),1.25*ax.get_ylim()[1],('min: '+format_str2+'\nmax: ' +
           format_str2) % (var2.min() ,var2.max()),ha='right')
    #ax.annotate(('min: '+format_str1+'\nmax: ' + format_str1) %
    #           (var1.min() ,var1.max()), (0.65, .0),
    #           textcoords='axes fraction',annotation_clip=False)
    #ax.annotate(('min: '+format_str2+'\nmax: ' + format_str2) %
    #           (var2.min() ,var2.max()), (0.35, .0),
    #           textcoords='axes fraction',annotation_clip=False,ha='right')
    
    return p1


def QuiverPlotDict(longitude,colatitude,scalars,vectors,plotOpts1=None,plotOpts2=None, 
                   longTicks=None, longLabels=None, colatTicks=None, colatLabels=None,
                   northPOV=True,useMesh=False,plotColorBar=True,plotQuiverKey=True,
                   userAxes=None):
   """
   Wrapper to produce well-labeled polar plot of a vector field overlaid on a
   color-coded scalar field.
   
   Inputs
     - first is a dictionary holding a 2D meshgrid of longitudes (azimuth)
     - second is a dictionary holding a 2D meshgrid of colatitudes (radius)
     - third is dictionary holding a 2D array of a scalar field whose elements 
       are located at coordinates specified in the first and second arguments;
       as per MIX convention: dim1 = longitudes, dim2=colatitudes
     - fourth is a 2-tuple of 2D arrays of local {long,colat} direction vector
       components whose elements are located at coordinates specified in the
       first and second arguments; dim1 = longitudes, dim2=colatitudes
   
   Keywords
     - plotOpts1  - dictionary holding various optional parameters for tweaking
                    the scalar field appearance
                    'colormap': string specifying a colormap for surface or 
                                contourf plots
                                FIXME: this should be an actual colormap object,
                                       NOT just the name of a standard colormap
                    'min': minimum data value mapped to colormap
                    'max': maximum data value mapped to colormap
                    'format_str': format string for min/max labels
                    'numContours': number of contours between min and max
                    'numTicks': number of ticks in colorbar
                    
     - plotOpts2  - dictionary holding various optional parameters for tweaking
                    the vector field appearance
                    'scale': floating point number specifying how many data unit
                             vector arrows will fit across the width of the plot;
                             default is for max. vector amplitude to be 1/10th
                             plot width
                    'width': floating point number specifying the width of an
                             arrow shaft as a fraction of plot width
                    'pivot': should be one of:
                             'tail' - pivot about tail (default)
                             'middle' - pivot about middle
                             'tip' - pivot about tip
                    'color': color of arrow
     
     - longTicks  - where to place 'longitude' ticks in radians
     - longLabels - labels to place at longTicks
     - colatTicks - where to place 'colatiude' ticks in 'colatitude' units
     - colatLabels- labels to place at colatTicks
     - northPOV   - if True, assume a POV above north pole, otherwise assume
                    POV below south pole (this does NOT transform data, but 
                    only corrects the plot axes to match the POV; it is up to
                    the user to ensure proper inputs)
     - useMesh    - if True, plot scalar field as surface plot, not filled contours
                    (should be quicker in theory, but seems to be much slower)
     - plotColorBar- if True, plot a colorbar
     - plotQuiverKey- if True, plot and label a scaled arrow outside the plot
     - userAxes   - FIXME: it currently does nothing, although if None (default),
                    this function creates a new figure...this is almost contrary
                    to normal Matplotlib behavior.
     
   Outputs
     Reference to a matplotlib.axes.PolarAxesSubplot object
   
   """
   
   # if user supplies axes use them other wise create our own polar axes
   if userAxes == None:
      ax = p.axes(polar=True)
   else:
      ax = userAxes
      
    
   # use longitude for azimuthal dimension
   # NOTE: polar plots have methods to set the 0 direction, BUT they do not
   #       handle vector fields properly (or rather, the vector field plot
   #       routines are not written correctly for polar projections)...we
   #       simply add pi/2, and make all the proper transformations below
   theta = longitude['data'].copy() + p.pi/2
      
   # use colatitude for radial dimension
   r = colatitude['data'].copy()
   
   # adjust r so colatitudes increase from 0->pi/2, then decrease from pi/2->pi;
   # a corresponding adjustment is made to the vrs vector field below
   gt90 = r > p.pi/2
   r[gt90] = p.pi - r[gt90]
   
   
   
   # Draw Polar grids with 0 longitude pointing up, correcting for POV
   if longTicks == None:
      longTicks = [elem*p.pi/180 for elem in [0, 90, 180, 270]]
      # if longTicks was not passed, quietly ignore any longLabels passed
      longLabels = [r'0'+'\xb0',r'90'+'\xb0',r'180'+'\xb0',r'270'+'\xb0'] 
   else:
      longTicks = [elem for elem in longTicks]
   if northPOV:
      longTicks = [elem + p.pi/2 for elem in longTicks]
   else:
      longTicks = [p.mod(p.arctan2(p.sin(elem), -p.cos(elem)) - p.pi/2, 2*p.pi) for elem in longTicks]
   if longLabels == None:
      thetaLines,thetaLabels = p.thetagrids([elem*180/p.pi for elem in longTicks])
   else:
      thetaLines,thetaLabels = p.thetagrids([elem*180/p.pi for elem in longTicks], longLabels)
   
   p.setp(thetaLabels, fontsize=10, color='0.4')
   
   if colatTicks == None:
      # let Matplotlib determine colatTicks
      # if longTicks was not passed, quietly ignore any longLabels passed
      colatLabels = None
   if colatLabels == None:
      rhoLines,rhoLabels = p.rgrids()
   else:
      rhoLines,rhoLabels = p.rgrids(colatTicks, colatLabels)
   
   p.setp(rhoLabels, fontsize=10, color='gray')
   
   p.axis([0,2.0*n.pi,0,r.max()],'tight')
   
     
   
   # if scalars is False (or None, or empty, etc.), skip and proceed to quiver plot
   if scalars:
      
      # PROCESS PLOT OPTIONS FOR FIRST (SCALAR) INPUT
      
      if plotOpts1 == None:
         plotOpts1 = {}
      
      # if colormap is supplied use it
      if 'colormap' in plotOpts1:
         cmap1=eval('p.cm.'+plotOpts1['colormap'])
      else:
         cmap1=None # default is used

      # if limits are given use them, if not use the variables min/max values
      if 'min' in plotOpts1:
         lower1 = plotOpts1['min']
      else:
         lower1 = scalars['data'].min()
      if 'max' in plotOpts1:
         upper1 = plotOpts1['max']
      else:
         upper1 = scalars['data'].max()
      # if format string for max/min is given use otherwise do default
      if 'format_str' in plotOpts1:
         format_str1 = plotOpts1['format_str']
      else:
         format_str1='%.2f'

      # if number of contours is given use it otherwise do 51
      if 'numContours' in plotOpts1:
         ncontours1 = plotOpts1['numContours']
      else:
         ncontours1 = 51

      # if number of ticks is given use it otherwise do 51
      if 'numTicks' in plotOpts1:
         nticks1 = plotOpts1['numTicks']
      else:
         nticks1 = 11
      
      
      contours1 = n.linspace(lower1,upper1,ncontours1)
      ticks1 = n.linspace(lower1,upper1,nticks1)
      var1=scalars['data']
      
      if (useMesh):
         p.pcolor(theta,r,var1,cmap=cmap1,vmin=lower1,vmax=upper1)
      else:
         p.contourf(theta,r,var1,contours1,extend='both',cmap=cmap1)
      
      if (plotColorBar):
         cb1=p.colorbar(pad=0.075,ticks=ticks1)
         cb1.set_label(scalars['name']+' ['+scalars['units']+']')
         cb1.solids.set_rasterized(True)

      p.annotate(('min: '+format_str1+'\nmax: ' + format_str1) % 
                 (var1.min() ,var1.max()), (0.65, -.05), 
                 textcoords='axes fraction',annotation_clip=False)
      
   
   # if vectors is False (or None, or empty, etc.), skip quiver plot
   if vectors:
      
      # copy 'longitudinal' component of directional vector to polar theta component
      vts = vectors[0]['data'].copy()
      
      # copy 'colatitudinal' component of directional vector to polar r component
      vrs = vectors[1]['data'].copy()
      
      # adjust vrs to accomodate vectors positioned at r > pi/2
      vrs[gt90] = -vrs[gt90]
      
      
      # PROCESS PLOT OPTIONS FOR SECOND (VECTOR) INPUT
      if plotOpts2 == None:
         plotOpts2 = {}
      
      # use scale if given, otherwise max. magnitude generates arrow 1/10 plot-width
      if 'scale' in plotOpts2:
         scale = plotOpts2['scale']
      else:
         scale = p.sqrt(vrs**2 + vts**2).max() / 0.1
      
      # use width if given, otherwise default is .0025 plot-width
      if 'width' in plotOpts2:
         width = plotOpts2['width']
      else:
         width = .0025
      
      # use pivot if given, otherwise default is 'tail'
      if 'pivot' in plotOpts2:
         pivot = plotOpts2['pivot']
      else:
         pivot = 'tail'
      
      # use color if given, otherwise default is black
      if 'color' in plotOpts2:
         color2 = plotOpts2['color']
      else:
         color2 = 'k'
      
      # consider adding options to control spacing of vector field arrows
      
      
      # interpolate to a polar grid that alleviates some of the so-called
      # "pole problem" described by Randall (2011 Online notes) -EJR 12/2013
      for j in range(r[0,:].size):
         
         if r[0,j]==0:
            
            # if radius==0, there should be only one unique vector, so just
            # use the first element 
            r_tmp = 0
            theta_tmp = theta[0,j]
            vr_tmp = vrs[0,j]
            vt_tmp = vts[0,j]
             
         else:
            
            # OK, it took the better part of a day to determine that a "bug" 
            # was nothing more than me neglecting to ensure that the known x's
            # were monotonically increasing in calls to p.interp()...ARGH!!!
            #
            # Now, this whole business of allowing colatitudes that are greater
            # than pi/2 needs to be re-visited, since it is likely to lead to
            # even bigger problems if/when someone attempts to pass colatitudes
            # from both hemispheres simultaneously..then again, I would like to
            # eventually supercede this function with one based on the BASEMAP
            # extension to Matplotlib, so maybe this is a moot point. -EJR 2/2014
            theta_tmp = p.linspace(theta[:,0].min(), theta[:,0].max(), 3*j+1)
            r_tmp = theta_tmp * 0 + r[0,j] # all r's are the same for each j
                     
            # currently only 1D interpolation along columns of constant theta;
            # 2D interpolation could be use to achieve more uniform distribution
            # of quiver positions, however this is already easily obtained if
            # the BASEMAP extenstion to Matplotlib is used, so leave as-is. -EJR 2/2014
            minc = theta[:,j].argsort()
            vr_tmp = p.interp(theta_tmp, theta[minc,j], vrs[minc,j])
            vt_tmp = p.interp(theta_tmp, theta[minc,j], vts[minc,j])
            
         
         # call pylab's quiver, correcting for local polar coordinates
         # NOTE: if pylab/matplotlib ever fixes quiver to properly handle polar
         #       plots (or projections in general), the rotation below will need 
         #       to be removed -EJR 11/2013  ...once again, this is probably a
         #       moot point if/when BASEMAP is incoporated into our plot functions.
         units='width' # use plot width as basic unit for all quiver attributes
         qh = p.quiver(theta_tmp, r_tmp, 
                       vr_tmp * p.cos(theta_tmp) - vt_tmp * p.sin(theta_tmp), 
                       vr_tmp * p.sin(theta_tmp) + vt_tmp * p.cos(theta_tmp), 
                       units=units, scale=scale, width=width, pivot=pivot, color=color2)         
         
         ## uncomment for debugging
         #print 'Number of longitude gridpoints: ',vr_tmp.size - 1
         #blah = raw_input('Hit key for next plot:')
         
      if plotQuiverKey:
         # Since we forced units to be 'width', a scale keyword equal to 1 implies 
         # that an arrow whose length is the full width of the plot will be equal 
         # to 1 data unit, a scale keyword of 2 implies that an arrow whose length 
         # is the full width of the plot will be equal to two data units, etc.
         # Here we draw a "Quiver Key" that is 1/10th the width of the plot, and 
         # properly scale its value...much pained thought went into convincing my-
         # self that this is correct, but feel free to verify -EJR 12/2013
         p.quiverkey(qh, .3, 0, .1*scale, 
                     ('%3.1e '+'%s') % (.1*scale,vectors[0]['units']), 
                     coordinates='axes',labelpos='S')
   
   return p.gca()
