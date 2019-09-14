#!/usr/bin/env python
"""
Script to create 6-panel summary plot of MIX run.

Execute mixIonSummary.py with the '--about' or '--help'  flags for more information.
"""

# Custom
import pyLTR

# 3rd-party
import pylab as p
import numpy as n
from multiprocessing import Pool
from psutil import cpu_count

# Standard
import datetime
import math
import optparse
import os
import re
import sys
import subprocess

def parseArgs():
    """
    Creates a 6 panel summary plot of MIX run
      - path: Path to a directory containing a MIX run.
      - runName: Name of run (ie. 'RUN-IDENTIFIER')
      - t0: datetime object of first step to be used in time series
      - t1: datetime object of last step to be used in time series
      - renderNorth: Render only northern hemisphere
      - renderSouth: Render only southern hemisphere
    Execute `mixIonSummary.py --help` for more information.
    """
    # additional optparse help available at:
    # http://docs.python.org/library/optparse.html
    # http://docs.python.org/lib/optparse-generating-help.html
    parser = optparse.OptionParser(usage='usage: %prog -p [PATH] [options]',
                                   version=pyLTR.Release.version)

    parser.add_option('-p', '--path', dest='path',
                      default='/Users/schmitt/paraview/testData/March1995_LM_r1432_single',
                      metavar='PATH', help='Path to base run directory containig MIX HDF files.')

    parser.add_option('-r', '--runName', dest='runName',
                      default='', metavar='RUN_IDENTIFIER', help='Optional name of run. Leave empty unless there is more than one run in a directory.')

    parser.add_option('-f', '--first', dest='t0',
                      default='', metavar='YYYY-MM-DD-HH-MM-SS', help='Date & Time that should be the first element of the time series')

    parser.add_option('-l', '--last', dest='t1',
                      default='', metavar='YYYY-MM-DD-HH-MM-SS', help='Date & Time of last element for the time series')
    parser.add_option('-n', '--north', dest='north', 
                      default=False, action='store_true',
                      help='Only render Northen hemisphere.')

    parser.add_option('-s', '--south', dest='south',
                       default=False, action='store_true',
                      help='Only render Southern hemisphere.')
    parser.add_option("-c", "--configFile", dest="configFile",
                      default=None, action="store", metavar="FILE",
                      help="Path to a configuration file.  Note: "
                      "these config files are automatically written to "
                      "the fig directory whenever you run mixIonSummary "
                      "(look for a file ending with "
                      "\".config\").")
    parser.add_option('-k', '--cpu', dest='ncpus',
                        default=1, action='store',
                        help='Number of cpus for the parallelism. Note default is 8')
    parser.add_option("-m", "--movieEncoder", dest="movieEncoder", default="ffmpeg",
                      metavar="[ffmpeg|mencoder|none]",
                      help="Movie encoder. Currently supported formats are ffmpeg (recommended), mencoder or none.")

    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    if options.about:
        print((sys.argv[0] + ' version ' + pyLTR.Release.version))
        print('')
        print('This script searches a path for any MIX ionosphere output files and ')
        print('makes a six panel summary plot of ionospheric paramters')
        print('for both the Northern and Southern hemispheres.')
        print('')
        print('To limit the time range, use the "--first" and/or "--last" flags.')
        print('To render only north or south use the "--north" or "--south" flags.')
        print(('Execute "' + sys.argv[0] + ' --help" for details.'))
        print('')
        sys.exit()

    # --- Sanitize inputs

    path = options.path
    assert( os.path.exists(path) )

    runName = options.runName

    # --- Parse date & time strings
    regex  = r'^(\d{4})-' # YYYY year
    regex += r'(0?[1-9]|1[012])-' # MM month
    regex += r'(0?[1-9]|[12]\d|3[01])-' # DD day 
    regex += r'(0?\d|1\d|2[0-4])-' # HH hours
    regex += r'([0-5]?\d?|60)-'# MM minutes
    regex += r'([0-5]?\d?|60)$' # SS seconds
    t0 = None

    if options.t0:
        g = re.match(regex, options.t0).groups()
        assert( len(g) == 6 ) # Year, Month, Day, Hour, Minute, Second
        t0 = datetime.datetime( int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]) )

    # --- Make sure the datetime string is valid
    t1 = None
    if options.t1:
        g = re.match(regex, options.t1).groups()
        assert( len(g) == 6 ) # Year, Month, Day, Hour, Minute, Second
        t1 = datetime.datetime( int(g[0]), int(g[1]), int(g[2]), int(g[3]), int(g[4]), int(g[5]) )

    # --- Make sure we run ncpu=1 by default
    ncpus = 1
    if options.ncpus:
       ncpus = options.ncpus
            
    # only allow one north or south option
    if ((options.north and options.south)):
        raise Exception('Can only have one --north (-n) or --south (-s) option')

    # make sure config file exists
    if (options.configFile):
        assert( os.path.exists(options.configFile) )

    # make sure we chose a valid movie encoder.
    assert(options.movieEncoder in ['none','ffmpeg','mencoder'])

    return (path, runName, t0, t1, options.north, options.south, options.configFile, ncpus, options.movieEncoder)

def CreateFrames(path, runName, t0, t1, hemisphere, configFile, ncpus):
    assert( (hemisphere == 'north') | (hemisphere == 'south') )

    hemiSelect = {'north': 'North', 'south': 'South'}[hemisphere]

    # Make sure the output directory exisits if not make it
    dirname = os.path.join(path, 'figs', hemisphere)
    if not os.path.exists(dirname):
        os.makedirs( dirname )

    print(('Rendering ' + hemiSelect + 'ern hemisphere, storing frames at ' + dirname))        
    #Now check to make sure the files are correct
    data = pyLTR.Models.MIX(path, runName)
    modelVars = data.getVarNames()
    for v in ['Grid X', 'Grid Y', 
              'Potential North [V]', 'Potential South [V]', 
              'FAC North [A/m^2]', 'FAC South [A/m^2]',
              'Pedersen conductance North [S]', 'Pedersen conductance South [S]', 
              'Hall conductance North [S]', 'Hall conductance South [S]', 
              'Average energy North [keV]', 'Average energy South [keV]',
              'Number flux North [1/cm^2 s]', 'Number flux South [1/cm^2 s]',
              'Electron energy flux North [KeV/cm^2 s]','Electron energy flux South [KeV/cm^2 s]']:
        assert( v in modelVars )

    timeRange = data.getTimeRange()
    if len(timeRange) == 0:
        raise Exception(('No data files found.  Are you pointing to the correct run directory?'))

    index0 = 0
    if t0:
        for i,t in enumerate(timeRange):
            if t0 >= t:
                index0 = i

    index1 = len(timeRange)
    if t1:
        for i,t in enumerate(timeRange):
            if t1 >= t:
                index1 = i                

    print(( 'Extracting MIX quantities for time series over %d time steps.' % (index1-index0) ))
        
    # Output a status bar displaying how far along the computation is.
    #progress = pyLTR.StatusBar(0, index1-index0)
    #progress.start()

    # Pre-compute r and theta
    x = data.read('Grid X', timeRange[index0])
    y = data.read('Grid Y', timeRange[index0])
    theta=n.arctan2(y,x)
    theta[theta<0]=theta[theta<0]+2*n.pi
    # plotting routines now rotate local noon to point up
    #theta=theta+n.pi/2 # to put noon up
    r=n.sqrt(x**2+y**2)
    # plotting routines now expect longitude and colatitude, in radians, stored in dictionaries
    longitude = {'data':theta,'name':r'\phi','units':r'rad'}
    colatitude = {'data':n.arcsin(r),'name':r'\theta','units':r'rad'}
    
    # Deal with the plot options
    if (configFile == None): 
       #potOpts={'min':-100.,'max':100.,'colormap':'bwr'} #'RdBu_r'}
       #facOpts={'min':-1.,'max':1.,'colormap':'bwr'} #'RdBu_r'}
       rengOpts={'min':0.,'max':10.}
       rflxOpts={'min':0.,'max':1.0e9,'format_str':'%.1e','colormap':'jet'}
       reflxOpts={'min':0.,'max':1.0e10,'format_str':'%.1e','colormap':'jet'}
       pedOpts={'min':1.,'max':10.,'colormap':'plasma'}
       halOpts={'min':1.,'max':10.,'colormap':'plasma'}
       engOpts={'min':0.,'max':20.}
       flxOpts={'min':0.,'max':1.0e9,'format_str':'%.1e','colormap':'jet'}
       eflxOpts={'min':0.,'max':1.0e10,'format_str':'%.1e','colormap':'jet'}
       optsObject = {'reng':rengOpts,
                     'rflx':rflxOpts,
                     'reflx':reflxOpts,
                     'ped':pedOpts,
                     'hall':halOpts,
                     'energy':engOpts,
                     'flux':flxOpts,
                     'eflux':eflxOpts}
       configFilename=os.path.join(dirname,'IonSum.config')
       print(("Writing plot config file at " + configFilename))
       f=open(configFilename,'w')
       f.write(pyLTR.yaml.safe_dump(optsObject,default_flow_style=False))
       f.close()
    else:
       f=open(configFile,'r')
       optsDict=pyLTR.yaml.safe_load(f.read())
       f.close()
       if ('pot' in optsDict):
          potOpts = optsDict['pot']
       else:
          potOpts={'min':-100.,'max':100.,'colormap':'RdBu_r'}
       if ('fac' in optsDict):
          facOpts = optsDict['fac']
       else:
          facOpts={'min':-1.,'max':1.,'colormap':'RdBu_r'}
       if ('ped' in optsDict):
          pedOpts = optsDict['ped']
       else:
          pedOpts={'min':1.,'max':8.}
       if ('hall' in optsDict):
          halOpts = optsDict['hall']
       else:
          halOpts={'min':1.,'max':8.}
       if ('energy' in optsDict):
          engOpts = optsDict['energy']
       else:
          engOpts={'min':0.,'max':20.}
       if ('flux' in optsDict):
          flxOpts = optsDict['flux']
       else:
          flxOpts={'min':0.,'max':1.0e9,'format_str':'%.1e'}
          
    args = ((i,time,hemiSelect,hemisphere,data,rengOpts,rflxOpts,reflxOpts,pedOpts,halOpts,engOpts,flxOpts,eflxOpts,longitude,colatitude,path) for i,time in enumerate(timeRange[index0:index1]) )
    #pl=Pool(processes=8)
    #for i,time in enumerate(timeRange[index0:index1]):
    #    #print(i,time)
    #    pl.apply_async(PlotStuff,args=(i,time,hemiSelect,hemisphere,data,potOpts,facOpts,pedOpts,halOpts,engOpts,flxOpts,longitude,colatitude,path))
    #pl=Pool()
    #pl.starmap(PlotStuff,args)
    #pl.close()
    #pl.join()
    print('This system has ',cpu_count(logical= False),' cpus.')
    ncpus = min(int(ncpus),cpu_count(logical=False))
    print('We will use ',ncpus,' cpus for parallelization')
    with Pool(processes=ncpus) as pl:
        pl.starmap(PlotStuff,args)

    return  os.path.join(path,'figs',hemisphere)

def PlotStuff(i,time,hemiSelect,hemisphere,data,rengOpts,rflxOpts,reflxOpts,pedOpts,halOpts,engOpts,flxOpts,eflxOpts,longitude,colatitude,path):
        #try:
           #first read the data
           #vals=data.read('Potential '+hemiSelect+' [V]',time)/1000.0
           #psi={'data':vals,'name':r'$\Phi$','units':r'kV'}
           #vals=data.read('FAC '+hemiSelect+' [A/m^2]',time)*1.0e6
           #fac={'data':vals,'name':r'$J_{||}$',
           #       'units':r'$\mu\mathrm{A/m^2}$'}
           vals1=data.read('Electron average energy '+hemiSelect+' [keV]',time)
           vals2=data.read('Electron energy flux '+hemiSelect+' [KeV/cm^2 s]',time)
           vals3=vals2/vals1
           vals3[n.isnan(vals3)] = 0.
           #vals1[vals2<1.e3] = 0.
           #print('max: ', n.amax(vals2[vals1>10.]), ' min: ',n.amin(vals2[vals1>10.]))
           reng={'data':vals1,'name':r'Energy','units':r'keV'}
           reflx={'data':vals2,'name':r'EFlux','units':r'$\mathrm{keV/cm^2s}$'}
           rflx={'data':vals3,'name':r'Flux','units':r'$\mathrm{1/cm^2s}$'}
           vals1=data.read('Average energy '+hemiSelect+' [keV]',time)
           eng={'data':vals1,'name':r'Energy','units':r'keV'}
           vals2=data.read('Number flux '+hemiSelect+' [1/cm^2 s]',time)
           flx={'data':vals2,'name':r'Flux','units':r'$\mathrm{1/cm^2s}$'}
           vals3=vals2*vals1
           eflx={'data':vals3,'name':r'EFlux','units':r'$\mathrm{keV/cm^2s}$'}
           vals=data.read('Pedersen conductance '+hemiSelect+' [S]',time)
           ped={'data':vals,'name':r'$\Sigma_{P}$','units':r'S'}
           vals=data.read('Hall conductance '+hemiSelect+' [S]',time)
           hal={'data':vals,'name':r'$\Sigma_{H}$','units':r'S'}
           # Now onto the plot
           tt=time.timetuple()
           print(time.isoformat(),i)
           p.figure(i,figsize=(20,12))
           p.figtext(0.5,0.92,'MIX '+hemiSelect+
                     '\n%4d-%02d-%02d  %02d:%02d:%02d' %
                  (tt.tm_year,tt.tm_mon,tt.tm_mday,
                  tt.tm_hour,tt.tm_min,tt.tm_sec),
                  fontsize=14,multialignment='center')
           p.figtext(0.05,0.7,'RCM Precip',fontsize=14,multialignment='right')
           p.figtext(0.05,0.3,'MIX Precip',fontsize=14,multialignment='right')
           ax=p.subplot(241,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,reng,
                  rengOpts,userAxes=ax)#,useMesh=True)
           ax=p.subplot(242,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,rflx,
                  rflxOpts,userAxes=ax)
           ax=p.subplot(243,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,reflx,
                  reflxOpts,userAxes=ax)
           ax=p.subplot(248,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,hal,
                  halOpts,userAxes=ax)
           ax=p.subplot(244,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,ped,
                  pedOpts,userAxes=ax)
           ax=p.subplot(245,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,eng,
                  engOpts,userAxes=ax)
           ax=p.subplot(246,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,flx,
                  flxOpts,userAxes=ax)
           ax=p.subplot(247,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,eflx,
                  eflxOpts,userAxes=ax)
           savefigName = os.path.join(path,'figs',hemisphere,'frame_rsummary_%05d.png'%i)
           p.savefig(savefigName,dpi=100)
           p.close()
        #except KeyboardInterrupt:
        #    # Exit when the user hits CTRL+C.
        #    print('Exiting.')
        #    import sys
        #    sys.exit(0)
        #except:
        #    # Cleanup progress bar if something bad happened.
        #    print('Wha?')
        #    raise

if __name__ == '__main__':

    (dirPath, runName, t0, t1, northOnly, southOnly, configFile, ncpus,  movieEncoder) = parseArgs()

    if movieEncoder == 'ffmpeg':
        movieEncoder = pyLTR.Graphics.ffmpeg()
        movieExtension = 'mp4'
        hasMovieConverter = True
    elif movieEncoder == 'mencoder':
        movieEncoder = pyLTR.Graphics.mencoder()
        movieExtension = 'avi'
        hasMovieConverter = True
    else:
        hasMovieConverter = False

    if (not southOnly):
        frameDir=CreateFrames(dirPath, runName, t0, t1,'north',configFile,ncpus)
        if hasMovieConverter:
            movieEncoder.encode( frameDir, 'frame_rsummary_0', '%04d', 'png', 'mix_rsummary_north.'+movieExtension )
    if (not northOnly):
       frameDir=CreateFrames(dirPath, runName, t0, t1,'south',configFile,ncpus)
       if hasMovieConverter:
            movieEncoder.encode( frameDir, 'frame_rsummary_0', '%04d', 'png', 'mix_rsummary_south.'+movieExtension )
       
