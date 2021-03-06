#!/usr/bin/env python
"""
Script to create 6-panel summary plot of LFM ION run.

Execute lfmionIonSummary.py with the '--about' or '--help'  flags for more information.
"""

# Custom
import pyLTR

# 3rd-party
import pylab as p
import numpy as n

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
    Creates a 6 panel summary plot of LFM ION run
      - path: Path to a directory containing a LFM ION run.
      - runName: Name of run (ie. 'RUN-IDENTIFIER')
      - t0: datetime object of first step to be used in time series
      - t1: datetime object of last step to be used in time series
      - renderNorth: Render only northern hemisphere
      - renderSouth: Render only southern hemisphere
    Execute `lfmionIonSummary.py --help` for more information.
    """
    # additional optparse help available at:
    # http://docs.python.org/library/optparse.html
    # http://docs.python.org/lib/optparse-generating-help.html
    parser = optparse.OptionParser(usage='usage: %prog -p [PATH] [options]',
                                   version=pyLTR.Release.version)

    parser.add_option('-p', '--path', dest='path',
                      default='/Users/schmitt/paraview/testData/March1995_LM_r1432_single',
                      metavar='PATH', help='Path to base run directory containing LFM ION HDF files.')

    parser.add_option('-r', '--runName', dest='runName',
                      default='', metavar='RUN_IDENTIFIER', help='Optional name of run. Leave empty unless there is more than one run in a directory.')

    parser.add_option('-e', '--ext', dest='runExt',
                      default='.hdf', metavar='RUN_EXTENSION', help='Optional run extension. Set to .hdf5 for HDF5 files.')
    
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
                      "the fig directory whenever you run lfmionIonSummary "
                      "(look for a file ending with "
                      "\".config\").")

    parser.add_option("-m", "--movieEncoder", dest="movieEncoder", default="ffmpeg",
                      metavar="[ffmpeg|mencoder|none]",
                      help="Movie encoder. Currently supported formats are ffmpeg (recommended), mencoder or none.")

    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    if options.about:
        print((sys.argv[0] + ' version ' + pyLTR.Release.version))
        print('')
        print('This script searches a path for any LFM ION ionosphere output files and ')
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
    runExt = options.runExt
    
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
            
    # only allow one north or south option
    if ((options.north and options.south)):
        raise Exception('Can only have one --north (-n) or --south (-s) option')

    # make sure config file exists
    if (options.configFile):
      assert( os.path.exists(options.configFile) )

    assert(options.movieEncoder in ['none','ffmpeg','mencoder'])

    return (path, runName, runExt, t0, t1, options.north, options.south, options.configFile, options.movieEncoder)

def CreateFrames(path, runName, runExt, t0, t1, hemisphere, configFile):
    assert( (hemisphere == 'north') | (hemisphere == 'south') )

    hemiSelect = {'north': 'north', 'south': 'south'}[hemisphere]

    # Make sure the output directory exisits if not make it
    dirname = os.path.join(path, 'figs', hemisphere)
    if not os.path.exists(dirname):
        os.makedirs( dirname )

    print(('Rendering ' + hemiSelect + 'ern hemisphere, storing frames at ' + dirname))        
    #Now check to make sure the files are correct
    data = pyLTR.Models.LFMION(path, runName, runExt)
    modelVars = data.getVarNames()
    for v in ['x_interp','y_interp',
              'potnorth','potsouth',
              'curnorth','cursouth',
              'SigmaP_north','SigmaP_south',
              'SigmaH_north','SigmaH_south',
              'avE_north','avE_south',
              'fluxnorth','fluxsouth']:
        assert( v in modelVars )

    timeRange = data.getTimeRange()
    if len(timeRange) == 0:
        raise Exception(('No data files found.  Are you pointing to the correct run directory?'))

    index0 = 0
    if t0:
        for i,t in enumerate(timeRange):
            if t0 >= t:
                index0 = i

    index1 = len(timeRange)-1
    if t1:
        for i,t in enumerate(timeRange):
            if t1 >= t:
                index1 = i                

    print(( 'Extracting LFM ION quantities for time series over %d time steps.' % (index1-index0) ))
        
    # Output a status bar displaying how far along the computation is.
    progress = pyLTR.StatusBar(0, index1-index0)
    progress.start()

    # Pre-compute r and theta
    x = data.read('x_interp', timeRange[index0])
    y = data.read('y_interp', timeRange[index0])
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
       potOpts={'min':-100.,'max':100.,'colormap':'RdBu_r'}
       facOpts={'min':-1.,'max':1.,'colormap':'RdBu_r'}
       pedOpts={'min':1.,'max':20.}
       halOpts={'min':1.,'max':20.}
       engOpts={'min':0.,'max':20.}
       flxOpts={'min':0.,'max':1.0e9,'format_str':'%.1e'}
       optsObject = {'pot':potOpts,
                     'fac':facOpts,
                     'ped':pedOpts,
                     'hall':halOpts,
                     'energy':engOpts,
                     'flux':flxOpts}
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
          
     

    for i,time in enumerate(timeRange[index0:index1]):
        try:
           #first read the data
           vals=data.read('pot'+hemiSelect,time)/1000.0
           psi={'data':vals,'name':r'$\Phi$','units':r'kV'}
           vals=data.read('cur'+hemiSelect,time)*1.0e6
           fac={'data':vals,'name':r'$J_{||}$',
                  'units':r'$\mu\mathrm{A/m^2}$'}
           vals=data.read('avE_'+hemiSelect,time)
           eng={'data':vals,'name':r'Energy','units':r'keV'}
           vals=data.read('flux'+hemiSelect,time)
           flx={'data':vals,'name':r'Flux','units':r'$\mathrm{1/cm^2s}$'}
           vals=data.read('SigmaP_'+hemiSelect,time)
           ped={'data':vals,'name':r'$\Sigma_{P}$','units':r'S'}
           vals=-1.0*data.read('SigmaH_'+hemiSelect,time)
           hal={'data':vals,'name':r'$\Sigma_{H}$','units':r'S'}
           # Now onto the plot
           tt=time.timetuple()
           p.figure(figsize=(16,12))
           p.figtext(0.5,0.92,'LFM ION '+hemiSelect+
                     '\n%4d-%02d-%02d  %02d:%02d:%02d' %
                  (tt.tm_year,tt.tm_mon,tt.tm_mday,
                  tt.tm_hour,tt.tm_min,tt.tm_sec),
                  fontsize=14,multialignment='center')
           ax=p.subplot(231,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,psi,
                  potOpts,userAxes=ax,useMesh=True)
           ax=p.subplot(234,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,fac,
                  facOpts,userAxes=ax,useMesh=True)
           ax=p.subplot(232,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,ped,
                  pedOpts,userAxes=ax,useMesh=True)
           ax=p.subplot(235,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,hal,
                  halOpts,userAxes=ax,useMesh=True)
           ax=p.subplot(233,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,eng,
                  engOpts,userAxes=ax,useMesh=True)
           ax=p.subplot(236,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,flx,
                  flxOpts,userAxes=ax,useMesh=True)
           savefigName = os.path.join(path,'figs',hemisphere,'frame_summary_%05d.png'%i)
           p.savefig(savefigName,dpi=100)
           p.close()
           progress.increment()
        except KeyboardInterrupt:
            # Exit when the user hits CTRL+C.
            progress.stop()
            progress.join()            
            print('Exiting.')
            import sys
            sys.exit(0)
        except:
            # Cleanup progress bar if something bad happened.
            progress.stop()
            progress.join()
            raise
    progress.stop()
    progress.join()

    return  os.path.join(path,'figs',hemisphere)

if __name__ == '__main__':

    (dirPath, runName, runExt, t0, t1, northOnly, southOnly, configFile, movieEncoder) = parseArgs()


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
        frameDir=CreateFrames(dirPath, runName, runExt, t0, t1,'north',configFile)
        if hasMovieConverter:
            movieEncoder.encode( frameDir, 'frame_summary_00', '%03d', 'png', 'lfm_summary_north.'+movieExtension )
    if (not northOnly):
       frameDir=CreateFrames(dirPath, runName, runExt, t0, t1,'south',configFile)
       if hasMovieConverter:
            movieEncoder.encode( frameDir, 'frame_summary_00', '%03d', 'png', 'lfm_summary_south.'+movieExtension )
