#!/usr/bin/env python
"""
Script to create 6-panel summary plot of MIX run.

Execute mixEfieldSummary.py with the '--about' or '--help'  flags for more information.
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
    Creates a 6 panel summary plot of MIX run
      - path: Path to a directory containing a MIX run.
      - runName: Name of run (ie. 'RUN-IDENTIFIER')
      - t0: datetime object of first step to be used in time series
      - t1: datetime object of last step to be used in time series
      - renderNorth: Render only northern hemisphere
      - renderSouth: Render only southern hemisphere
    Execute `mixEfieldSummary.py --help` for more information.
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
                      "the fig directory whenever you run mixEfieldSummary "
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
            
    # only allow one north or south option
    if ((options.north and options.south)):
        raise Exception('Can only have one --north (-n) or --south (-s) option')

    # make sure config file exists
    if (options.configFile):
      assert( os.path.exists(options.configFile) )

    assert(options.movieEncoder in ['none','ffmpeg','mencoder'])

    return (path, runName, t0, t1, options.north, options.south, options.configFile, options.movieEncoder)

def CreateFrames(path, runName, t0, t1, hemisphere, configFile):
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
              'Number flux North [1/cm^2 s]', 'Number flux South [1/cm^2 s]']:
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

    print(( 'Extracting MIX quantities for time series over %d time steps.' % (index1-index0) ))
        
    # Output a status bar displaying how far along the computation is.
    progress = pyLTR.StatusBar(0, index1-index0)
    progress.start()

    # Pre-compute r and theta
    x = data.read('Grid X', timeRange[index0])
    xdict={'data':x*6500e3,'name':'X','units':r'm'}
    y = data.read('Grid Y', timeRange[index0])
    ydict={'data':y*6500e3,'name':'Y','units':r'm'}
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
       pedOpts={'min':1.,'max':8.}
       halOpts={'min':1.,'max':8.}
       engOpts={'min':0.,'max':20.}
       flxOpts={'min':0.,'max':1.0e9,'format_str':'%.1e'}
       ephiOpts={'min':-200.,'max':200.}
       ethetaOpts={'min':-200.,'max':200.}
       emagOpts={'min':0,'max':200,'colormap':'hsv'}
       jhOpts={'min':0,'max':200}
       optsObject = {'ped':pedOpts,
                     'ephi':ephiOpts,        
                     'etheta':ethetaOpts,        
                     'emag':emagOpts,        
                     'joule heat':jhOpts}
       configFilename=os.path.join(dirname,'EfieldSum.config')
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
       if ('ephi' in optsDict):
          ephiOpts = optsDict['ephi']
       else:
          ephiOpts={'min':-200.,'max':200.}
       if ('etheta' in optsDict):
          ethetaOpts = optsDict['etheta']
       else:
          ethetaOpts={'min':-200.,'max':200.}
       if ('emag' in optsDict):
          emagOpts = optsDict['emag']
       else:
          emagOpts={'min':0,'max':200,'colormap':'hsv'}
       if ('joule heat' in optsDict):
          jhOpts = optsDict['joule heat']
       else:
          jhOpts={'min':0,'max':200}

    for i,time in enumerate(timeRange[index0:index1]):
        try:
           #first read the data
           vals=data.read('Potential '+hemiSelect+' [V]',time)/1000.0
           psi={'data':vals,'name':r'$\Phi$','units':r'kV'}
           vals=data.read('Pedersen conductance '+hemiSelect+' [S]',time)
           ped={'data':vals,'name':r'$\Sigma_{P}$','units':r'S'}
           angles_mid,efielddict=pyLTR.Physics.MIXCalcs.efieldDict(xdict,ydict,
                                psi,6500.e3)
           emagdict={'data':
                     n.sqrt(efielddict[0]['data']**2+efielddict[1]['data']**2),
                     'name':r'|E|','units':'mV/m'}
           jhdict=pyLTR.Physics.MIXCalcs.jouleDict(efielddict,ped)
           # Now onto the plot
           tt=time.timetuple()
           p.figure(figsize=(28,5))
           p.figtext(0.5,0.92,'MIX '+hemiSelect+
                     '\n%4d-%02d-%02d  %02d:%02d:%02d' %
                  (tt.tm_year,tt.tm_mon,tt.tm_mday,
                  tt.tm_hour,tt.tm_min,tt.tm_sec),
                  fontsize=14,multialignment='center')
           ax=p.subplot(141,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,efielddict[0],
                  ephiOpts,userAxes=ax)
           ax=p.subplot(142,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,efielddict[1],
                  ethetaOpts,userAxes=ax)
           ax=p.subplot(143,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,emagdict,
                  emagOpts,userAxes=ax)
           ax=p.subplot(144,polar=True)
           pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,jhdict,
                  jhOpts,userAxes=ax)
           savefigName = os.path.join(path,'figs',hemisphere,'frame_Efield_%05d.png'%i)
           p.savefig(savefigName,dpi=100)
           p.clf()
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

    (dirPath, runName, t0, t1, northOnly, southOnly, configFile, movieEncoder) = parseArgs()


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
        frameDir=CreateFrames(dirPath, runName, t0, t1,'north',configFile)
        if hasMovieConverter:
            movieEncoder.encode( frameDir, 'frame_Efield_00', '%03d', 'png', 'mix_Efield_north.'+movieExtension )

    if (not northOnly):
        frameDir=CreateFrames(dirPath, runName, t0, t1,'south',configFile)
        if hasMovieConverter:
            movieEncoder.encode( frameDir, 'frame_Efield_00', '%03d', 'png', 'mix_Efield_south.'+movieExtension )

       
