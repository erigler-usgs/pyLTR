#!/usr/bin/env python
"""
Script to create 6-panel summary plot of LFM MAG run.

Execute lfmCutPlanes.py with the '--about' or '--help'  flags for more information.
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

def hasMencoder():
    not_found_msg = """
    The mencoder command was not found;
    mencoder is used by this script to make an avi file from a set of pngs.
    It is typically not installed by default on linux distros because of
    legal restrictions, but it is widely available.
    http://www.mplayerhq.hu/
    """

    try:
        subprocess.call(['mencoder'])
	hasEncoder = True
    except OSError:
        print(not_found_msg)
        hasEncoder = False

    return hasEncoder

def parseArgs():
    """
    Creates XY and XZ summary plot of LFM run
      - path: Path to a directory containing a LFM run.
      - runName: Name of run (ie. 'RUN-IDENTIFIER')
      - t0: datetime object of first step to be used in time series
      - t1: datetime object of last step to be used in time series
      - xz: Render only XZ plane
      - xy: Render only XY plane
    Execute `lfmCutPlanes.py --help` for more information.
    """
    # additional optparse help available at:
    # http://docs.python.org/library/optparse.html
    # http://docs.python.org/lib/optparse-generating-help.html
    parser = optparse.OptionParser(usage='usage: %prog -p [PATH] [options]',
                                   version=pyLTR.Release.version)

    parser.add_option('-p', '--path', dest='path',
                      default='/Users/schmitt/paraview/testData/March1995_LM_r1432_single',
                      metavar='PATH', help='Path to base run directory containing LFM MAG HDF files.')

    parser.add_option('-r', '--runName', dest='runName',
                      default='', metavar='RUN_IDENTIFIER', help='Optional name of run. Leave empty unless there is more than one run in a directory.')
   
    parser.add_option('-e', '--ext', dest='runExt',
                      default='.hdf', metavar='RUN_EXTENSION', help='Optional run extension. Set to .hdf5 for HDF5 files.')
    
    parser.add_option('-f', '--first', dest='t0',
                      default='', metavar='YYYY-MM-DD-HH-MM-SS', help='Date & Time that should be the first element of the time series')

    parser.add_option('-l', '--last', dest='t1',
                      default='', metavar='YYYY-MM-DD-HH-MM-SS', help='Date & Time of last element for the time series')
    parser.add_option('-z', '--xzplane', dest='xzplane', 
                      default=False, action='store_true',
                      help='Only render XZ plane.')

    parser.add_option('-y', '--xyplane', dest='xyplane',
                       default=False, action='store_true',
                      help='Only render XY plane.')
    parser.add_option("-c", "--configFile", dest="configFile",
                      default=None, action="store", metavar="FILE",
                      help="Path to a configuration file.  Note: "
                      "these config files are automatically written to "
                      "the fig directory whenever you run lfmCutPlanes "
                      "(look for a file ending with "
                      "\".config\").")

    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    
    if options.about:
        print((sys.argv[0] + ' version ' + pyLTR.Release.version))
        print('')
        print('This script searches a path for any LFM output files and ')
        print('makes a cut summary plot ')
        print('for both the XZ and XY planes.')
        print('')
        print('To limit the time range, use the "--first" and/or "--last" flags.')
        print('To render only xzplane or xyplane use the "--xzplane" or "--xyplane" flags.')
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
            
    # only allow one xzplane or xyplane option
    if ((options.xzplane and options.xyplane)):
        raise Exception('Can only have one --xzplane (-n) or --xyplane (-s) option')

    # make sure config file exists
    if (options.configFile):
      assert( os.path.exists(options.configFile) )

    return (path, runName, runExt, t0, t1, options.xzplane, options.xyplane, options.configFile)

def CreateFrames(path, runName, runExt, t0, t1, plane, configFile):
    assert( (plane == 'xzplane') | (plane == 'xyplane') )

    planeSelect = {'xzplane': 'xzplane', 'xyplane': 'xyplane'}[plane]

    # Make sure the output directory exisits if not make it
    dirname = os.path.join(path, 'figs', plane)
    if not os.path.exists(dirname):
        os.makedirs( dirname )

    print(('Rendering ' + planeSelect + 'plane, storing frames at ' + dirname))        
    #Now check to make sure the files are correct
    data = pyLTR.Models.LFM(path, runName,ext=runExt)
    modelVars = data.getVarNames()
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

    print(( 'Extracting LFM MAG quantities for time series over %d time steps.' % (index1-index0) ))
        
    # Output a status bar displaying how far along the computation is.
    progress = pyLTR.StatusBar(0, index1-index0)
    progress.start()

    # Deal with the plot options
    if (configFile == None): 
       vxOpts={'min':-800.,'max':800.,'colormap':'RdBu_r'}
       vyOpts={'min':-100.,'max':100.,'colormap':'RdBu_r'}
       vzOpts={'min':-100.,'max':100.,'colormap':'RdBu_r'}
       bxOpts={'min':-10.,'max':10.}
       byOpts={'min':-10.,'max':10.}
       bzOpts={'min':-10.,'max':10.}
       rhoOpts={'min':0,'max':50}
       optsObject = {'vx':vxOpts,
                     'vy':vyOpts,
                     'vz':vzOpts,
                     'bx':bxOpts,
                     'by':byOpts,
                     'bz':bzOpts,
                     'rho':rhoOpts}
       configFilename=os.path.join(dirname,'MHDPlanes.config')
       print(("Writing plot config file at " + configFilename))
       f=open(configFilename,'w')
       f.write(pyLTR.yaml.safe_dump(optsObject,default_flow_style=False))
       f.close()
    else:
       f=open(configFile,'r')
       optsDict=pyLTR.yaml.safe_load(f.read())
       f.close()
       if ('vx' in optsDict):
          vxOpts = optsDict['vx']
       else:
          vxOpts={'min':-800.,'max':800.,'colormap':'RdBu_r'}
       if ('vy' in optsDict):
          vyOpts = optsDict['vy']
       else:
          vyOpts={'min':-100.,'max':100.,'colormap':'RdBu_r'}
       if ('vz' in optsDict):
          vzOpts = optsDict['vz']
       else:
          vzOpts={'min':-100.,'max':100.,'colormap':'RdBu_r'}
       if ('bx' in optsDict):
          bxOpts = optsDict['bx']
       else:
          bxOpts={'min':-10.,'max':10.}
       if ('by' in optsDict):
          byOpts = optsDict['by']
       else:
          byOpts={'min':-10.,'max':10.}
       if ('bz' in optsDict):
          bzOpts = optsDict['bz']
       else:
          bzOpts={'min':-10.,'max':10.}
       if ('rho' in optsDict):
          bzOpts = optsDict['rho']
       else:
          rhoOpts={'min':0.,'max':50.}

    vals=data.getEqSlice('X_grid',timeRange[0])/6.38e8
    xeq={'data':vals,'name':r'$X$','units':r'$R_E$'}
    vals=data.getEqSlice('Y_grid',timeRange[0])/6.38e8
    yeq={'data':vals,'name':r'$X$','units':r'$R_E$'}
    vals=data.getEqSlice('Z_grid',timeRange[0])/6.38e8
    zeq={'data':vals,'name':r'$X$','units':r'$R_E$'}
    vals=data.getMerSlice('X_grid',timeRange[0])/6.38e8
    xmer={'data':vals,'name':r'$X$','units':r'$R_E$'}
    vals=data.getMerSlice('Y_grid',timeRange[0])/6.38e8
    ymer={'data':vals,'name':r'$X$','units':r'$R_E$'}
    vals=data.getMerSlice('Z_grid',timeRange[0])/6.38e8
    zmer={'data':vals,'name':r'$X$','units':r'$R_E$'}
    for i,time in enumerate(timeRange[index0:index1]):
        try:
           #first read the data
           vals=data.getEqSlice('vx_',time)/1000.0
           vx={'data':vals,'name':r'$V_X$','units':r'km/s'}
           vals=data.getEqSlice('vy_',time)/1000.0
           vy={'data':vals,'name':r'$V_Y$',
                  'units':r'km/s'}
           vals=data.getEqSlice('vz_',time)/1000.0
           vz={'data':vals,'name':r'$V_Z$','units':r'km/s'}
           vals=data.getEqSlice('bx_',time)
           bx={'data':vals,'name':r'$B_X$','units':r'nT'}
           vals=data.getEqSlice('by_',time)
           by={'data':vals,'name':r'$B_Y$','units':r'nT'}
           vals=-1.0*data.getEqSlice('bz_',time)
           bz={'data':vals,'name':r'$B_Z$','units':r'S'}
           # Now onto the plot
           tt=time.timetuple()
           p.figure(figsize=(16,12))
           p.figtext(0.5,0.92,'LFM '+
                     '\n%4d-%02d-%02d  %02d:%02d:%02d' %
                  (tt.tm_year,tt.tm_mon,tt.tm_mday,
                  tt.tm_hour,tt.tm_min,tt.tm_sec),
                  fontsize=14,multialignment='center')
           ax=p.subplot(231)
           x=xeq['data']
           y=yeq['data']
           pyLTR.Graphics.CutPlane.CutPlaneDict(x,y,vx,
                  vxOpts,userAxes=ax)
           ax=p.subplot(234)
           pyLTR.Graphics.CutPlane.CutPlaneDict(x,y,vy,
                  vyOpts,userAxes=ax)
           ax=p.subplot(232)
           pyLTR.Graphics.CutPlane.CutPlaneDict(x,y,vz,
                  vzOpts,userAxes=ax)
           ax=p.subplot(235)
           pyLTR.Graphics.CutPlane.CutPlaneDict(x,y,bx,
                  bxOpts,userAxes=ax)
           ax=p.subplot(233)
           pyLTR.Graphics.CutPlane.CutPlaneDict(x,y,by,
                  byOpts,userAxes=ax)
           ax=p.subplot(236)
           pyLTR.Graphics.CutPlane.CutPlaneDict(x,y,bz,
                  bzOpts,userAxes=ax)
           savefigName = os.path.join(path,'figs',plane,'frame_%05d.png'%i)
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

    return  os.path.join(path,'figs',plane)

def createMovie(dirPath, imageExt, outputAvi):
    """
    stitch images together using Mencoder to create a movie.  Each
    image will become a single frame in the movie.
    """

    #
    # We want to use Python to make what would normally be a command line
    # call to Mencoder.  Specifically, the command line call we want to
    # emulate is (without the initial '#'):
    # mencoder mf://*.png -mf type=png:w=800:h=600:fps=25 -ovc lavc -lavcopts
    # vcodec=mpeg4 -oac copy -o output.avi
    # See the MPlayer and Mencoder documentation for details.
    #

    command = ('mencoder',
               'mf://'+os.path.join(dirPath, '*.'+imageExt),
               '-mf',
               'type='+imageExt+':fps=15',
               '-ovc',
               'lavc',
               '-lavcopts',
               'vcodec=mpeg4',
               '-oac',
               'copy',
               '-o',
               outputAvi)

    #os.spawnvp(os.P_WAIT, 'mencoder', command)

    print(("\n\nabout to execute:\n%s\n\n" % ' '.join(command)))
    subprocess.call(command)

    print(("\n\n The movie was written to '%s'" % outputAvi))

    print(("\n\n You may want to delete "+os.path.join(dirPath, '*.'+imageExt)+" now.\n\n"))

if __name__ == '__main__':

    (dirPath, runName, runExt, t0, t1, xzplaneOnly, xyplaneOnly, configFile) = parseArgs()
    print(('Run Extension ' + runExt))

    hasMovieConverter=hasMencoder()

    if (not xyplaneOnly):
        frameDir=CreateFrames(dirPath, runName, runExt, t0, t1,'xzplane',configFile)
        if hasMovieConverter:
            createMovie( frameDir, 'png', os.path.join(frameDir,'output_xzplane.avi'))
    if (not xzplaneOnly):
       frameDir=CreateFrames(dirPath, runName, runExt, t0, t1,'xyplane',configFile)
       if hasMovieConverter:
           createMovie( frameDir, 'png', os.path.join(frameDir,'output_xyplane.avi'))
       
