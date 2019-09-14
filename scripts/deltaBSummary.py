#!/usr/bin/env python
"""
pyLTR deltaB summary snapshot generator
 Execute 'deltaBSummary.py --help' for more info on command-line usage.
 Examine member function docstrings for module usage
"""

# Custom
import pyLTR

# 3rd-party
import pylab as p
import numpy as n

# Standard
import pickle
import scipy.io as sio
import datetime
import math
import optparse
import os
import re
import sys
import subprocess

def parseArgs():
    """
    Creates a 4 panel summary plot of ground deltaB derived from 1) MIX
    ionospheric currents; 2) FACs; 3) magnetospheric currents, and their sum.
      - path: Path to a directory containing a MIX run.
      - runName: Name of run (ie. 'RUN-IDENTIFIER')
      - t0: datetime object of first step to be used in time series
      - t1: datetime object of last step to be used in time series
      - renderNorth: Render only northern hemisphere
      - renderSouth: Render only southern hemisphere
    Execute `deltaBSummary.py --help` for more information.
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
    
    
#    parser.add_option("-i", "--ignorePKL", dest="ignorePKL", action="store_true", default=False,
#                      help="Ignore existing PKL files and recalculate all summary data")
    
    parser.add_option("-i", "--ignoreBinary", dest="ignoreBinary", action="store_true", default=False,
                      help="Ignore existing binary files and recalculate all summary data")
    
    parser.add_option("-b", "--binaryType", dest="binaryType", action="store", default="pkl",
                      help="Set type of binary output file")
    
    parser.add_option("-g", "--geoGrid", dest="geoGrid", action="store_true", default=False,
                      help="Assume GEOgraphic coordinates instead of typical SM")
    
    
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
        print('This script searches path for any LFM and MIX ionosphere output files,')
        print('generates magnetospheric, FAC, and ionospheric currents, then calculates')
        print('for hemispheric grids: dB_phi, dB_theta, and dB_rho.')
        print('')
        print('This script generates the following files:')
        print('  1a. runPath/figs/north|south/frame_deltaB_yyyy-mm-ddTHH-MM-SSZ.pkl -')
        print('      a pickle file holding hemispheric snapshots of gridded delta B,')
        print('      decomposed into ionospheric, FAC, and magnetospheric contributions,')
        print('      all in spherical coordinates (i.e., phi,theta,rho).')
        print('  1b. runPath/figs/north|south/frame_deltaB_yyyy-mm-ddTHH-MM-SSZ.mat -')
        print('      a mat binary file holding hemispheric snapshots of gridded delta B,')
        print('      decomposed into ionospheric, FAC, and magnetospheric contributions,')
        print('      all in spherical coordinates (i.e., phi,theta,rho).')
        print('  2.  runPath/figs/north|south/frame_deltaB_yyyy-mm-ddTHH-MM-SSZ.png')
        print('      a PNG image file showing 4 panels, showing the combined delta B,')
        print('      and each of the ionospheric, FAC, and magnetospheric constituents.')
        print('      Horizontal vector components are a represented by arrows, while')
        print('      vertical vector components are represented by colored contours.')
        print('  3.  current_directory/deltaB_north.mp4|deltaB_south.mp4 - ')
        print('      animation of evolving deltaB summary PNGs')
        print('')
        print('FIXME: southern hemisphere summary plots are presented from a point')
        print('      of view below the south pole, looking up. Likewise for the data')
        print('      files. This is not technically a right-handed frame of reference,')
        print('      so any analysis of southern hemisphere data must rotate fields')
        print('      180 degrees about the X axis, or 0 longitude, 0 latitude line.')
        print('      This odd data file convention is consistent with how southern')
        print('      hemisphere data is stored in MIX files, but there is a strong')
        print('      argument to just store everything assuming the same (northern)')
        print('      POV, and rotate the POV for display purposes only; this will')
        print('      require significant work')
        print('')
        print('To limit the time range, use the "--first" and/or "--last" flags.')
        print('To treat observatories as geographic coordinates, use the "--geoGrid" flag.')
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

    assert(options.movieEncoder.lower() in ['none','ffmpeg','mencoder'])

    return (path, runName, t0, t1, options.north, options.south, options.geoGrid, 
            options.ignoreBinary, options.binaryType, options.configFile, options.movieEncoder)


def _roundTime(dt=None, roundTo=60):
   """
   Support routine shamelessly stolen form stackoverflow.com:
   
   Round a datetime object to any time laps in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   if dt == None : dt = datetime.datetime.now()
   seconds = (dt - dt.min).seconds
   # // is a floor division, not a comment on following line:
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def CreateFrames(path='./', run='', 
                 t0='', t1='', 
                 hemisphere='north', geoGrid=False, 
                 ignoreBinary=False, binaryType='pkl',
                 configFile=None):
    """
    Compute deltaBs hemispheric grid given LFM-MIX output files in path.
    
    Computes:
      pngFiles  - a list of PNG filenames, each corresponding to a snapshot in
                  time of LFM-MIX ground deltaB vector fields
                  NOTE: while the output is a list of filenames, the function
                        generates binary pkl/mat files that can be read in by
                        other software for further analysis.
                  FIXME: southern hemisphere summary plots are presented from a
                        point of view below the south pole, looking up. Likewise
                        for the data files. This is not technically a right-handed 
                        frame of reference, so any analysis of southern hemisphere 
                        data must rotate fields 180 degrees about the X axis, or 
                        0 longitude, 0 latitude line. This odd data file convention 
                        is consistent with how southern hemisphere data is stored 
                        in MIX files, but there is a strong argument to just store 
                        everything assuming the same (northern) POV, and rotate 
                        to a southern POV for display purposes only; this will 
                        require significant work.
                        
    Requires:
      Nothing, all inputs are optional
    
    Optional:
      path      - path to data directory holding LFM and MIX model output files
                  (default is current directory)
      run       - output filename prefix identifying LFM-MIX run (i.e., the part
                  of the filename prior to [mhd|mix]_yyyy-mm-ddTHH-MM-SSZ.hdf)
                  (default is any mhd|mix files in path)
      t0        - datetime object specifying the earliest of available time step
                  to include in the extraction
                  (default is earliest available)
      t1        - datetime object specfiying the latest of available time step
                  to include in the extraction
                  (default is last available)
      hemisphere - specify 'north' or 'south' hemisphere
                  (default is 'north')
      geoGrid   - if True, assume observatory coordinates are in geographic
                  coordinates rather than solar magnetic; same for outputs
                  (default is False)
      ignoreBinary - if True, ignore any pre-computed binary files and re-
                  compute everything from scratch; NOTE: individual binary files
                  will be ignored anyway if they are incompatible with specified
                  inputs, but this option avoids reading the binary file entirely.
                  (default is False)
      binaryType - binary type to generate, NOT to read in...routine looks for
                  PKL files first, then mat files, then proceeds to re-compute
                  if neither are available.
                  (default is 'pkl')
      configFile - specifies plotting config file; if None, default config file
                   is path/figs/north|south/deltaBSum.config; if this doesn't
                   exist, create new one with default config parameters.
    """
    
    assert( (hemisphere == 'north') | (hemisphere == 'south') )
    
    hemiSelect = {'north': 'North', 'south': 'South'}[hemisphere]

    # Make sure the output directory exisits if not make it
    dirname = os.path.join(path, 'figs', hemisphere)
    if not os.path.exists(dirname):
        os.makedirs( dirname )

    print(('Rendering ' + hemiSelect + 'ern hemisphere, storing frames at ' + dirname))        
    #Now check to make sure the MIX files are correct
    dMIX = pyLTR.Models.MIX(path, run)
    modelVars = dMIX.getVarNames()
    for v in ['Grid X', 'Grid Y', 
              'Potential North [V]', 'Potential South [V]', 
              'FAC North [A/m^2]', 'FAC South [A/m^2]',
              'Pedersen conductance North [S]', 'Pedersen conductance South [S]', 
              'Hall conductance North [S]', 'Hall conductance South [S]']:
        assert( v in modelVars )
    
    
    timeRange = dMIX.getTimeRange()
    
    #Now check to make sure the LFM files are correct
    dLFM = pyLTR.Models.LFM(path, run)
    modelVars = dLFM.getVarNames()
    for v in ['X_grid', 'Y_grid', 'Z_grid', 
              'bx_', 'by_', 'bz_']:
        assert( v in modelVars )

    # check that LFM output timeRanges are exactly the same as MIX output timeRanges
    trLFM = dLFM.getTimeRange()
    # _roundTime() rounds to nearest minute by default, which should suffice here
    # NOTE: we do NOT change the time stamps at all, just make sure they match
    #       to a 1-minute tolerance
    if list(map(_roundTime, timeRange)) != list(map(_roundTime, trLFM)):
       raise Exception(('Mismatched MIX and LFM output files'))
    
    
    if len(timeRange) == 0:
        raise Exception(('No data files found.  Are you pointing to the correct run directory?'))
    
    
    
    ## Original code defaulted to entire data set if the user-supplied time range
    ## fell outside of the available data, almost certainly not a desired result.
    ## Now if the user requests a time range that falls completely outside of the
    ## available data, and exception is raised.
    
    ##index0 = 0
    ##if t0:
    ##    for i,t in enumerate(timeRange):
    ##        if t0 >= t:
    ##            index0 = i
    ##
    ###index1 = len(timeRange)-1
    ##index1 = len(timeRange) # we were skipping the last valid time step
    ##if t1:
    ##    for i,t in enumerate(timeRange):
    ##        if t1 >= t:
    ##            #index1 = i
    ##            index1 = i + 1 # we were skipping the last valid time step
    
    
    if t0:
       if t1 and t1 < timeRange[0]: # upper time stamp below lowest available time stamp
          raise Exception('Requested time range falls outside available data')
       if t0 > timeRange[-1]: # lower time stamp above highest available time stamp
          raise Exception('Requested time range falls outside available data')
       
       for i,t in enumerate(timeRange):
          if t0 >= t:
             index0 = i
    else:
       index0 = 0
    
    
    if t1:
       if t0 and t0 > timeRange[-1]: # lower time stamp above highest available time stamp
          raise Exception('Requested time range falls outside available data')
       if t1 < timeRange[0]: # upper time stamp below lowest available time stamp
          raise Exception('Requested time range falls outside available data')
       
       for i,t in enumerate(timeRange):
          if t1 >= t:
             index1 = i+1
    else:
       index1 = len(timeRange)
    
    
    if index1 > index0:
       print(( 'Extracting LFM and MIX quantities for time series over %d time steps.' % (index1-index0) ))
    else:
       raise Exception('Requested time range is invalid')
    
    
    
    # Output a status bar displaying how far along the computation is.
    progress = pyLTR.StatusBar(0, index1-index0)
    progress.start()

    # Pre-compute r and theta
    x = dMIX.read('Grid X', timeRange[index0])
    xdict={'data':x*6500e3,'name':'X','units':r'm'}
    y = dMIX.read('Grid Y', timeRange[index0])
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
    if (configFile == None and os.path.exists(os.path.join(dirname,'deltaBSum.config')) ):
       configFile = os.path.join(dirname,'deltaBSum.config')
    
    if configFile == None:
       # scalar radial magnetic pertubations
       dBradialTotOpts={'min':-100,'max':100,'colormap':'jet'}
       dBradialIonOpts={'min':-100,'max':100,'colormap':'jet'}
       dBradialFACOpts={'min':-100,'max':100,'colormap':'jet'}
       dBradialMagOpts={'min':-100,'max':100,'colormap':'jet'}
       
       # 2D vector horizontal perturbations
       dBhvecTotOpts={'width':.0025,'scale':1e3,'pivot':'middle'}
       dBhvecIonOpts={'width':.0025,'scale':1e3,'pivot':'middle'}
       dBhvecFACOpts={'width':.0025,'scale':1e3,'pivot':'middle'}
       dBhvecMagOpts={'width':.0025,'scale':1e3,'pivot':'middle'}
              
       # place all config dictionaries in one big dictionary
       optsObject = {'dBradialTot':dBradialTotOpts,
                     'dBradialIon':dBradialIonOpts,
                     'dBradialFAC':dBradialFACOpts,
                     'dBradialMag':dBradialMagOpts,
                     'dBhvecTot':dBhvecTotOpts,
                     'dBhvecIon':dBhvecIonOpts,
                     'dBhvecFAC':dBhvecFACOpts,
                     'dBhvecMag':dBhvecMagOpts,
                     'altPole':altPoleOpts}
       configFilename=os.path.join(dirname,'deltaBSum.config')
       print(("Writing plot config file at " + configFilename))
       f=open(configFilename,'w')
       f.write(pyLTR.yaml.safe_dump(optsObject,default_flow_style=False))
       f.close()
    else:
       f=open(configFile,'r')
       optsDict=pyLTR.yaml.safe_load(f.read())
       f.close()
       if ('dBradialTot' in optsDict):
          dBradialTotOpts = optsDict['dBradialTot']
       else:
          dBradialTotOpts={'min':-100.,'max':100.,'colormap':'jet'}
       
       if ('dBradialIon' in optsDict):
          dBradialIonOpts = optsDict['dBradialIon']
       else:
          dBradialIonOpts={'min':-100.,'max':100.,'colormap':'jet'}
           
       if ('dBradialFAC' in optsDict):
          dBradialFACOpts = optsDict['dBradialFAC']
       else:
          dBradialFACOpts={'min':-100.,'max':100.,'colormap':'jet'}
       
       if ('dBradialMag' in optsDict):
          dBradialMagOpts = optsDict['dBradialMag']
       else:
          dBradialMagOpts={'min':-100.,'max':100.,'colormap':'jet'}
       
       
       if ('dBhvecTot' in optsDict):
          dBhvecTotOpts = optsDict['dBhvecTot']
       else:
          dBhvecTotOpts={'min':-100.,'max':100.,'colormap':'jet'}
       
       if ('dBhvecIon' in optsDict):
          dBhvecIonOpts = optsDict['dBhvecIon']
       else:
          dBhvecIonOpts={'min':-100.,'max':100.,'colormap':'jet'}
           
       if ('dBhvecFAC' in optsDict):
          dBhvecFACOpts = optsDict['dBhvecFAC']
       else:
          dBhvecFACOpts={'min':-100.,'max':100.,'colormap':'jet'}
       
       if ('dBhvecMag' in optsDict):
          dBhvecMagOpts = optsDict['dBhvecMag']
       else:
          dBhvecMagOpts={'min':-100.,'max':100.,'colormap':'jet'}
       
       if ('altPole' in optsDict):
          altPoleOpts = optsDict['altPole']
       else:
          altPoleOpts = {'poleMarker1':'x', 'poleMarker2':'x',
                         'poleSize1':7, 'poleSize2':5,
                         'poleWidth1':3, 'poleWidth2':1,
                         'poleColor1':'blue', 'poleColor2':'white'}
       
    # initialize output list
    pngFilenames = []
    for i,time in enumerate(timeRange[index0:index1]):
        
        try:
           
           try:
              
              # ignore binary file even if one exists
              if ignoreBinary:
                 raise Exception
              
              # look for a .pkl file that already holds all the data required for
              # subsequent plots before recalculating all the derived data...if
              # this fails, look for a .mat file, if this fails, fall through to 
              # recalculate all summary data
              filePrefix = os.path.join(path,'figs',hemisphere)
              
              # this is a possible race condition, but try/except just doesn't do what I want
              if os.path.exists(os.path.join(filePrefix,
                                             'frame_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.pkl'%
                                             (time.year,time.month,time.day,
                                              time.hour,time.minute,time.second))):
                 
                 binFilename = os.path.join(filePrefix,
                                            'frame_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.pkl'%
                                            (time.year,time.month,time.day,
                                             time.hour,time.minute,time.second))
                 fh=open(binFilename,'rb')
                 allDict = pickle.load(fh)
                 fh.close()
                 
              elif os.path.exists(os.path.join(filePrefix,
                                               'frame_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.mat'%
                                               (time.year,time.month,time.day,
                                                time.hour,time.minute,time.second))):
                 
                 binFilename = os.path.join(filePrefix,
                                            'frame_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.mat'%
                                            (time.year,time.month,time.day,
                                             time.hour,time.minute,time.second))
                 fh=open(binFilename,'rb')
                 allDict = sio.loadmat(fh, squeeze_me=True)
                 fh.close()
                 
              
              else:
              
                 print(('No binary file found, recalculating '+
                        'frame_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ'%
                        (time.year,time.month,time.day,time.hour,time.minute,time.second)+
                        '...'))
                 raise Exception
              
              
              
              
              # ignore binary file if the coordinate system is not consistent with geoGrid
              if ((geoGrid and allDict['coordinates'] != 'Geographic') or
                  (not(geoGrid) and allDict['coordinates'] != 'Solar Magnetic')):
                 raise Exception
              
              phi_dict,theta_dict,rho_dict = allDict['dB_obs']            
              phi = phi_dict['data'] * 1. # '*1' forces array of floats, NOT objects
              theta = theta_dict['data'] * 1. # '*1' forces array of floats, NOT objects
              rho = rho_dict['data'] * 1. # '*1' forces array of floats, NOT objects
              
              if geoGrid:
                 phi_geo = phi
                 theta_geo = theta
                 rho_geo = rho
              
              dBphi_ion_dict,dBtheta_ion_dict,dBrho_ion_dict = allDict['dB_ion']
              dBphi_ion = dBphi_ion_dict['data'] / 1e9 # convert to Tesla
              dBtheta_ion = dBtheta_ion_dict['data'] / 1e9 # convert to Tesla
              dBrho_ion = dBrho_ion_dict['data'] / 1e9 # convert to Tesla
              
              dBphi_fac_dict,dBtheta_fac_dict,dBrho_fac_dict = allDict['dB_fac']
              dBphi_fac = dBphi_fac_dict['data'] / 1e9 # convert to Tesla
              dBtheta_fac = dBtheta_fac_dict['data'] / 1e9 # convert to Tesla
              dBrho_fac = dBrho_fac_dict['data'] / 1e9 # convert to Tesla
              
              dBphi_mag_dict,dBtheta_mag_dict,dBrho_mag_dict = allDict['dB_mag']
              dBphi_mag = dBphi_mag_dict['data'] / 1e9 # convert to Tesla
              dBtheta_mag = dBtheta_mag_dict['data'] / 1e9 # convert to Tesla
              dBrho_mag = dBrho_mag_dict['data'] / 1e9 # convert to Tesla
              
              
           except:
              
              
              # first read the MIX data
              vals=dMIX.read('Potential '+hemiSelect+' [V]',time)/1000.0
              psi_dict={'data':vals,'name':r'$\Phi$','units':r'kV'}
              vals=dMIX.read('Pedersen conductance '+hemiSelect+' [S]',time)
              sigmap_dict={'data':vals,'name':r'$\Sigma_{P}$','units':r'S'}
              vals=dMIX.read('Hall conductance '+hemiSelect+' [S]',time)
              sigmah_dict={'data':vals,'name':r'$\Sigma_{H}$','units':r'S'}
              vals=dMIX.read('FAC '+hemiSelect+' [A/m^2]',time)
              fac_dict={'data':vals*1e6,'name':r'$J_\parallel$','units':r'$\mu A/m^2$'}
              
              
              
              # then compute the electric field vectors
              ((phi_dict,theta_dict),
               (ephi_dict,etheta_dict)) = pyLTR.Physics.MIXCalcs.efieldDict(
                       xdict, ydict, psi_dict, ri=6500e3)
              
              
              # then compute total, Pedersen, and Hall current vectors
              if hemisphere=='north':
                 ((Jphi_dict,Jtheta_dict),
                  (Jpedphi_dict,Jpedtheta_dict),
                  (Jhallphi_dict,Jhalltheta_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                          (ephi_dict,etheta_dict), sigmap_dict, sigmah_dict, colatitude['data'])
              else:
                 ((Jphi_dict,Jtheta_dict),
                  (Jpedphi_dict,Jpedtheta_dict),
                  (Jhallphi_dict,Jhalltheta_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                          (ephi_dict,etheta_dict), sigmap_dict, sigmah_dict, n.pi-colatitude['data'])
               
              
              
              # then generate the SSECS, starting with min/max bounds of ionosphere
              # segments
              phi = phi_dict['data']
              theta = theta_dict['data']

              # caclulate MIX grid cell boundaries in phi and theta
              rion_min = [None] * 3 # initialize empty 3 list
              rion_min[0] = p.zeros(phi.shape)
              rion_min[0][1:,:] = phi[1:,:] - p.diff(phi, axis=0)/2.
              rion_min[0][0,:] = phi[0,:] - p.diff(phi[0:2,:], axis=0).squeeze()/2.

              rion_min[1] = p.zeros(theta.shape)
              rion_min[1][:,1:] = theta[:,1:] - p.diff(theta, axis=1)/2.
              rion_min[1][:,0] = theta[:,0] - p.diff(theta[:,0:2], axis=1).squeeze()/2.

              rion_min[2] = p.zeros(theta.shape)
              rion_min[2][:,:] = 6500.e3


              rion_max = [None] * 3 # initialize empty 3 list
              rion_max[0] = p.zeros(phi.shape)
              rion_max[0][:-1,:] = phi[:-1,:] + p.diff(phi, axis=0)/2.
              rion_max[0][-1,:] = phi[-1,:] + p.diff(phi[-2:,:], axis=0).squeeze()/2.

              rion_max[1] = p.zeros(theta.shape)
              rion_max[1][:,:-1] = theta[:,:-1] + p.diff(theta, axis=1)/2.
              rion_max[1][:,-1] = theta[:,-1] + p.diff(theta[:,-2:], axis=1).squeeze()/2.

              rion_max[2] = p.zeros(theta.shape)
              rion_max[2][:,:] = p.Inf
              
              
              
              # generate SSECS for ionospheric currents only
              (rv_ion, 
               Jv_ion, 
               dv_ion) = pyLTR.Physics.SSECS.ssecs_sphere([rion_min[0],rion_min[1],rion_min[2]], 
                                                          [rion_max[0],rion_max[1],rion_min[2]], 
                                                          (Jphi_dict['data']/1e6, 
                                                           Jtheta_dict['data']/1e6), 
                                                          10, False)
              
              # generate SSECS inside LFM inner boundary
              (rv_IBin, 
               Jv_IBin, 
               dv_IBin) = pyLTR.Physics.SSECS.ssecs_sphere([rion_min[0],rion_min[1],rion_min[2]], 
                                                           [rion_max[0],rion_max[1],2.5*rion_min[2]], 
                                                           (Jphi_dict['data']/1e6, 
                                                            Jtheta_dict['data']/1e6), 
                                                           10, False)
              
              # do NOT attempt to generate SSECS for FACs alone...this isn't possible
              # with existing code; However, the deltaB from FACs is the difference
              # between deltaBs calculated from the two SSECS above
              
              
              # generate SSECS outside LFM inner boundary
              # (this is serving as a proxy for magnetosphere currents for now)
              (rv_mag, 
               Jv_mag, 
               dv_mag) = pyLTR.Physics.SSECS.ssecs_sphere([rion_min[0],rion_min[1],2.5*rion_min[2]], 
                                                          [rion_max[0],rion_max[1],rion_min[2]],
                                                          (Jphi_dict['data']/1e6, 
                                                           Jtheta_dict['data']/1e6), 
                                                          10, False)
              
              # extract currents from MHD data
              # NOTE: LFM time stamps are not necessarily the same as MIX, so it
              #       is necessary to use the LFM's timeRange list (i.e., trLFM)
              x=dLFM.read('X_grid', trLFM[index0:index1][i]) # this is in cm
              y=dLFM.read('Y_grid', trLFM[index0:index1][i]) # this is in cm
              z=dLFM.read('Z_grid', trLFM[index0:index1][i]) # this is in cm
              Bx=dLFM.read('bx_', trLFM[index0:index1][i]) # this is in G
              By=dLFM.read('by_', trLFM[index0:index1][i]) # this is in G
              Bz=dLFM.read('bz_', trLFM[index0:index1][i]) # this is in G
              hgrid=pyLTR.Grids.HexahedralGrid(x,y,z)
              xB,yB,zB=hgrid.cellCenters()
              hgridcc=pyLTR.Grids.HexahedralGrid(xB,yB,zB) # B is at cell centers
              Jx,Jy,Jz = pyLTR.Physics.LFMCurrent(hgridcc,Bx,By,Bz,rion=1) # ...should be A/m^2 given default input units
              xJ,yJ,zJ = hgridcc.cellCenters() # ...and J is at the centers of these cells
              xJ = xJ/100 # ...and the coordinates should be in meters for BS.py
              yJ = yJ/100 # ...and the coordinates should be in meters for BS.py
              zJ = zJ/100 # ...and the coordinates should be in meters for BS.py
              ldV = hgridcc.cellVolume()/(100**3) # ...and we need dV in m^3 for BS.py
              
              if hemisphere=='south':
                 # it's easier to rotate the LFM grid than convert the MIX coordinates
                 # for southern hemisphere output
                 yJ = -yJ
                 zJ = -zJ
                 Jy = -Jy
                 Jz = -Jz
              
              
              
              #
              # This is a little ugly...Quad (and Oct) resolution LFM runs use
              # MIX grids that are different resoltions than Single and Double
              # runs; not surprising, but I was slow to figure this out. Anyway,
              # we need to visualize and cross-validate on similar grids, thus
              # the following kludge ('kludge' because the better answer is to
              # specify a useful grid without any reference to the MIX grid).
              #
              if phi.size == 181*27:
                 pass
              elif phi.size == 361*48:
                 phi = phi[::2,[0,1,2,3,4]+list(range(5,48,2))]
                 theta = theta[::2,[0,1,2,3,4]+list(range(5,48,2))]
              else:
                 raise Exception('Unrecognized MIX grid dimensions')
              
              
              
              # calculate deltaBs on a grid that decimates MIX grid by 2/3, and removes
              # the lowest 3 colatitutdes
              phi = phi[::3,3:]
              theta = theta[::3,3:]
              rho = p.tile(6378e3,phi.shape)
              
              if geoGrid:
                 phi_geo = phi
                 theta_geo = theta
                 rho_geo = rho
                 x,y,z = pyLTR.transform.SPHtoCAR(phi,theta,rho)
                 x,y,z = pyLTR.transform.GEOtoSM(x,y,z,time)
                 phi,theta,rho = pyLTR.transform.CARtoSPH(x,y,z)
              
              
              # deltaB for ionospheric currents
              (dBphi_ion, 
               dBtheta_ion, 
               dBrho_ion) = pyLTR.Physics.BS.bs_sphere(rv_ion, 
                                                       Jv_ion, 
                                                       dv_ion, 
                                                       (phi,theta,rho))
              
              
              # deltaB for currents inside IB
              (dBphi_IBin, 
               dBtheta_IBin, 
               dBrho_IBin) = pyLTR.Physics.BS.bs_sphere(rv_IBin, 
                                                        Jv_IBin, 
                                                        dv_IBin, 
                                                        (phi,theta,rho))
              # difference between dB*_IBin and dB*_ion is the FAC inside IB
              dBphi_fac = dBphi_IBin - dBphi_ion
              dBtheta_fac = dBtheta_IBin - dBtheta_ion
              dBrho_fac = dBrho_IBin - dBrho_ion
                            
              
              # deltaB from magnetospheric currents
              # convert cartesian positions and vectors to spherical
              lphi,ltheta,lrho,lJphi,lJtheta,lJrho=pyLTR.transform.CARtoSPH(xJ,yJ,zJ,Jx,Jy,Jz)
              (dBphi_mag, 
               dBtheta_mag, 
               dBrho_mag) = pyLTR.Physics.BS.bs_sphere((lphi,ltheta,lrho), 
                                                       (lJphi,lJtheta,lJrho), 
                                                       ldV, 
                                                       (phi,theta,rho))
              
              
              if geoGrid:
                 
                 # rotate ionospheric contribution from SM to GEO coordinates; leave
                 # position vectors unchanged for subsequent rotations
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(phi,theta,rho,dBphi_ion,dBtheta_ion,dBrho_ion)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphi_ion, dBtheta_ion, dBrho_ion = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
                 
                 # rotate FAC contribution from SM to GEO coordinates leave
                 # position vectors unchanged for subsequent rotations
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(phi,theta,rho,dBphi_fac,dBtheta_fac,dBrho_fac)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphi_fac, dBtheta_fac, dBrho_fac = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
                 
                 # rotate magnetospheric contribution from SM to GEO coordinates
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(phi,theta,rho,dBphi_mag,dBtheta_mag,dBrho_mag)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphi_mag, dBtheta_mag, dBrho_mag = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
                 
                 phi = phi_geo
                 theta = theta_geo
                 rho = rho_geo
              
              
              #
              # end of [re]processing 'except' block
              #
           
           
           
           
           # (re)create grid dictionary for subsequent plots and pickling
           toPickle={}
           
           if hemisphere=='south':
              toPickle['pov'] = 'south'
           else:
              toPickle['pov'] = 'north'
           
           if geoGrid:
              toPickle['coordinates'] = 'Geographic'
              
              # get geographic coordinates of sm pole
              x,y,z = pyLTR.transform.SPHtoCAR(0,0,1)
              x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
              poleCoords = pyLTR.transform.CARtoSPH(x,y,z)
           else:
              toPickle['coordinates'] = 'Solar Magnetic'
              
              ## # get sm coordinates of geographic pole
              ## x,y,z = pyLTR.transform.SPHtoCAR(0,0,1)
              ## x,y,z = pyLTR.transform.GEOtoSM(x,y,z,time)
              ## poleCoords = pyLTR.transform.CARtoSPH(x,y,z)
              
              ## now that MapPlot is being used, and until it can properly
              ## plot in centered SM coordinates, we want to just plot the 
              ## magnetic pole;
              poleCoords = (0, 0, 1)
           
           phi_dict = {'data':phi,'name':r'$\phi$','units':r'rad'}
           theta_dict = {'data':theta,'name':r'$\theta$','units':r'rad'}
           rho_dict = {'data':rho,'name':r'$\rho$','units':r'm'}
           
           toPickle['dB_obs'] = (phi_dict,theta_dict,rho_dict)
           
           
           
           
           #####################################################################
           ##
           ## FIXME: move plots from this function into if __name__ == '__main__'
           ##        section like deltaBTimeSeries.py; plots will then only be
           ##        generated if user requests it, and this function may be
           ##        called as part of a module...
           ##        But note: one reason the plots are generated inside this
           ##                  function is that a long time series of gridded
           ##                  data becomes unmanageable memory-wise fairly
           ##                  quickly...think this through carefully. -EJR
           ##
           #####################################################################
           
           
           
           # Now onto the plots
           tt=time.timetuple()
           
           p.figure(1,figsize=(28,6))
           p.figtext(0.5,0.92,'Ground '+'$\Delta{\mathbf{B}}$'+' - '+hemiSelect+
                     '\n%4d-%02d-%02d  %02d:%02d:%02d' %
                  (tt.tm_year,tt.tm_mon,tt.tm_mday,
                  tt.tm_hour,tt.tm_min,tt.tm_sec),
                  fontsize=14,multialignment='center')
           
           
           # plot total deltaB
           ax=p.subplot(141)
           # temporary dictionaries
           dBphi_dict = {'data':(dBphi_ion + dBphi_fac + dBphi_mag)*1e9,'name':r'$\Delta \mathrm{B}_{\phi}$','units':'nT'}
           dBtheta_dict = {'data':(dBtheta_ion + dBtheta_fac + dBtheta_mag)*1e9,'name':r'$\Delta \mathrm{B}_{\theta}$','units':'nT'}
           dBrho_dict = {'data':(dBrho_ion + dBrho_fac + dBrho_mag)*1e9,'name':r'$\Delta \mathrm{B}_{\rho}$','units':'nT'}
           bm = pyLTR.Graphics.MapPlot.QuiverPlotDict(phi_dict,theta_dict, 
                                                      dBrho_dict,(dBphi_dict, dBtheta_dict),
                                                      dtUTC=time, coordSystem=toPickle['coordinates'],
                                                      plotOpts1=dBradialTotOpts,
                                                      plotOpts2=dBhvecTotOpts,
                                                      points=[(poleCoords[0],poleCoords[1])],
                                                      userAxes=ax, northPOV=hemisphere=='north')
           
           to=p.text(.05, .95, r"$\Delta \mathbf{B}_{\mathrm{Total}}$", 
                     fontsize=14, transform=ax.transAxes)
          
           
           
           
           
           # plot ionospheric deltaB
           ax=p.subplot(142)
           # temporary dictionaries
           dBphi_dict = {'data':(dBphi_ion)*1e9,'name':r'$\Delta \mathrm{B}_{\phi}$','units':'nT'}
           dBtheta_dict = {'data':(dBtheta_ion)*1e9,'name':r'$\Delta \mathrm{B}_{\theta}$','units':'nT'}
           dBrho_dict = {'data':(dBrho_ion)*1e9,'name':r'$\Delta \mathrm{B}_{\rho}$','units':'nT'}
           bm = pyLTR.Graphics.MapPlot.QuiverPlotDict(phi_dict,theta_dict, 
                                                      dBrho_dict,(dBphi_dict, dBtheta_dict),
                                                      dtUTC=time, coordSystem=toPickle['coordinates'],
                                                      plotOpts1=dBradialIonOpts,
                                                      plotOpts2=dBhvecIonOpts,
                                                      points=[(poleCoords[0],poleCoords[1])],
                                                      userAxes=ax, northPOV=hemisphere=='north')
                      
           to=p.text(.05, .95, r"$\Delta \mathbf{B}_{\mathrm{ion}}$", 
                     fontsize=14, transform=ax.transAxes)
           
           # for subsequent pickling
           toPickle['dB_ion'] = (dBphi_dict,dBtheta_dict,dBrho_dict)
           
           
           # plot FAC deltaB
           ax=p.subplot(143)
           # temporary dictionaries
           dBphi_dict = {'data':(dBphi_fac)*1e9,'name':r'$\Delta \mathrm{B}_{\phi}$','units':'nT'}
           dBtheta_dict = {'data':(dBtheta_fac)*1e9,'name':r'$\Delta \mathrm{B}_{\theta}$','units':'nT'}
           dBrho_dict = {'data':(dBrho_fac)*1e9,'name':r'$\Delta \mathrm{B}_{\rho}$','units':'nT'}
           bm = pyLTR.Graphics.MapPlot.QuiverPlotDict(phi_dict,theta_dict, 
                                                      dBrho_dict,(dBphi_dict, dBtheta_dict),
                                                      dtUTC=time, coordSystem=toPickle['coordinates'],
                                                      plotOpts1=dBradialFACOpts,
                                                      plotOpts2=dBhvecFACOpts,
                                                      points=[(poleCoords[0],poleCoords[1])],
                                                      userAxes=ax, northPOV=hemisphere=='north')
           
           to=p.text(.05, .95, r"$\Delta \mathbf{B}_{\mathrm{fac}}$", 
                     fontsize=14, transform=ax.transAxes)
           
           # for subsequent pickling
           toPickle['dB_fac'] = (dBphi_dict,dBtheta_dict,dBrho_dict)
           
           
           # plot magnetospheric deltaB
           ax=p.subplot(144)
           # temporary dictionaries
           dBphi_dict = {'data':(dBphi_mag)*1e9,r'name':'$\Delta \mathrm{B}_{\phi}$','units':'nT'}
           dBtheta_dict = {'data':(dBtheta_mag)*1e9,r'name':'$\Delta \mathrm{B}_{\theta}$','units':'nT'}
           dBrho_dict = {'data':(dBrho_mag)*1e9,'name':r'$\Delta \mathrm{B}_{\rho}$','units':'nT'}
           bm = pyLTR.Graphics.MapPlot.QuiverPlotDict(phi_dict,theta_dict, 
                                                      dBrho_dict,(dBphi_dict, dBtheta_dict),
                                                      dtUTC=time, coordSystem=toPickle['coordinates'],
                                                      plotOpts1=dBradialMagOpts,
                                                      plotOpts2=dBhvecMagOpts,
                                                      points=[(poleCoords[0],poleCoords[1])],
                                                      userAxes=ax, northPOV=hemisphere=='north')
                      
           to=p.text(.05, .95, r"$\Delta \mathbf{B}_{\mathrm{mag}}$", 
                     fontsize=14, transform=ax.transAxes)
           
           # for subsequent pickling
           toPickle['dB_mag'] = (dBphi_dict,dBtheta_dict,dBrho_dict)
           
           
           
           
           #savefigName = os.path.join(path,'figs',hemisphere,'frame_deltaB_%05d.png'%i)
           filePrefix = os.path.join(path,'figs',hemisphere)
           pngFilename = os.path.join(filePrefix,'frame_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.png'%
                                                 (time.year,time.month,time.day,time.hour,time.minute,time.second))
           
           p.savefig(pngFilename,dpi=150)
           p.clf()
           
           
           if binaryType.lower() == 'pkl' or binaryType.lower() == '.pkl' or binaryType.lower() == 'pickle':
              # --- Dump a pickle!
              pklFilename = os.path.join(filePrefix,'frame_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.pkl'%
                                                    (time.year,time.month,time.day,time.hour,time.minute,time.second))
              fh = open(pklFilename, 'wb')
              pickle.dump(toPickle, fh, protocol=2)
              fh.close()
           elif binaryType.lower() == 'mat' or binaryType.lower() == '.mat' or binaryType.lower() == 'matlab':
              # --- Dump a .mat file!
              matFilename = os.path.join(filePrefix,'frame_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.mat'%
                                                    (time.year,time.month,time.day,time.hour,time.minute,time.second))
              sio.savemat(matFilename, toPickle)
           elif binaryType.lower() == 'none':
              pass
           else:
              print(('Unrecognized binary type '+binaryType+' requested'))
              raise Exception
              
              
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
    
        
        # append pngFilename to list of fully qualified filenames 
        pngFilenames.append(pngFilename)
    
        
    progress.stop()
    progress.join()

    
    #return  os.path.join(path,'figs',hemisphere)
    return pngFilenames

if __name__ == '__main__':

    (dirPath, run, t0, t1, northOnly, southOnly, geoGrid, ignoreBinary, binaryType, configFile, movieEncoder) = parseArgs()


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
        #frameDir=CreateFrames(dirPath, run, t0, t1,'north',configFile)
        frameList=CreateFrames(dirPath, run, t0, t1,'north',geoGrid,ignoreBinary,binaryType,configFile)
        if hasMovieConverter:
            #movieEncoder.encode( frameDir, 'frame_deltaB_', '%05d', 'png', 'deltaB_north.'+movieExtension )
            movieEncoder.encode_list(frameList, 'deltaB_north.'+movieExtension)

    if (not northOnly):
        #frameDir=CreateFrames(dirPath, run, t0, t1,'south',configFile)
        frameList=CreateFrames(dirPath, run, t0, t1,'south',geoGrid,ignoreBinary,binaryType,configFile)
        if hasMovieConverter:
            #movieEncoder.encode( frameDir, 'frame_deltaB_', '%05d', 'png', 'deltaB_south.'+movieExtension )
            movieEncoder.encode_list(frameList, 'deltaB_south.'+movieExtension)

       
