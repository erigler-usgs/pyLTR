#!/usr/bin/env python
"""
pyLTR deltaB time series generator
 Execute 'deltaBTimeSeries.py --help' for more info on command-line usage.
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
import copy as cp
import datetime
#from time import sleep
import math
import optparse
import os
import re
import sys

def parseArgs():
    """
    Returns time series data from deltaB output.
    Execute `mixTimeSeries.py --help` for more information.
    """
    # additional optparse help available at:
    # http://docs.python.org/library/optparse.html
    # http://docs.python.org/lib/optparse-generating-help.html
    parser = optparse.OptionParser(usage='usage: %prog -p [PATH] [options]',
                                   version=pyLTR.Release.version)

    parser.add_option('-p', '--path', dest='path',
                      default='./',
                      metavar='PATH', help='Path to base run directory containig MIX HDF files.')

    parser.add_option('-d', '--outDir', dest='outDir',
                      default='./',
                      metavar='PATH', help='Path, relative to path, to output data directory.')

    parser.add_option('-r', '--runName', dest='runName',
                      default='', metavar='RUN_IDENTIFIER', help='Optional name of run. Leave empty unless there is more than one run in a directory.')

    parser.add_option('-f', '--first', dest='t0',
                      default='', metavar='YYYY-MM-DD-HH-MM-SS', help='Date & Time that should be the first element of the time series')

    parser.add_option('-l', '--last', dest='t1',
                      default='', metavar='YYYY-MM-DD-HH-MM-SS', help='Date & Time of last element for the time series')

    parser.add_option('-o', '--observatories', dest='observ', default=[], action='append',
                      help='Observatory coordinates as comma-separted list (i.e., long,colat,radius);\n'+
                           'can also specify observatory label as 4th element of list, or accept default;\n'+
                           'this option can be repeated for multiple observatories, OR if its argument is\n'+
                           'not a comma-separated list, assume it is a filename to be read that contains\n'+
                           'multiple lines, each with a comma-separated list of coordinates.')

    parser.add_option("-i", "--ignoreBinary", dest="ignoreBinary", action="store_true", default=False,
                      help="Ignore existing binary files and recalculate all summary data")

    parser.add_option("-b", "--binaryType", dest="binaryType", action="store", default="pkl",
                      help="Set type of binary output file")

    parser.add_option("-g", "--geoGrid", dest="geoGrid", action="store_true", default=False,
                      help="Assume GEOgraphic coordinates instead of typical SM")


    parser.add_option("-m", "--multiPlot", dest="multiPlot", default='tot,ion,fac,mag',
                      help="Comma-separated list of strings specifying which deltaB sources to\n"+
                           "include in each panel of the multiplot summaries. Any, some, or all\n"+
                           "of tot,ion,fac,mag are acceptable, or None to forego plotting.\n"+
                           "(note: ts are plotted in order received; repeats are allowed")


    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    if options.about:
        print(sys.argv[0] + ' version ' + pyLTR.Release.version)
        print('')
        print('This script searches path for any LFM and MIX ionosphere output files,')
        print('generates magnetospheric, FAC, and ionospheric currents, then calculates')
        print('for each specified observatory:')
        print('  - H: horizontal (i.e., -dB_theta); ionosphere, FAC, magnetosphere, total')
        print('  - E: eastward (i.e., dB_phi); ionosphere, FAC, magnetosphere, total')
        print('  - Z: downward (i.e., -dB_rho); ionosphere, FAC, magnetosphere, total')
        print('')
        print('This script generates the following files, depending on inputs:')
        print('  1a. runPath/figs/obs_deltaB_yyyy-mm-ddTHH-MM-SSZ.pkl - PKL file')
        print('      holding particular time steps\' delta B values, decomposed')
        print('      into ionospheric, FAC, and magnetospheric contributions, in')
        print('      spherical coordinates (i.e., phi,theta,rho).')
        print('  1b. runPath/figs/obs_deltaB_yyyy-mm-ddTHH-MM-SSZ.mat - MAT file')
        print('      holding particular time steps\' delta B values.')
        print('      (These files are mostly to avoid unnecessary recomputation')
        print('       when all that is desired is to replot the data, and are)')
        print('       probably not very useful for subsequent analysis)')
        print('  2.  runPath/figs/dBTS_ObsName.png - PNG graphic, one for each')
        print('      observatory "ObsName", showing a multi-panel time series plot')
        print('      of delta B constituents for HEZ coordinate components')
        print('  3a. runPath/figs/dBTS_ObsName.pkl - PKL files, one for each')
        print('      observatory "ObsName", holding pyLTR time series objects;')
        print('      this is probably most useful for subsequent analysis that')
        print('      is not possible from the time series plots.')
        print('  3b. runPath/figs/dBTS_ObsName.mat - MAT files, one for each')
        print('      observatory "ObsName", holding pyLTR time series objects;')
        print('      this is probably most useful for subsequent analysis that')
        print('      is not possible from the time series plots.')
        print('')
        print('To limit the time range, use the "--first" and/or "--last" flags.')
        print('To specify observatories, use the "--observatories" flag 1 or more times.')
        print('To treat observatories as geographic coordinates, use the "--geoGrid" flag.')
        print('Execute "' + sys.argv[0] + ' --help" for details.')
        print('')
        sys.exit()

    # --- Sanitize inputs

    path = options.path
    assert( os.path.exists(path) )

    run = options.runName

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



    # check option.observs for single-element strings that *might* be filenames
    obsFiles = [obs for obs in options.observ if len(obs.split(','))==1]
    # everything else is placed in obsStrings for now (invalid strings will crash program)
    obsStrings = [obs for obs in options.observ if len(obs.split(','))>1]
    # append observatory strings from file(s)
    for obs in obsFiles:
       fh=open(obs,'rb')
       obsStrings+=[s for s in fh.read().splitlines()
                    if (not s.strip().startswith('#') and # ignore comment lines
                        not s.strip() == '' )] # ignore empty lines
       fh.close()

    # parse observatory coordinates and labels from obsStrings
    obsList = [list(map(float, obs.split(',')[:3])) for obs in obsStrings]
    obsLabels = [obs.split(',')[3].strip() if len(obs.split(',')) > 3
                 else 'Obs%04d'%i # gotta love Python list comprehension
                 for i,obs in enumerate(obsStrings,1)]
    obsList = [obsList[i]+[obsLabels[i],] for i in range(len(obsList))]




    geoGrid = options.geoGrid

    ignoreBinary = options.ignoreBinary

    binaryType = options.binaryType

    multiPlot = options.multiPlot.split(',')

    outDir = options.outDir

    return (path, run, t0, t1, obsList, geoGrid, ignoreBinary, binaryType, multiPlot,outDir)



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




def _dipoleMF(dBts, geoGrid=False):
   """
   Support routine that replaces deltaBs in dBts (pyLTR.TimeSeries object created
   by extractQuantities() below) with a dipole estimate of the magnetic main field.
   This function generates a time series that might be compared directly with
   observations.

   The geoGrid parameter specifies whether SM or GEO coordinates are used and
   calculated.
   """

   R0 = 6378000 # radius at Earth's equator in meters
   B0 = 3.12e-5 # magnetic field strength in Tesla, at equator, at R0

   # extract datetimes
   datetimes = dBts['datetime']['data']

   # copy dBts to Bts
   Bts = cp.deepcopy(dBts)

   for i,ts in enumerate(datetimes):

      # regardless of geoGrid, dipole main field must be calculated at
      # pre-calculated observatory SM coordinates
      phiSM = dBts['phiSM']['data'][i]
      thetaSM = dBts['thetaSM']['data'][i]
      rhoSM = dBts['rhoSM']['data'][i]
      Bphi = 0.0
      Btheta = -B0*p.sin(thetaSM)
      Brho = -2.0*B0*p.cos(thetaSM)


      # if geoGrid is true, transform B_SM into B_GEO
      if geoGrid:
         x,y,z,Bx,By,Bz = pyLTR.transform.SPHtoCAR(phiSM,thetaSM,rhoSM,Bphi,Btheta,Brho)
         x,y,z = pyLTR.transform.SMtoGEO(x,y,z,ts)
         Bx,By,Bz = pyLTR.transform.SMtoGEO(Bx,By,Bz,ts)
         phiGEO,thetaGEO,rhoGEO,Bphi,Btheta,Brho = pyLTR.transform.CARtoSPH(x,y,z,Bx,By,Bz)

         # check if *GEO match the GEO coordinates stored in dBts


      # replace deltaBs in Bts with Bs, in nanoTeslas
      Bts['North']['data'][i] = -Btheta * 1e9
      Bts['East']['data'][i] = Bphi * 1e9
      Bts['Down']['data'][i] = -Brho * 1e9



   # replace names
   Bts['North']['name'] = r'$B_{north}$'
   Bts['East']['name'] = r'$B_{east}$'
   Bts['Down']['name'] = r'$B_{down}$'



   return (Bts)



def extractQuantities(path='./', run='',
                      t0='', t1='',
                      obsList=None, geoGrid=False,
                      ignoreBinary=False, binaryType='pkl',
                      outDirName='figs'):
    """
    Compute deltaBs at virtual observatories given LFM-MIX output files in path.

    Computes:
      dBObs     - a list of pyLTR.TimeSeries objects, each corresponding to a
                  single virtual observatory defined in obsList
                  NOTE: while the outputs are pyLTR.TimeSeries objects, the
                        function generates binary pkl/mat files that combine
                        each specified observatory for each time step; this
                        helps reduce re-computation time for subsequent calls
                        to this function, and provides snapshots of output to
                        be read in by alternative analysis software

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
      obsList   - a list of lists of observatory coordinates [phi,theta,rho,ID]
                  where ID is an optional 4th element to each coordinate set
                  that, ideally, uniquely identifies the observatory...if ID is
                  not specified, it is assigned an empty string.
                  (default is coordinate system origin)
      geoGrid   - if True, assume observatory coordinates are in geographic
                  coordinates rather than solar magnetic; same for outputs
                  (default is False)
      ignoreBinary - if True, ignore any pre-computed binary files and re-
                  compute everything from scratch; NOTE: individual binary files
                  will be ignored anyway if they are incompatible with specified
                  inputs, but this option avoids reading the binary file entirely.
                  (default is False)
      binaryType   - binary type to generate, NOT to read in...routine looks for
                  PKL files first, then mat files, then proceeds to re-compute
                  if neither are available.
                  (default is 'pkl')
      outDirName - name of directory into which all output will be placed; must
                  be a relative path, to be appended to the input path; this is
                  also where binary pkl/mat files will be expected if/when the
                  function tries to read pre-computed data.
                  (default is 'figs')
    """

    # Make sure the output directory exisits if not make it
    dirname = os.path.join(path, outDirName)
    if not os.path.exists(dirname):
        os.makedirs( dirname )

    # process observatory coordinates
    if obsList==None or len(obsList) == 0:
       obs_phi = 0
       obs_theta = 0
       obs_rho = 0
       obs_label = 'Origin'
    elif p.all(p.greater_equal(list(map(len,obsList)), 3)):
       obs_phi = p.array([])
       obs_theta = p.array([])
       obs_rho = p.array([])
       obs_label = p.array([])
       for i in range(len(obsList)):
          obs_phi = p.append(obs_phi,obsList[i][0])
          obs_theta = p.append(obs_theta,obsList[i][1])
          obs_rho = p.append(obs_rho,obsList[i][2])
          obs_label = p.append(obs_label,obsList[i][3] if len(obsList[i])>3 else '')
    else:
       raise Exception(('Each observatory list must contain 3 floats (coordinates) plus a string (label)'))

    # sort observatory list by rho, then theta, then phi
    sortIdx = p.lexsort(p.vstack((obs_phi,obs_theta,obs_rho)))
    obs_phi = obs_phi[sortIdx]
    obs_theta = obs_theta[sortIdx]
    obs_rho = obs_rho[sortIdx]
    obs_label = obs_label[sortIdx]


    # create MIX data object
    dMIX = pyLTR.Models.MIX(path, run)

    # make sure necessary variables are defined in the model.
    modelVars = dMIX.getVarNames()
    for v in ['Grid X', 'Grid Y',
              'Potential North [V]', 'Potential South [V]',
              'FAC North [A/m^2]', 'FAC South [A/m^2]',
              'Pedersen conductance North [S]', 'Pedersen conductance South [S]',
              'Hall conductance North [S]', 'Hall conductance South [S]',
              'Average energy North [keV]', 'Average energy South [keV]',
              'Number flux North [1/cm^2 s]', 'Number flux South [1/cm^2 s]']:
        assert( v in modelVars )

    # create vector of MIX time steps
    timeRange = dMIX.getTimeRange()
    
    # basic sanity check(s) before proceeding
    if len(timeRange) == 0:
        raise Exception(('No MIX data files found.',
                         'Are you pointing to the correct run directory?'))

    # get indices to first and last desired time steps
    index0 = 0
    if t0:
        for i,t in enumerate(timeRange):
            if t0 >= t:
                index0 = i
    index1 = len(timeRange)
    if t1:
        for i,t in enumerate(timeRange):
            if t1 >= t:
                index1 = i + 1 
    print(( 'Extracting quantities for %d time steps.' % (index1-index0) ))





    # attempt to create corresponding TIEGCM data object; 
    # continue with warning if failure
    try:
        dTIEGCM = pyLTR.Models.TIEGCM(path, run)

        # make sure necessary variables are defined in the model
        modelVars = dTIEGCM.getVarNames()
        for v in ['KQLAM', 'KQPHI','mlat', 'mlon']:
            assert( v in modelVars )

        # create vector of TIEGCM time steps
        trTIEGCM = list(map(_roundTime, dTIEGCM.getTimeRange()))
        
        # check that there is a single corresponding TIEGCM time step for each
        # MIX time step (to a 1-second tolerance)
        for dt in timeRange[index0:index1]:
            assert(len([val for val in trTIEGCM if val == dt]) is 1)
        
    except:
        print("No or incompatible TIEGCM output found; continuing anyway...")
        dTIEGCM = None
    
    
    
    
    # attempt to create corresponding LFM data object;
    # continue with warning if failure
    try:
        dLFM = pyLTR.Models.LFM(path, run)

        # make sure necessary variables are defined in the model
        modelVars = dLFM.getVarNames()
        for v in ['X_grid', 'Y_grid', 'Z_grid',
                  'bx_', 'by_', 'bz_']:
            assert( v in modelVars )

        # create vector of LFM time steps, rounded to nearest second
        # (only use these to check for correspondence with MIX time steps)
        trLFM = list(map(_roundTime, dLFM.getTimeRange()))
           
        # check that there is a single corresponding LFM time step for each
        # MIX time step (to a 1-second tolerance)
        for dt in timeRange[index0:index1]:
            assert(len([val for val in trLFM if val == dt]) is 1)

    except:
        print("No or incompatible LFM output found; continuing anyway...")
        dLFM = None    
    

    # Output a status bar displaying how far along the computation is.
    try:
        rows, columns = os.popen('stty size', 'r').read().split()
    except ValueError:
        print('Likely not a run from terminal so no progress bar')
        useProgressBar = False
    else:
        useProgressBar = True
        progress = pyLTR.StatusBar(0, index1-index0)
        progress.start()


    # initialize intermediate and final output lists
    t_doy   = []
    obsGEO = [[],[],[]] # list of 3 lists
    obsSM = [[],[],[]] # list of 3 lists

    dBNorth_total = []
    dBNorth_iono = []
    dBNorth_fac = []
    dBNorth_mag = []

    dBEast_total = []
    dBEast_iono = []
    dBEast_fac = []
    dBEast_mag = []

    dBDown_total = []
    dBDown_iono = []
    dBDown_fac = []
    dBDown_mag = []


    # Pre-compute static sun-fixed coordinates for northern hemisphere
    xN = dMIX.read('Grid X', timeRange[index0])
    yN = dMIX.read('Grid Y', timeRange[index0])
    thetaN = p.arctan2(yN,xN)
    thetaN[thetaN<0] = thetaN[thetaN<0]+2*p.pi
    rN = p.sqrt(xN**2+yN**2)
    xN_dict = {'data':xN*6500e3,'name':'X','units':'m'}
    yN_dict = {'data':yN*6500e3,'name':'Y','units':'m'}
    longNdict = {'data':thetaN,'name':r'\phi','units':r'rad'}
    colatNdict = {'data':n.arcsin(rN),'name':r'\theta','units':r'rad'}

    # Pre-compute static sun-fixed coordinates for southern hemisphere
    xS = xN
    yS = -yN
    thetaS = p.arctan2(yS,xS)
    thetaS[thetaS<0] = thetaS[thetaS<0]+2*p.pi
    rS = p.sqrt(xS**2+yS**2)
    xS_dict = {'data':xS*6500e3,'name':'X','units':'m'}
    yS_dict = {'data':yS*6500e3,'name':'Y','units':'m'}
    longSdict = {'data':thetaS,'name':r'\phi','units':r'rad'}
    colatSdict = {'data':p.pi-n.arcsin(rS),'name':r'\theta','units':r'rad'}




    if dTIEGCM:
        # TIEGCM coordinates are magnetic latitude and longitude. They will need
        # to be tranformed into sun-fixed SM coordinates below, but at the least,
        # we can extract them once here. Convert to colatitude and polar angles,
        # from degrees into radians, and turn into meshgrids for later use
        theta_TIE_mag, phi_TIE_mag = p.meshgrid(
            (90 - dTIEGCM.read('mlat', timeRange[index0])) * p.pi/180.,
            dTIEGCM.read('mlon', timeRange[index0]) * p.pi/180.)


    
    
    if dLFM:
        # Pre-compute static sun-fixed coordinates for magnetosphere
        xM = dLFM.read('X_grid', timeRange[index0]) # this is in cm
        yM = dLFM.read('Y_grid', timeRange[index0]) # this is in cm
        zM = dLFM.read('Z_grid', timeRange[index0]) # this is in cm
        hgrid=pyLTR.Grids.HexahedralGrid(xM,yM,zM)
        xBM,yBM,zBM=hgrid.cellCenters()
        hgridcc=pyLTR.Grids.HexahedralGrid(xBM,yBM,zBM) # B is at cell centers
        xJM,yJM,zJM = hgridcc.cellCenters() # ...and J is at the centers of these cells
        xJM = xJM/100 # ...and the coordinates should be in meters for BS.py
        yJM = yJM/100 # ...and the coordinates should be in meters for BS.py
        zJM = zJM/100 # ...and the coordinates should be in meters for BS.py
        dVM = hgridcc.cellVolume()/(100**3) # ...and we need dV in m^3 for BS.py




    # Prepare to initiate main loop
    noFilePrinted=False
    t0 = timeRange[index0]
    tt = t0.timetuple() # required for serial DOY (i.e., doesn't wrap at new year)
    doy0 = tt.tm_yday + tt.tm_hour/24.0 + tt.tm_min/1440.0 + tt.tm_sec/86400.0
    for i,time in enumerate(timeRange[index0:index1]):

        # this try/except block is just to handle interrupts and to clean up
        # the progress bar if something goes wrong
        try:

           # -- Day of Year

           # assuming the following is supposed to provide a monotonic time sequence,
           # it breaks down at the new year...fixed it; consider fixing it in other
           # pyLTR time series functions
           #tt = time.timetuple()
           #t_doy.append(tt.tm_yday+tt.tm_hour/24.0+tt.tm_min/1440.0+tt.tm_sec/86400.0)
           t_doy.append(doy0 + (time-t0).total_seconds()/86400.)


           if geoGrid:
              # convert observatory inputs into SM coordinates for BS integration
              obs_phi_geo = obs_phi
              obs_theta_geo = obs_theta
              obs_rho_geo = obs_rho
              x,y,z = pyLTR.transform.SPHtoCAR(obs_phi,obs_theta,obs_rho)
              x,y,z = pyLTR.transform.GEOtoSM(x,y,z,time)
              obs_phi,obs_theta,obs_rho = pyLTR.transform.CARtoSPH(x,y,z)
           else:
              x,y,z = pyLTR.transform.SPHtoCAR(obs_phi,obs_theta,obs_rho)
              x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
              obs_phi_geo,obs_theta_geo,obs_rho_geo = pyLTR.transform.CARtoSPH(x,y,z)


           # this try/except block is designed to read pre-computed binary files
           # if they exist, otherwise everything gets recalculated, which can be
           # time-consuming.
           try:

              # ignore binary file even if one exists
              if ignoreBinary:
                 raise Exception


              # look for a .pkl file holding deltaB data for this time step
              # before recalculating all the derived data; if this fails, look
              # for a .mat file; if this fails, fall through to recalculate
              # summary data
              filePrefix = os.path.join(path,outDirName)

              # this is a possible race condition, but try/except doesn't do what I want
              if os.path.exists(os.path.join(filePrefix,
                                             'obs_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.pkl'%
                                             (time.year,time.month,time.day,
                                              time.hour,time.minute,time.second))):

                 binFilename = os.path.join(filePrefix,
                                            'obs_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.pkl'%
                                            (time.year,time.month,time.day,
                                             time.hour,time.minute,time.second))
                 fh=open(binFilename,'rb')
                 allDict = pickle.load(fh)
                 fh.close()

                 # extract object lists of dictionaries from Pickle file
                 dB_obs = allDict['dB_obs']
                 dB_ion = allDict['dB_ion']
                 dB_fac = allDict['dB_fac']
                 dB_mag = allDict['dB_mag']


              elif os.path.exists(os.path.join(filePrefix,
                                               'obs_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.mat'%
                                               (time.year,time.month,time.day,
                                                time.hour,time.minute,time.second))):

                 binFilename = os.path.join(filePrefix,
                                            'obs_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.mat'%
                                            (time.year,time.month,time.day,
                                             time.hour,time.minute,time.second))
                 fh=open(binFilename,'rb')
                 allDict = sio.loadmat(fh, squeeze_me=True)
                 fh.close()

                 # sio.loadmat() converts ML structures into NumPy "structured
                 # arrays", and ML cell arrays into NumPy "object arrays"; we
                 # prefer these as Python dictionaries and lists

                 # extract object arrays of records from sio.loadmat's output dict
                 dB_obs = allDict['dB_obs']
                 dB_ion = allDict['dB_ion']
                 dB_fac = allDict['dB_fac']
                 dB_mag = allDict['dB_mag']


                 # convert arrays of records into lists of dictionaries
                 # NOTE: this invokes a handy trick for extracting the 'object' from
                 #       a 0-d object array, which is what is iterated on by c below
                 dB_obs = [dict(list(zip(rec.dtype.names, rec[()]))) for rec in dB_obs]
                 dB_ion = [dict(list(zip(rec.dtype.names, rec[()]))) for rec in dB_ion]
                 dB_fac = [dict(list(zip(rec.dtype.names, rec[()]))) for rec in dB_fac]
                 dB_mag = [dict(list(zip(rec.dtype.names, rec[()]))) for rec in dB_mag]

              else:
                 print(('\rNo valid binary file found, recalculating '+
                        'obs_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ'%
                        (time.year,time.month,time.day,time.hour,time.minute,time.second)+
                        '...'))
                 raise Exception


              # ignore binary file if the coordinate system is not consistent with geoGrid
              if ((geoGrid and allDict['coordinates'] != 'Geographic') or
                  (not(geoGrid) and allDict['coordinates'] != 'Solar Magnetic')):
                 raise Exception

              # resort dB_* according to rho, then theta, then phi
              phi = dB_obs[0]['data']
              theta = dB_obs[1]['data']
              rho = dB_obs[2]['data']
              # note, axis=0 and kind='heapsort' are important here
              dBSortIdx = p.lexsort(p.vstack((phi,theta,rho)))
              for d in range(len(dB_obs)):
                 dB_obs[d]['data'] =  dB_obs[d]['data'][dBSortIdx]
                 dB_ion[d]['data'] =  dB_ion[d]['data'][dBSortIdx]
                 dB_fac[d]['data'] =  dB_fac[d]['data'][dBSortIdx]
                 dB_mag[d]['data'] =  dB_mag[d]['data'][dBSortIdx]

              # check that all obs coordinates match those requested
              #dB_obs = allDict['dB_obs'] # this is done above now
              if geoGrid:
                 if not (all(p.array(dB_obs[0]['data']) == p.array(obs_phi_geo)) and
                         all(p.array(dB_obs[1]['data']) == p.array(obs_theta_geo)) and
                         all(p.array(dB_obs[2]['data']) == p.array(obs_rho_geo)) ):
                    raise Exception
              else:
                 if not (all(p.array(dB_obs[0]['data']) == p.array(obs_phi)) and
                         all(p.array(dB_obs[1]['data']) == p.array(obs_theta)) and
                         all(p.array(dB_obs[2]['data']) == p.array(obs_rho)) ):
                    raise Exception

              # re-pack everything into a dictionary to be dumped to binary
              # (we do this because allDict returned from loadmat() is not
              #  *exactly* like the dictionary we want to dump; this would
              #  not be necessary if a Pickle file was loaded)
              toPickle = {}
              toPickle['pov'] = allDict['pov']
              toPickle['coordinates'] = allDict['coordinates']
              toPickle['pov'] = 'north'
              toPickle['dB_obs'] = dB_obs
              toPickle['dB_ion'] = dB_ion
              toPickle['dB_fac'] = dB_fac
              toPickle['dB_mag'] = dB_mag


           except:

              # read the northern hemisphere MIX data
              vals=dMIX.read('Potential '+'North'+' [V]',time)/1000.0
              psiN_dict={'data':vals,'name':r'$\Phi$','units':r'kV'}
              vals=dMIX.read('Pedersen conductance '+'North'+' [S]',time)
              sigmapN_dict={'data':vals,'name':r'$\Sigma_{P}$','units':r'S'}
              vals=dMIX.read('Hall conductance '+'North'+' [S]',time)
              sigmahN_dict={'data':vals,'name':r'$\Sigma_{H}$','units':r'S'}

              # read the southern hemisphere MIX data
              vals=dMIX.read('Potential '+'South'+' [V]',time)/1000.0
              psiS_dict={'data':vals,'name':r'$\Phi$','units':r'kV'}
              vals=dMIX.read('Pedersen conductance '+'South'+' [S]',time)
              sigmapS_dict={'data':vals,'name':r'$\Sigma_{P}$','units':r'S'}
              vals=dMIX.read('Hall conductance '+'South'+' [S]',time)
              sigmahS_dict={'data':vals,'name':r'$\Sigma_{H}$','units':r'S'}


              # compute the electric field vectors
              ((phiN_dict,thetaN_dict),
               (ephiN_dict,ethetaN_dict)) = pyLTR.Physics.MIXCalcs.efieldDict(
                       xN_dict, yN_dict, psiN_dict, ri=6500e3)

              ((phiS_dict,thetaS_dict),
               (ephiS_dict,ethetaS_dict)) = pyLTR.Physics.MIXCalcs.efieldDict(
                       xS_dict, yS_dict, psiS_dict, ri=6500e3, oh=True)


              # compute total, Pedersen, and Hall ionospheric current vectors
              ((JphiN_dict,JthetaN_dict),
               (JpedphiN_dict,JpedthetaN_dict),
               (JhallphiN_dict,JhallthetaN_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                          (ephiN_dict,ethetaN_dict), sigmapN_dict, sigmahN_dict, colatNdict['data'])

              ((JphiS_dict,JthetaS_dict),
               (JpedphiS_dict,JpedthetaS_dict),
               (JhallphiS_dict,JhallthetaS_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                          (ephiS_dict,ethetaS_dict), sigmapS_dict, sigmahS_dict, colatSdict['data'])
              
              
              
              
              # Calculate DALECS current segments
              if dMIX:
              
                  # start by generating min/max bounds of ionosphere segments

                  # Northern hemisphere
                  phiN = phiN_dict['data']
                  thetaN = thetaN_dict['data']
                  rionN_min, rionN_max = pyLTR.Physics.DALECS._edgeGrid((phiN,thetaN))
                  rionN_min.append(p.zeros(phiN.shape)+6500e3)
                  rionN_max.append(p.zeros(phiN.shape)+p.Inf)

                  # Southern hemisphere
                  phiS = phiS_dict['data']
                  thetaS = thetaS_dict['data']
                  rionS_min, rionS_max = pyLTR.Physics.DALECS._edgeGrid((phiS,thetaS))
                  rionS_min.append(p.zeros(phiS.shape)+6500e3)
                  rionS_max.append(p.zeros(phiS.shape)+p.Inf)


                  # next, generate DALECS for Northern ionospheric currents only
                  (rvN_ion,
                   JvN_ion,
                   dvN_ion) = pyLTR.Physics.DALECS_v2.dalecs_sphere([rionN_min[0],rionN_min[1],rionN_min[2]],
                                                               [rionN_max[0],rionN_max[1],rionN_min[2]],
                                                               (JphiN_dict['data']/1e6,
                                                                JthetaN_dict['data']/1e6),
                                                               10, False)

                  # next, generate DALECS for Northern hemisphere inside LFM inner boundary
                  (rvN_IBin,
                   JvN_IBin,
                   dvN_IBin) = pyLTR.Physics.DALECS_v2.dalecs_sphere([rionN_min[0],rionN_min[1],rionN_min[2]],
                                                               [rionN_max[0],rionN_max[1],2.5*rionN_min[2]],
                                                               (JphiN_dict['data']/1e6,
                                                                JthetaN_dict['data']/1e6),
                                                               10, False)

                  # do NOT attempt to generate DALECS for FACs alone...this isn't possible
                  # with existing code; However, the deltaB from FACs is the difference
                  # between deltaBs calculated from the two DALECS above

                  """
                  # generate DALECS for Northern Hemisphere outside LFM inner boundary
                  # (this is serving as a proxy for magnetosphere currents for now)
                  (rvN_mag,
                   JvN_mag,
                   dvN_mag) = pyLTR.Physics.DALECS.dalecs_sphere([rionN_min[0],rionN_min[1],2.5*rionN_min[2]],
                                                              [rionN_max[0],rionN_max[1],rionN_min[2]],
                                                              (JphiN_dict['data']/1e6,
                                                               JthetaN_dict['data']/1e6),
                                                              10, False)
                  """

                  # next, generate DALECS for Southern ionospheric currents only
                  (rvS_ion,
                   JvS_ion,
                   dvS_ion) = pyLTR.Physics.DALECS_v2.dalecs_sphere([rionS_min[0],rionS_min[1],rionS_min[2]],
                                                              [rionS_max[0],rionS_max[1],rionS_min[2]],
                                                              (JphiS_dict['data']/1e6,
                                                               JthetaS_dict['data']/1e6),
                                                              10, False)

                  # next, generate DALECS for Southern hemisphere inside LFM inner boundary
                  (rvS_IBin,
                   JvS_IBin,
                   dvS_IBin) = pyLTR.Physics.DALECS_v2.dalecs_sphere([rionS_min[0],rionS_min[1],rionS_min[2]],
                                                               [rionS_max[0],rionS_max[1],2.5*rionS_min[2]],
                                                               (JphiS_dict['data']/1e6,
                                                                JthetaS_dict['data']/1e6),
                                                               10, False)

                  # do NOT attempt to generate DALECS for FACs alone...this isn't possible
                  # with existing code; However, the deltaB from FACs is the difference
                  # between deltaBs calculated from the two DALECS above

                  """
                  # generate DALECS for Southern Hemisphere outside LFM inner boundary
                  # (this is serving as a proxy for magnetosphere currents for now)
                  (rvS_mag,
                   JvS_mag,
                   dvS_mag) = pyLTR.Physics.DALECS.dalecs_sphere([rionS_min[0],rionS_min[1],2.5*rionS_min[2]],
                                                              [rionS_max[0],rionS_max[1],rionS_min[2]],
                                                              (JphiS_dict['data']/1e6,
                                                               JthetaS_dict['data']/1e6),
                                                              10, False)
                  """

                  #
                  # Calculate deltaBs
                  #

                  # deltaB for ionospheric currents
                  (dBphiN_ion,
                   dBthetaN_ion,
                   dBrhoN_ion) = pyLTR.Physics.BS.bs_sphere(rvN_ion,
                                                           JvN_ion,
                                                           dvN_ion,
                                                           (obs_phi,obs_theta,obs_rho))
                  # deltaB for currents inside IB
                  (dBphiN_IBin,
                   dBthetaN_IBin,
                   dBrhoN_IBin) = pyLTR.Physics.BS.bs_sphere(rvN_IBin,
                                                            JvN_IBin,
                                                            dvN_IBin,
                                                            (obs_phi,obs_theta,obs_rho))
                  # difference between dB*_IBin and dB*_ion is the FAC inside IB
                  dBphiN_fac = dBphiN_IBin - dBphiN_ion
                  dBthetaN_fac = dBthetaN_IBin - dBthetaN_ion
                  dBrhoN_fac = dBrhoN_IBin - dBrhoN_ion


                  # deltaB for Southern ionospheric currents
                  (dBphiS_ion,
                   dBthetaS_ion,
                   dBrhoS_ion) = pyLTR.Physics.BS.bs_sphere(rvS_ion,
                                                           JvS_ion,
                                                           dvS_ion,
                                                           (obs_phi,obs_theta,obs_rho))
                  # deltaB for Southern DALECS currents inside IB
                  (dBphiS_IBin,
                   dBthetaS_IBin,
                   dBrhoS_IBin) = pyLTR.Physics.BS.bs_sphere(rvS_IBin,
                                                            JvS_IBin,
                                                            dvS_IBin,
                                                            (obs_phi,obs_theta,obs_rho))
                  # difference between dB*_IBin and dB*_ion is the FAC inside IB
                  dBphiS_fac = dBphiS_IBin - dBphiS_ion
                  dBthetaS_fac = dBthetaS_IBin - dBthetaS_ion
                  dBrhoS_fac = dBrhoS_IBin - dBrhoS_ion
              else:

                  # set dBs to zero if no MIX data is available
                  dBphiN_ion = p.zeros(p.array(obs_phi).shape)
                  dBthetaN_ion = p.zeros(p.array(obs_theta).shape)
                  dBrhoN_ion = p.zeros(p.array(obs_rho).shape)
                  dBphiN_fac = p.zeros(p.array(obs_phi).shape)
                  dBthetaN_fac = p.zeros(p.array(obs_theta).shape)
                  dBrhoN_fac = p.zeros(p.array(obs_rho).shape)

                  dBphiS_ion = p.zeros(p.array(obs_phi).shape)
                  dBthetaS_ion = p.zeros(p.array(obs_theta).shape)
                  dBrhoS_ion = p.zeros(p.array(obs_rho).shape)
                  dBphiS_fac = p.zeros(p.array(obs_phi).shape)
                  dBthetaS_fac = p.zeros(p.array(obs_theta).shape)
                  dBrhoS_fac = p.zeros(p.array(obs_rho).shape)



              if dTIEGCM:
              #if False:
                  # retrieve TIEGCM height-integrated currents:convert to SM 
                  # coordinates, set thetas that overlap with MIX to zero, 
                  # generate DALECS, calculate deltaB
                  x, y, z, dx, dy, dz = pyLTR.transform.SPHtoCAR(
                    phi_TIE_mag,
                    theta_TIE_mag,
                    6500e3,
                    dTIEGCM.read('KQPHI', time).T,
                    -dTIEGCM.read('KQLAM', time).T,
                    0.0 )
                                      
                  x, y, z = pyLTR.transform.MAGtoSM(x, y, z, time)
                  dx, dy, dz = pyLTR.transform.MAGtoSM(dx, dy, dz, time)
                  
                  (phi_TIE_sm, theta_TIE_sm, rho_TIE_sm,
                   Jphi_TIE_sm, Jtheta_TIE_sm, Jrho_TIE_sm) = pyLTR.transform.CARtoSPH(
                    x, y, z, dx, dy, dz)


                  # FIXME: we really need to detect minimum MIX thetas, and
                  #        maybe even blend MIX and TIEGCM currents smoothly 
                  #        across this boundary
                  print("Discarding ", 
                    (theta_TIE_sm < (60 * p.pi/180.)).sum() +
                    (theta_TIE_sm > (120 * p.pi/180.)).sum(),
                    " of ", theta_TIE_sm.size,
                    "TIEGCM points outside +/- 30 degrees lat.")
                  Jphi_TIE_sm[theta_TIE_sm < (60 * p.pi/180.)] = 0
                  Jphi_TIE_sm[theta_TIE_sm > (120 * p.pi/180.)] = 0
                  Jtheta_TIE_sm[theta_TIE_sm < (60 * p.pi/180.)] = 0
                  Jtheta_TIE_sm[theta_TIE_sm > (120 * p.pi/180.)] = 0
                  

                  rionTIE_min, rionTIE_max = pyLTR.Physics.DALECS._edgeGrid((phi_TIE_sm, theta_TIE_sm))
                  rionTIE_min.append(p.zeros(rho_TIE_sm.shape) + 6500e3)
                  rionTIE_max.append(p.zeros(rho_TIE_sm.shape) + p.Inf)
                  
                  # ionospheric currents only
                  (rvTIE_ion,
                   JvTIE_ion,
                   dvTIE_ion) = pyLTR.Physics.DALECS.dalecs_sphere(
                    [rionTIE_min[0], rionTIE_min[1], rionTIE_min[2]],
                    [rionTIE_max[0], rionTIE_max[1], rionTIE_min[2]],
                    (Jphi_TIE_sm, Jtheta_TIE_sm),
                    10, False)
                  
                  
                  # deltaB for ionospheric currents
                  (dBphiTIE_ion,
                   dBthetaTIE_ion,
                   dBrhoTIE_ion) = pyLTR.Physics.BS.bs_sphere(
                    rvTIE_ion,
                    JvTIE_ion,
                    dvTIE_ion,
                    (obs_phi, obs_theta, obs_rho))
                                      
              else:
                  
                  # set dBs to zero if no TIEGCM data is available
                  dBphiTIE_ion = p.zeros(p.array(obs_phi).shape)
                  dBthetaTIE_ion = p.zeros(p.array(obs_theta).shape)
                  dBrhoTIE_ion = p.zeros(p.array(obs_rho).shape)



              #if False:
              if dLFM:
                  
                  # finally, retrieve magnetospheric currents from LFM file,
                  # convert to spherical coordinates, and calculate deltaB
                  BxM = dLFM.read('bx_', trLFM[index0:index1][i]) # this is in G
                  ByM = dLFM.read('by_', trLFM[index0:index1][i]) # this is in G
                  BzM = dLFM.read('bz_', trLFM[index0:index1][i]) # this is in G
                  JxM, JyM, JzM = pyLTR.Physics.LFMCurrent(
                    hgridcc, BxM, ByM, BzM, rion=1) # ...should be A/m^2 given default input units

                  (phiM, thetaM, rhoM,
                   JphiM, JthetaM, JrhoM) = pyLTR.transform.CARtoSPH(
                    xJM, yJM, zJM, JxM, JyM, JzM)
                  
                  (dBphi_mag,
                   dBtheta_mag,
                   dBrho_mag) = pyLTR.Physics.BS.bs_sphere(
                    (phiM,thetaM,rhoM),
                    (JphiM,JthetaM,JrhoM),
                    dVM,
                    (obs_phi,obs_theta,obs_rho))
                    
              else:
                  
                  # set dBs to zero if no LFM data is available
                  dBphi_mag = p.zeros(p.array(obs_phi).shape)
                  dBtheta_mag = p.zeros(p.array(obs_theta).shape)
                  dBrho_mag = p.zeros(p.array(obs_rho).shape)
                  





              if geoGrid:

                 # rotate north ionospheric contribution from SM to GEO coordinates;
                 # leave position vectors unchanged for subsequent rotations
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho,
                                                           dBphiN_ion, dBthetaN_ion, dBrhoN_ion)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphiN_ion, dBthetaN_ion, dBrhoN_ion = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)

                 # rotate north FAC contribution from SM to GEO coordinates;
                 # leave position vectors unchanged for subsequent rotations
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho,
                                                           dBphiN_fac, dBthetaN_fac, dBrhoN_fac)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphiN_fac, dBthetaN_fac, dBrhoN_fac = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)


                 # rotate south ionospheric contribution from SM to GEO coordinates;
                 # leave position vectors unchanged for subsequent rotations
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho,
                                                           dBphiS_ion, dBthetaS_ion, dBrhoS_ion)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphiS_ion, dBthetaS_ion, dBrhoS_ion = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)

                 # rotate south FAC contribution from SM to GEO coordinates;
                 # leave position vectors unchanged for subsequent rotations
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho,
                                                           dBphiS_fac, dBthetaS_fac, dBrhoS_fac)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphiS_fac, dBthetaS_fac, dBrhoS_fac = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)


                     
                 # rotate TIEGCM ionospheric contribution from SM to GEO coordinates;
                 # leave position vectors unchanged for subsequent rotations
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho,
                                                           dBphiTIE_ion, dBthetaTIE_ion, dBrhoTIE_ion)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphiTIE_ion, dBthetaTIE_ion, dBrhoTIE_ion = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
                                        


                 # rotate magnetospheric contribution from SM to GEO coordinates; leave
                 # position vectors unchanged for subsequent rotations
                 x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho,
                                                           dBphi_mag, dBtheta_mag, dBrho_mag)
                 x,y,z = pyLTR.transform.SMtoGEO(x,y,z,time)
                 dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,time)
                 _, _, _, dBphi_mag, dBtheta_mag, dBrho_mag = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
                    

              
              # combine dBs into lists of dicts, then dump to binary file
              toPickle = {}

              toPickle['pov'] = 'north'

              if geoGrid:
                 toPickle['coordinates'] = 'Geographic'
                 toPickle['dB_obs'] = [{'data':obs_phi_geo,'name':r'$\phi$','units':r'rad'},
                                       {'data':obs_theta_geo,'name':r'$\theta$','units':r'rad'},
                                       {'data':obs_rho_geo,'name':r'$\rho$','units':r'm'}]
              else:
                 toPickle['coordinates'] = 'Solar Magnetic'
                 toPickle['dB_obs'] = [{'data':obs_phi,'name':r'$\phi$','units':r'rad'},
                                       {'data':obs_theta,'name':r'$\theta$','units':r'rad'},
                                       {'data':obs_rho,'name':r'$\rho$','units':r'm'}]

              dB_ion = [{'data': (dBphiN_ion + dBphiS_ion + dBphiTIE_ion)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\phi}$'},
                        {'data': (dBthetaN_ion + dBthetaS_ion + dBthetaTIE_ion)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\theta}$'},
                        {'data': (dBrhoN_ion + dBrhoS_ion + dBrhoTIE_ion)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\rho}$'}]
              dB_fac = [{'data': (dBphiN_fac + dBphiS_fac)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\phi}$'},
                        {'data': (dBthetaN_fac + dBthetaS_fac)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\theta}$'},
                        {'data': (dBrhoN_fac + dBrhoS_fac)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\rho}$'}]
              dB_mag = [{'data': (dBphi_mag)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\phi}$'},
                        {'data': (dBtheta_mag)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\theta}$'},
                        {'data': (dBrho_mag)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\rho}$'}]

              toPickle['dB_ion'] = dB_ion
              toPickle['dB_fac'] = dB_fac
              toPickle['dB_mag'] = dB_mag

              # end except:


           filePrefix = os.path.join(path,outDirName) # "figs" is for all output, not just figs
           if binaryType.lower() == 'pkl' or binaryType.lower() == '.pkl' or binaryType.lower() == 'pickle':
              # --- Dump a pickle!
              pklFilename = os.path.join(filePrefix,'obs_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.pkl'%
                                                    (time.year,time.month,time.day,time.hour,time.minute,time.second))
              fh = open(pklFilename, 'wb')
              pickle.dump(toPickle, fh, protocol=2)
              fh.close()
           elif binaryType.lower() == 'mat' or binaryType.lower() == '.mat' or binaryType.lower() == 'matlab':
              # --- Dump a .mat file!
              matFilename = os.path.join(filePrefix,'obs_deltaB_%04d-%02d-%02dT%02d-%02d-%02dZ.mat'%
                                                    (time.year,time.month,time.day,time.hour,time.minute,time.second))
              sio.savemat(matFilename, toPickle)
           elif binaryType.lower() == 'none':
              pass
           else:
              print('Unrecognized binary type '+binaryType+' requested')
              raise Exception


           #
           # append North,East,Down values to intermediate output lists
           #
           dBNorth_total.append(-dB_ion[1]['data'] + -dB_fac[1]['data'] + -dB_mag[1]['data'])
           dBNorth_iono.append(-dB_ion[1]['data'])
           dBNorth_fac.append(-dB_fac[1]['data'])
           dBNorth_mag.append(-dB_mag[1]['data'])

           dBEast_total.append(dB_ion[0]['data'] + dB_fac[0]['data'] + dB_mag[0]['data'])
           dBEast_iono.append(dB_ion[0]['data'])
           dBEast_fac.append(dB_fac[0]['data'])
           dBEast_mag.append(dB_mag[0]['data'])

           dBDown_total.append(-dB_ion[2]['data'] + -dB_fac[2]['data'] + -dB_mag[2]['data'])
           dBDown_iono.append(-dB_ion[2]['data'])
           dBDown_fac.append(-dB_fac[2]['data'])
           dBDown_mag.append(-dB_mag[2]['data'])


           #
           # append coordinates;
           # if geoGrid=True, obsGEO will be constant;
           # if geoGrid=False, obsSM will be constant;
           #
           obsGEO[0].append(obs_phi_geo)
           obsGEO[1].append(obs_theta_geo)
           obsGEO[2].append(obs_rho_geo)
           obsSM[0].append(obs_phi)
           obsSM[1].append(obs_theta)
           obsSM[2].append(obs_rho)


           # reset obs_phi to passed coordinates for next loop iteration
           if geoGrid:
              obs_phi = obs_phi_geo
              obs_theta = obs_theta_geo
              obs_rho = obs_rho_geo


           if useProgressBar:
              progress.increment()

        except KeyboardInterrupt:
           # Exit when the user hits CTRL+C.
           if useProgressBar:
              progress.stop()
              progress.join()
           print('Exiting.')
           import sys
           sys.exit(0)
        except:
           # Cleanup progress bar if something bad happened.
           if useProgressBar:
              progress.stop()
              progress.join()
           raise
    if useProgressBar:
       progress.stop()
       progress.join()


    #
    # generate pyLTR.TimeSeries objects for final output; convert units to nT
    #
    dBObs = [] # output is a list of dictionaries of time series objects
    for obs in range(len(obsList)):

       # initialize pyLTR TimeSeries objects
       dBTot = pyLTR.TimeSeries()

       # insert shared values between different TimeSeries objects
       dBTot.append('datetime', 'Date & Time', '', timeRange[index0:index1])
       dBTot.append('doy','Day of Year','days',t_doy)
       dBTot.append('obs',obs_label[obs],'','')
       dBTot.append('phiSM', r'$\phi_{SM}$', 'rad', [phi[obs] for phi in obsSM[0]])
       dBTot.append('phiGEO', r'$\phi_{GEO}$', 'rad', [phi[obs] for phi in obsGEO[0]])
       dBTot.append('thetaSM', r'$\theta_{SM}$', 'rad', [theta[obs] for theta in obsSM[1]])
       dBTot.append('thetaGEO', r'$\theta_{GEO}$', 'rad', [theta[obs] for theta in obsGEO[1]])
       dBTot.append('rhoSM', r'$\rho_{SM}$', 'rad', [rho[obs] for rho in obsSM[2]])
       dBTot.append('rhoGEO', r'$\rho_{GEO}$', 'rad', [rho[obs] for rho in obsGEO[2]])

       dBIon = cp.deepcopy(dBTot) # deepcopy necessary to retain class attributes/functions
       dBFAC = cp.deepcopy(dBTot) # deepcopy necessary to retain class attributes/functions
       dBMag = cp.deepcopy(dBTot) # deepcopy necessary to retain class attributes/functions

       # insert North,East,Down components, in nT, into each TimeSeries object
       dBTot.append('North',r'$\Delta B_{north}$','nT', [tstep[obs] for tstep in dBNorth_total])
       dBTot.append('East',r'$\Delta B_{east}$','nT', [tstep[obs] for tstep in dBEast_total])
       dBTot.append('Down',r'$\Delta B_{down}$','nT', [tstep[obs] for tstep in dBDown_total])

       dBIon.append('North',r'$\Delta B_{north}$','nT', [tstep[obs] for tstep in dBNorth_iono])
       dBIon.append('East',r'$\Delta B_{east}$','nT', [tstep[obs] for tstep in dBEast_iono])
       dBIon.append('Down',r'$\Delta B_{down}$','nT', [tstep[obs] for tstep in dBDown_iono])

       dBFAC.append('North',r'$\Delta B_{north}$','nT', [tstep[obs] for tstep in dBNorth_fac])
       dBFAC.append('East',r'$\Delta B_{east}$','nT', [tstep[obs] for tstep in dBEast_fac])
       dBFAC.append('Down',r'$\Delta B_{down}$','nT', [tstep[obs] for tstep in dBDown_fac])

       dBMag.append('North',r'$\Delta B_{north}$','nT', [tstep[obs] for tstep in dBNorth_mag])
       dBMag.append('East',r'$\Delta B_{east}$','nT', [tstep[obs] for tstep in dBEast_mag])
       dBMag.append('Down',r'$\Delta B_{down}$','nT', [tstep[obs] for tstep in dBDown_mag])

       # output is a list of dictionaries of time series objects
       dBObs.append({'dBTot':dBTot, 'dBIon':dBIon, 'dBFAC':dBFAC, 'dBMag':dBMag})


    # return list (one element per observatory) of dictionaries of time series
    return (dBObs)


if __name__ == '__main__':

    (path, run, t0, t1, obs, geoGrid, ignoreBinary, binaryType, multiPlot, outDir) = parseArgs()

    (dBObs) = extractQuantities(path, run, t0, t1, obs, geoGrid, ignoreBinary, binaryType, outDir)


    # convert multiPlot into proper list of indices
    mp_idx = []
    for i in range(len(multiPlot)):
       mp_idx.extend(p.find(p.array(['tot','ion','fac','mag'])==multiPlot[i].lower()).tolist() )


    #
    # --- Make plots of everything
    #

    # Make sure the output directory exisits if not make it
    dirname = os.path.join(path, outDir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


    # loop over observatories, plot 3 panels for HEZ, each with 4 lines for tot,ion,fac,mag
    for i in range(len(dBObs)):

       # convert dictionary into ordered list of TimeSeries objects
       tsList = []
       tsList.append(dBObs[i]['dBTot'])
       tsList.append(dBObs[i]['dBIon'])
       tsList.append(dBObs[i]['dBFAC'])
       tsList.append(dBObs[i]['dBMag'])

       print('Creating delta B time series summaries for observatory: %s'%tsList[0]['obs']['name'])

       if mp_idx:

          filename = os.path.join(dirname, 'dBTS_'+tsList[0]['obs']['name']+'.png')

          # plot 3 panels (i.e., North,East,Down), each with time series specified by multiPlot
          pyLTR.Graphics.TimeSeries.MultiPlotN([tsList[mp] for mp in mp_idx],
                                               'doy', ['North', 'East', 'Down'],
                                               [['k','r','g','b'][mp] for mp in mp_idx],
                                               [['tot','ion','fac','mag'][mp] for mp in mp_idx])
          # additional ticklabels
          ax=p.gca()
          xticks=ax.get_xticks().tolist()
          xlims=ax.get_xlim()
          ax.set_xticklabels(xticks)
          labels=ax.get_xticklabels()
          labels=[item.get_text() for item in labels]

          labels=[item+'\n'+
                  '%4.1f'%(tsList[0]['phiSM']['data'][j]*180./p.pi)+'\n'+
                  '%4.1f'%(tsList[0]['thetaSM']['data'][j]*180./p.pi)
                  for j,item in enumerate(labels)]
          labels[-2] = 'DoY' + '\n' + r'$\phi_{SM}$' + '\n' + r'$\theta_{SM}$'

          ax.set_xticks(xticks)
          ax.set_xticklabels(labels)
          ax.set_xlim(xlims) # setting xticks changes xlims for some reason
          fig=p.gcf()
          p.title(tsList[0]['obs']['name'])
          #fig.tight_layout()
          p.subplots_adjust(hspace=0)
          p.subplots_adjust(bottom=.16)
          p.savefig(filename)
          p.clf()


          # save observatory time series to a binary file
          # note: this is the full time series, not a single time-step like what
          #       is saved in extractQuantities() above...it is not used to avoid
          #       redoing computations in any way, but is intended to facilitate
          #       subsequent time-series analysis.
          if binaryType.lower() == 'pkl' or binaryType.lower() == '.pkl' or binaryType.lower() == 'pickle':
             # --- Dump a pickle!
             pklFilename = os.path.join(dirname, 'dBTS_'+tsList[0]['obs']['name']+'.pkl')
             try:
               fh = open(pklFilename, 'wb')
               pickle.dump(dBObs[i], fh, protocol=2)
               fh.close()
             except:
               print('Warning: Unable to write binary file: '+pklFilename)
          elif binaryType.lower() == 'mat' or binaryType.lower() == '.mat' or binaryType.lower() == 'matlab':
             # --- Dump a .mat file!
             matFilename = os.path.join(dirname, 'dBTS_'+tsList[0]['obs']['name']+'.mat')

             # savemat() cannot handle datetimes, so convert datetimes in output to
             # ML-compatible "days-since-epoch (31December0000)"
             dBObs[i]['dBTot']['datetime']['data'] = p.date2num(dBObs[i]['dBTot']['datetime']['data'])
             dBObs[i]['dBIon']['datetime']['data'] = p.date2num(dBObs[i]['dBIon']['datetime']['data'])
             dBObs[i]['dBFAC']['datetime']['data'] = p.date2num(dBObs[i]['dBFAC']['datetime']['data'])
             dBObs[i]['dBMag']['datetime']['data'] = p.date2num(dBObs[i]['dBMag']['datetime']['data'])

             try:
               sio.savemat(matFilename, dBObs[i])
             except:
               print('Warning: Unable to write binary file: '+matFilename)
          elif binaryType.lower() == 'none':
             pass
          else:
             print('Unrecognized binary type '+binaryType+' requested')
             raise Exception

    else:
       pass
