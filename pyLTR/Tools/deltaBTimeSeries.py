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
from multiprocessing import Pool
from psutil import cpu_count

# Standard
import pickle
import scipy.io as sio
import scipy.interpolate as sinterp
import copy as cp
import datetime
#from time import sleep
from time import perf_counter # only for rough performance profiling
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
    
    
    parser.add_option("--mix", dest="mix", action="store_true", default=False,
                      help="Attempt to process MIX data")
    parser.add_option("--tie", dest="tie", action="store_true", default=False,
                      help="Attempt to process TIEGCM data")
    parser.add_option("--lfm", dest="lfm", action="store_true", default=False,
                      help="Attempt to process LFM (MHD) data")

    parser.add_option("--mix_bs_mx", dest="mix_bs_mx", action="store_true", default=False,
                      help="Use matrix trick to speed up Biot-Savart for MIX data")
    parser.add_option("--tie_bs_mx", dest="tie_bs_mx", action="store_true", default=False,
                      help="Use matrix trick to speed up Biot-Savart for TIEGCM data")

    parser.add_option("--smGrid", dest="smGrid", action="store_true", default=False,
                      help="Assume solar magnetic coordinates")
    parser.add_option("-g", "--geoGrid", dest="geoGrid", action="store_true", default=False,
                      help="Assume GEOgraphic coordinates")
    parser.add_option("--magGrid", dest="magGrid", action="store_true", default=False,
                      help="Assume geoMAGnetic coordinates")


    parser.add_option("-i", "--ignoreBinary", dest="ignoreBinary", action="store_true", default=False,
                      help="Ignore existing binary files and recalculate all summary data")

    parser.add_option("-b", "--binaryType", dest="binaryType", action="store", default="pkl",
                      help="Set type of binary output file")


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


    mix = options.mix
    tie = options.tie
    lfm = options.lfm
    mix_bs_mx = options.mix_bs_mx
    tie_bs_mx = options.tie_bs_mx
    smGrid = options.smGrid
    geoGrid = options.geoGrid
    magGrid = options.magGrid

    ignoreBinary = options.ignoreBinary

    binaryType = options.binaryType

    multiPlot = options.multiPlot.split(',')

    outDir = options.outDir

    return (path, run, 
            t0, t1, 
            obsList,
            mix, tie, lfm,
            mix_bs_mx, tie_bs_mx,
            smGrid, geoGrid, magGrid, 
            ignoreBinary, binaryType, 
            multiPlot,
            outDir)



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


"""
Initial attempts to parallelize over observatories using  the Pool
class were ridiculously slow. Apparently everything gets serialized,
copied to worker processes, de-serialized, processed, re-serialized,
copied back, and de-serialized again. Since our input objects can be
very large, this slows down parallel execution immensely. This is a
well known problem when working with large data objects, and a more
complete explation can be found here:

https://thelaziestprogrammer.com/python/a-multiprocessing-pool-pickle

There are some special tools available in the multiprocessing module 
that allow certain data types to be shared between the main and worker 
processes, but they are not really able to handle a complicated object 
like our DALECS class. The only remaining option is to be creative 
with global variables passed to the worker processes. That is what the
simple methods below are for.
"""
def initializer(dalecs):
   # copy dalecs to worker's global namespace
   global dalecs_worker
   dalecs_worker = dalecs
def scale_parallel(Jion):
   # scale worker DALECS by Jion
   dalecs_worker.scale(Jion)
def bs_parallel(obs_xyz):
   # integrate Biot-Savart with worker DALECS
   results = dalecs_worker.bs_cart(obs_xyz)
   return results


def extractQuantities(path='./', run='',
                      t0='', t1='',
                      obsList=None,
                      mix=True, tie=True, lfm=True,
                      mix_bs_mx=False, tie_bs_mx=False,
                      smGrid=False, geoGrid=False, magGrid=False,
                      ignoreBinary=False, binaryType='pkl',
                      outDirName='figs',
                      nprocs=1):
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
      mix       - if True, attempt to read MIX data (default is True)
      tie       - if True, attempt to read TIEGCM data (default is True)
                  NOTE: this requires the TIEGCM module to have been configured
                        to output secondary history file(s) that contain at a 
                        minimum the mlat, mlon, KQLAM, adn KQPHI parameters
      lfm       - if True, attempt to read LFM data (default is True)
      mix_bs_mx - if True, pre-compute matrix to speed up Biot-Savart ~1000x after
                  first obs. (first obs. takes ~100 times longer; default=False)
      tie_bs_mx - if True, pre-compute matrix to speed up Biot-Savart ~1000x after
                  first obs. (first obs. takes ~100 times longer; default=False)
      smGrid    - if True, assume observatory coordinates are in solar magnetic
                  coordinates; same for outputs
                  (default is False unless no *Grid are set, then it is True)
      geoGrid   - if True, assume observatory coordinates are in geographic
                  coordinates; same for outputs
                  (default is False)
      magGrid   - if True, assume observatory coordinates are in geomagnetic
                  coordinates; same for outputs
                  (default is False)
                  NOTE: only one of smGrid, geoGrid, or magGrid may be true
      ignoreBinary - if True, ignore any pre-computed binary files and re-
                  compute everything from scratch
                  (default is False)
                  NOTE: individual binary files will be ignored anyway if they 
                        are incompatible with specified inputs, but this option 
                        avoids reading the binary file entirely
      binaryType - binary type to generate, NOT to read in...routine looks for
                  PKL files first, then mat files, then proceeds to re-compute
                  if neither are available.
                  (default is 'pkl')
      outDirName - name of directory into which all output will be placed; must
                  be a relative path, to be appended to the input path; this is
                  also where binary pkl/mat files will be expected if/when the
                  function tries to read pre-computed data.
                  (default is 'figs')
      nprocs    - specify the number of parallel processes over which to 
                  distribute observatory processing using multiprocessing.Pool
                  NOTE: if this is not 1, the DALECS matrix optimizations are 
                        not possible; this is probably a good thing, since
                        these optimizations can be huge memory hogs.
                  NOTE: there is overhead associated with parallelizing over
                        observatories that makes it not especially effective
                        when the number of observatories is not very large.
                        Setting nprocs > 1 should only be done when a large
                        number of observatories might to cause memory issues
                        when parallelizing over time steps (not unlikely when
                        the matrix optimzations are enabled).
                  (default is 1)
    """

    # this might be better handled as a keyword argument
    ion_rho = 6500e3

    # ensure that one and only one of smGrid, geoGrid, or magGrid are True
    if p.sum([smGrid, geoGrid, magGrid]) == 0:
       # default to smGrid
       smGrid = True
    if p.sum([smGrid, geoGrid, magGrid]) != 1:
       raise Exception('Only one of {smGrid, geoGrid, magGrid} may be True')

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


    #
    # Read in MIX, TIEGCM, and LFM data, if they exist, according to user-
    # specified `path` and `run` inputs. Find common time steps in MIX, TIE
    # and LFM data.
    #

    # create a MIX data object
    try:
       if mix is False:
          # if user specified mix=False, don't attempt to read
          raise Exception

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
       
       # create vector of MIX time steps, rounded to nearest minute
       trMIX = list(map(_roundTime, dMIX.getTimeRange()))

       #  # create vector of MIX time steps
       #  timeRange = dMIX.getTimeRange()
    except:
       print("MIX files missing or ignored; continuing anyway...")
       dMIX = None
       trMIX = []
    

    # create a TIE(GCM) data object
    try:
       if tie is False:
          # if user specified tiegcm=False, don't attempt to read
          raise Exception
       
       dTIE = pyLTR.Models.TIEGCM(path, run)

       # make sure necessary variables are defined in the model
       modelVars = dTIE.getVarNames()
       for v in ['KQLAM', 'KQPHI','mlat', 'mlon']:
           assert( v in modelVars )

       # create vector of TIEGCM time steps, rounded to nearest minute
       trTIE = list(map(_roundTime, dTIE.getTimeRange()))
               
    except:
       print("TIEGCM files missing or ignored; continuing anyway...")
       dTIE = None
       trTIE = []
    
    
    # create an LFM data object
    try:
       if lfm is False:
          # if user specified lfm=False, don't attempt to read
          raise Exception
       
       dLFM = pyLTR.Models.LFM(path, run)

       # make sure necessary variables are defined in the model
       modelVars = dLFM.getVarNames()
       for v in ['X_grid', 'Y_grid', 'Z_grid',
                 'bx_', 'by_', 'bz_']:
           assert( v in modelVars )

       # create vector of LFM time steps, rounded to nearest minute
       trLFM = list(map(_roundTime, dLFM.getTimeRange()))
       
    except:
       print("LFM files missing or ignored; continuing anyway...")
       dLFM = None
       trLFM = []


    # determine common time steps between MIX, TIE, and LFM data
    if not (len(trMIX) is 0 or len(trTIE) is 0):
       trMIXTIE = list(p.intersect1d(trMIX, trTIE))
    elif not len(trMIX) is 0:
       trMIXTIE = trMIX
    elif not len(trTIE) is 0:
       trMIXTIE = trTIE
    else:
       trMIXTIE = []

    if not (len(trMIXTIE) is 0 or len(trLFM) is 0):
       timeRange = list(p.intersect1d(trMIXTIE, trLFM))
    elif not len(trMIXTIE) is 0:
       timeRange = trMIXTIE
    elif not len(trLFM) is 0:
       timeRange = trLFM
    else:
       timeRange = []
  

    # basic sanity check before proceeding
    if len(timeRange) == 0:
        raise Exception('No compatible MIX, TIEGCM, and LFM data files found.')

    
    # get indices to first and last desired time steps    
    tmin = min(timeRange)
    tmax = max(timeRange)
    if t0: 
       if t0 >= tmin and t0 <= tmax:
          index0 = p.argmax([(t >= t0) for t in timeRange])
       elif t0 < tmin:
          index0 = 0
       else:
          index0 = len(timeRange) # t0 > tmax
    else:
       index0 = timeRange.index(tmin)

    if t1:
       if t1 >= tmin and t1 <= tmax:
          index1 = timeRange.index(tmax) - p.argmax([(t <= t1) for t in timeRange[::-1]]) + 1
       elif t1 > tmax:
          index1 = len(timeRange)
       else:
          index1 = 0 # t1 < tmin
    else:
       index1 = timeRange.index(tmax)


    if (index1 - index0) == 0:
      raise Exception('Requested time range not found in data files')
    else:
      print('Extracting quantities for %d time steps.' %(index1 - index0) )

   
   # Output a status bar displaying how far along the computation is.
    try:
        rows, columns = os.popen('stty size', 'r').read().split()
    except ValueError:
        print('Likely not a run from terminal so no progress bar')
        useProgressBar = False
    else:
        useProgressBar = False
      #   useProgressBar = True
      #   progress = pyLTR.StatusBar(0, index1-index0)
      #   progress.start()


    # initialize intermediate and final output lists
    t_doy   = []
    obsGEO = [[],[],[]] # list of 3 lists
    obsSM = [[],[],[]] # list of 3 lists
    obsMAG = [[],[],[]] # list of 3 lists

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


    DALECS = pyLTR.Physics.DALECS
    
    if dMIX:
      # MIX grid

      # Northern hemisphere
      xN = dMIX.read('Grid X', timeRange[index0])[:-1,:] # remove periodic longitude
      yN = dMIX.read('Grid Y', timeRange[index0])[:-1,:] # remove periodic longitude
      thetaN = p.arctan2(yN,xN)
      thetaN[:,0] = thetaN[:,1] # force valid thetas at the pole
      thetaN[thetaN<0] = thetaN[thetaN<0]+2*p.pi
      rN = p.sqrt(xN**2+yN**2)
      xN_dict = {'data':xN*ion_rho,'name':'X','units':'m'}
      yN_dict = {'data':yN*ion_rho,'name':'Y','units':'m'}
      longNdict = {'data':thetaN,'name':r'\phi','units':r'rad'}
      colatNdict = {'data':n.arcsin(rN),'name':r'\theta','units':r'rad'}
      
      # this will get initialized the first time they are needed in the loop
      # (FIXME: consider making these an input parameter)
      dalecs_N_ion = None
      dalecs_N_fac = None
      
      
      # Southern hemisphere
      xS = xN
      yS = -yN
      thetaS = p.arctan2(yS,xS)
      thetaS[:,0] = thetaS[:,1] # force valid thetas at the pole
      thetaS[thetaS<0] = thetaS[thetaS<0]+2*p.pi
      rS = p.sqrt(xS**2+yS**2)
      xS_dict = {'data':xS*ion_rho,'name':'X','units':'m'}
      yS_dict = {'data':yS*ion_rho,'name':'Y','units':'m'}
      longSdict = {'data':thetaS,'name':r'\phi','units':r'rad'}
      colatSdict = {'data':p.pi-n.arcsin(rS),'name':r'\theta','units':r'rad'}

      # this will get initialized the first time they are needed in the loop
      # (FIXME: consider making these an input parameter)
      dalecs_S_ion = None
      dalecs_S_fac = None
    
      
      # preliminary MIX weights; will be modified if there are TIEGCM data
      mixN_weights = p.ones(xN.shape)
      mixS_weights = p.ones(xN.shape)
      
    
    if dTIE:
      # TIEGCM grid

      if dMIX:
         # recalculate dMIX weights if there are TIEGCM data
         mixN_theta_max = colatNdict['data'].max()
         mixS_theta_min = colatSdict['data'].min()
      
         mixN_weights = p.ones(colatNdict['data'].shape)
         for i,theta in enumerate(colatNdict['data'].flat):
            mixN_weights.flat[i] = (1. if theta < (mixN_theta_max / 2) else
                                    mixN_theta_max / theta - 1)
         
         mixS_weights = p.ones(colatSdict['data'].shape)
         for i,theta in enumerate(colatSdict['data'].flat):
            mixS_weights.flat[i] = (1. if (p.pi - theta) < ((p.pi - mixS_theta_min) / 2) else
                                    (p.pi - mixS_theta_min) / (p.pi - theta) - 1)
      else:
         # for calculating TIEGCM weights if there are no MIX data
         # ...no MIX weights will ever be required inside the loop
         mixN_theta_max = 0
         mixS_theta_min = p.pi

      # create meshgrid in TIEGCM coordinates; trim periodic longitude
      TIE_theta = (90 - dTIE.read('mlat', timeRange[index0])) * p.pi/180.
      TIE_phi = p.mod(dTIE.read('mlon', timeRange[index0]) + 360, 360)[:-1] * p.pi/180.
      thetaT_mag_interp, phiT_mag_interp = p.meshgrid(TIE_theta, TIE_phi)

      TIE_weights = p.ones(thetaT_mag_interp.shape)
      for i,theta in enumerate(thetaT_mag_interp.flat):
         TIE_weights.flat[i] = (
            1 if (theta > mixN_theta_max and 
                  theta < mixS_theta_min) 
              else (2 * theta / mixN_theta_max - 1) 
                 if (theta <= mixN_theta_max and 
                     theta > mixN_theta_max / 2)
                 else (2 * (p.pi - theta) / (p.pi - mixS_theta_min) - 1)
                    if ((p.pi - theta) <= (p.pi - mixS_theta_min) and
                        (p.pi - theta) > (p.pi - mixS_theta_min) / 2 )
                    else 0
         )

      # this will get initialized the first time they are needed in the loop
      # (FIXME: consider making these an input parameter)
      dalecs_T_ion = None
      dalecs_T_fac = None


      TIE_weights[:,0] = 0 # force zero at south pole
      TIE_weights[:,-1] = 0 # force zero at north pole
      
        

    if dLFM:
      # LFM MHD grid

      # Magnetosphere
      x_sm = dLFM.read('X_grid', timeRange[index0]) # this is in cm
      y_sm = dLFM.read('Y_grid', timeRange[index0]) # this is in cm
      z_sm = dLFM.read('Z_grid', timeRange[index0]) # this is in cm
      hgrid = pyLTR.Grids.HexahedralGrid(x_sm, y_sm, z_sm)
      xB_sm, yB_sm, zB_sm = hgrid.cellCenters()
      hgridcc = pyLTR.Grids.HexahedralGrid(xB_sm, yB_sm, zB_sm) # B is at cell centers
      xJ_sm, yJ_sm, zJ_sm = hgridcc.cellCenters() # ...and J is at the centers of these cells
      xJ_sm = xJ_sm/100 # ...and the coordinates should be in meters for BS.py
      yJ_sm = yJ_sm/100 # ...and the coordinates should be in meters for BS.py
      zJ_sm = zJ_sm/100 # ...and the coordinates should be in meters for BS.py
      dV_sm = hgridcc.cellVolume()/(100**3) # ...and we need dV in m^3 for BS.py
      


    # Prepare to initiate main loop
    print()
    print("Starting main loop")
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


         
           # all calculations and results will be in the user-specified coordinates,
           # but we return all possible location coordinates, mostly to facilitate
           # subsequent visualization
           if geoGrid:
              
              # get GEO Cartesian and spherical coordinates
              obs_x, obs_y, obs_z = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho)
              obs_x_geo = obs_x
              obs_y_geo = obs_y
              obs_z_geo = obs_z
              obs_phi_geo = obs_phi
              obs_theta_geo = obs_theta
              obs_rho_geo = obs_rho
              
              # get MAG Cartesian and spherical coordinates
              obs_x_mag, obs_y_mag, obs_z_mag = pyLTR.transform.GEOtoMAG(
                 obs_x_geo, obs_y_geo, obs_z_geo, time
              )
              obs_phi_mag, obs_theta_mag, obs_rho_mag = pyLTR.transform.CARtoSPH(
                 obs_x_mag, obs_y_mag, obs_z_mag
              )

              # get SM Cartesian and spherical coordinates
              obs_x_sm, obs_y_sm, obs_z_sm = pyLTR.transform.GEOtoSM(
                 obs_x_geo, obs_y_geo, obs_z_geo, time
              )
              obs_phi_sm, obs_theta_sm, obs_rho_sm = pyLTR.transform.CARtoSPH(
                 obs_x_sm, obs_y_sm, obs_z_sm
              )

           elif magGrid:
              
              # get MAG Cartesian and spherical coordinates
              obs_x, obs_y, obs_z = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho)
              obs_x_mag = obs_x
              obs_y_mag = obs_y
              obs_z_mag = obs_z
              obs_phi_mag = obs_phi
              obs_theta_mag = obs_theta
              obs_rho_mag = obs_rho
              
              # get GEO Cartesian and spherical coordinates
              obs_x_geo, obs_y_geo, obs_z_geo = pyLTR.transform.MAGtoGEO(
                 obs_x_mag, obs_y_mag, obs_z_mag, time
              )
              obs_phi_geo, obs_theta_geo, obs_rho_geo = pyLTR.transform.CARtoSPH(
                 obs_x_geo, obs_y_geo, obs_z_geo
              )

              # get SM Cartesian and spherical coordinates
              obs_x_sm, obs_y_sm, obs_z_sm = pyLTR.transform.MAGtoSM(
                 obs_x_mag, obs_y_mag, obs_z_mag, time
              )
              obs_phi_sm, obs_theta_sm, obs_rho_sm = pyLTR.transform.CARtoSPH(
                 obs_x_sm, obs_y_sm, obs_z_sm
              )

           else:

              # get SM Cartesian and spherical coordinates
              obs_x, obs_y, obs_z = pyLTR.transform.SPHtoCAR(obs_phi, obs_theta, obs_rho)
              obs_x_sm = obs_x
              obs_y_sm = obs_y
              obs_z_sm = obs_z
              obs_phi_sm = obs_phi
              obs_theta_sm = obs_theta
              obs_rho_sm = obs_rho
              
              # get GEO Cartesian and spherical coordinates
              obs_x_geo, obs_y_geo, obs_z_geo = pyLTR.transform.SMtoGEO(
                 obs_x_sm, obs_y_sm, obs_z_sm, time
              )
              obs_phi_geo, obs_theta_geo, obs_rho_geo = pyLTR.transform.CARtoSPH(
                 obs_x_geo, obs_y_geo, obs_z_geo
              )

              # get MAG Cartesian and spherical coordinates
              obs_x_mag, obs_y_mag, obs_z_mag = pyLTR.transform.SMtoMAG(
                 obs_x_sm, obs_y_sm, obs_z_sm, time
              )
              obs_phi_mag, obs_theta_mag, obs_rho_mag = pyLTR.transform.CARtoSPH(
                 obs_x_mag, obs_y_mag, obs_z_mag
              )

           
           
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
              if ((smGrid and allDict['coordinates'] != 'Solar Magnetic') or
                  (geoGrid and allDict['coordinates'] != 'Geographic') or
                  (magGrid and allDict['coordinates'] != 'Geomagnetic')):
                 raise Exception

              # check that all obs coordinates match those requested
              # must re-sort dB_* first
              phi = dB_obs[0]['data']
              theta = dB_obs[1]['data']
              rho = dB_obs[2]['data']
              dBSortIdx = p.lexsort(p.vstack((phi,theta,rho)))
              for d in range(len(dB_obs)):
                 dB_obs[d]['data'] =  dB_obs[d]['data'][dBSortIdx]
                 dB_ion[d]['data'] =  dB_ion[d]['data'][dBSortIdx]
                 dB_fac[d]['data'] =  dB_fac[d]['data'][dBSortIdx]
                 dB_mag[d]['data'] =  dB_mag[d]['data'][dBSortIdx]

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

              if dMIX:
                  print()
                  print("MIX")

                  # Calculate MIX ionospheric currents
                  
                  # read the northern hemisphere MIX data
                  vals=dMIX.read('Potential '+'North'+' [V]',time)[:-1,:]/1000.0
                  psiN_dict={'data':vals,'name':r'$\Phi$','units':r'kV'}
                  vals=dMIX.read('Pedersen conductance '+'North'+' [S]',time)[:-1,:]
                  sigmapN_dict={'data':vals,'name':r'$\Sigma_{P}$','units':r'S'}
                  vals=dMIX.read('Hall conductance '+'North'+' [S]',time)[:-1,:]
                  sigmahN_dict={'data':vals,'name':r'$\Sigma_{H}$','units':r'S'}

                  # read the southern hemisphere MIX data
                  vals=dMIX.read('Potential '+'South'+' [V]',time)[:-1,:]/1000.0
                  psiS_dict={'data':vals,'name':r'$\Phi$','units':r'kV'}
                  vals=dMIX.read('Pedersen conductance '+'South'+' [S]',time)[:-1,:]
                  sigmapS_dict={'data':vals,'name':r'$\Sigma_{P}$','units':r'S'}
                  vals=dMIX.read('Hall conductance '+'South'+' [S]',time)[:-1,:]
                  sigmahS_dict={'data':vals,'name':r'$\Sigma_{H}$','units':r'S'}

                  # compute the electric field vectors
                  ((phiN_dict,thetaN_dict),
                   (ephiN_dict,ethetaN_dict)) = pyLTR.Physics.MIXCalcs.efieldDict(
                           xN_dict, yN_dict, psiN_dict, ri=ion_rho)

                  ((phiS_dict,thetaS_dict),
                   (ephiS_dict,ethetaS_dict)) = pyLTR.Physics.MIXCalcs.efieldDict(
                           xS_dict, yS_dict, psiS_dict, ri=ion_rho, oh=True)

                  # compute total, Pedersen, and Hall ionospheric current vectors
                  ((JphiN_dict,JthetaN_dict),
                   (JpedphiN_dict,JpedthetaN_dict),
                   (JhallphiN_dict,JhallthetaN_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                           (ephiN_dict,ethetaN_dict), sigmapN_dict, sigmahN_dict, colatNdict['data'])

                  ((JphiS_dict,JthetaS_dict),
                   (JpedphiS_dict,JpedthetaS_dict),
                   (JhallphiS_dict,JhallthetaS_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                           (ephiS_dict,ethetaS_dict), sigmapS_dict, sigmahS_dict, colatSdict['data'])
                                

                  
                  #
                  # transform and interpolate ionospheric currents to requested coordinates
                  # if not Solar Magnetic
                  #

                  # these are the fixed coordinates to which we will interpolate
                  phiN_interp = longNdict['data']
                  thetaN_interp = colatNdict['data']

                  # retrieve Northern MIX currents in SM coordinates
                  phiN_sm = longNdict['data']
                  thetaN_sm = colatNdict['data']
                  rhoN_sm = p.full(phiN_sm.shape, ion_rho)
                  JphiN_sm = JphiN_dict['data']
                  JthetaN_sm = JthetaN_dict['data']
                  JrhoN_sm = p.full(JphiN_sm.shape, 0)
                  
                  # convert to Cartesian coordinates
                  ( xN_sm,  yN_sm,  zN_sm,
                     JxN_sm, JyN_sm, JzN_sm) = pyLTR.transform.SPHtoCAR(
                     phiN_sm, thetaN_sm, rhoN_sm,
                     JphiN_sm, JthetaN_sm, JrhoN_sm
                  )

                  
                  # these are fixed coordinates to which we will interpolate
                  phiS_interp = longSdict['data']
                  thetaS_interp = colatSdict['data']

                  # retrieve Southern MIX currents in SM coordinates
                  phiS_sm = longSdict['data']
                  thetaS_sm = colatSdict['data']
                  rhoS_sm = p.full(phiS_sm.shape, ion_rho)
                  JphiS_sm = JphiS_dict['data']
                  JthetaS_sm = JthetaS_dict['data']
                  JrhoS_sm = p.full(JphiS_sm.shape, 0)
                  
                  # convert to Cartesian coordinates
                  ( xS_sm,  yS_sm,  zS_sm,
                     JxS_sm, JyS_sm, JzS_sm) = pyLTR.transform.SPHtoCAR(
                     phiS_sm, thetaS_sm, rhoS_sm,
                     JphiS_sm, JthetaS_sm, JrhoS_sm
                  )
                  
                  
                  # rotate into requested coordinates
                  if geoGrid:

                     # rotate SM locations and currents into GEO coordinates
                     ( xN_geo,  yN_geo,  zN_geo) = pyLTR.transform.SMtoGEO(
                        xN_sm, yN_sm, zN_sm, time
                     )
                     (JxN_geo, JyN_geo, JzN_geo) = pyLTR.transform.SMtoGEO(
                        JxN_sm, JyN_sm, JzN_sm, time
                     )

                     # convert back to spherical
                     ( phiN_geo,  thetaN_geo,  rhoN_geo,
                      JphiN_geo, JthetaN_geo, JrhoN_geo) = pyLTR.transform.CARtoSPH(
                         xN_geo,  yN_geo,  zN_geo,
                        JxN_geo, JyN_geo, JzN_geo
                     )
                     
                     # prepare to interpolate onto fixed GEO grid

                     # add 2*pi to smallest phi, concatenate to end;
                     # subtract 2*pi from largest phi, concatenate to begining;
                     # this ensures that there are no gaps in the periodic grid to
                     # to be interpolated
                     small_idx = phiN_geo[:,0].argmin()
                     large_idx = phiN_geo[:,0].argmax()
                     phiN_pre = p.vstack(
                        (phiN_geo[large_idx,:] - 2*p.pi, phiN_geo, phiN_geo[small_idx,:] + 2*p.pi)
                     )
                     thetaN_pre = p.vstack(
                        (thetaN_geo[large_idx,:], thetaN_geo, thetaN_geo[small_idx,:])
                     )
                     JphiN_pre = p.vstack(
                        (JphiN_geo[large_idx,:], JphiN_geo, JphiN_geo[small_idx,:])
                     )
                     JthetaN_pre = p.vstack(
                        (JthetaN_geo[large_idx,:], JthetaN_geo, JthetaN_geo[small_idx,:])
                     )


                     # rotate SM locations and currents into GEO coordinates
                     ( xS_geo,  yS_geo,  zS_geo) = pyLTR.transform.SMtoGEO(
                        xS_sm, yS_sm, zS_sm, time
                     )
                     (JxS_geo, JyS_geo, JzS_geo) = pyLTR.transform.SMtoGEO(
                        JxS_sm, JyS_sm, JzS_sm, time
                     )

                     # convert back to spherical
                     ( phiS_geo,  thetaS_geo,  rhoS_geo,
                      JphiS_geo, JthetaS_geo, JrhoS_geo) = pyLTR.transform.CARtoSPH(
                         xS_geo,  yS_geo,  zS_geo,
                        JxS_geo, JyS_geo, JzS_geo
                     )
                     
                     # prepare to interpolate onto fixed GEO grid

                     # add 2*pi to smallest phi, concatenate to end;
                     # subtract 2*pi from largest phi, concatenate to begining;
                     # this ensures that there are no gaps in the periodic grid to
                     # to be interpolated
                     small_idx = phiS_geo[:,0].argmin()
                     large_idx = phiS_geo[:,0].argmax()
                     phiS_pre = p.vstack(
                        (phiS_geo[large_idx,:] - 2*p.pi, phiS_geo, phiS_geo[small_idx,:] + 2*p.pi)
                     )
                     thetaS_pre = p.vstack(
                        (thetaS_geo[large_idx,:], thetaS_geo, thetaS_geo[small_idx,:])
                     )
                     JphiS_pre = p.vstack(
                        (JphiS_geo[large_idx,:], JphiS_geo, JphiS_geo[small_idx,:])
                     )
                     JthetaS_pre = p.vstack(
                        (JthetaS_geo[large_idx,:], JthetaS_geo, JthetaS_geo[small_idx,:])
                     )


                  elif magGrid:

                     # rotate SM locations and currents into MAG coordinates
                     ( xN_mag,  yN_mag,  zN_mag) = pyLTR.transform.SMtoMAG(
                        xN_sm, yN_sm, zN_sm, time
                     )
                     (JxN_mag, JyN_mag, JzN_mag) = pyLTR.transform.SMtoMAG(
                        JxN_sm, JyN_sm, JzN_sm, time
                     )

                     # convert back to spherical
                     ( phiN_mag,  thetaN_mag,  rhoN_mag,
                      JphiN_mag, JthetaN_mag, JrhoN_mag) = pyLTR.transform.CARtoSPH(
                         xN_mag,  yN_mag,  zN_mag,
                        JxN_mag, JyN_mag, JzN_mag
                     )
                     
                     # prepare to interpolate onto fixed MAG grid

                     # add 2*pi to smallest phi, concatenate to end;
                     # subtract 2*pi from largest phi, concatenate to begining;
                     # this ensures that there are no gaps in the periodic grid to
                     # to be interpolated
                     small_idx = phiN_mag[:,0].argmin()
                     large_idx = phiN_mag[:,0].argmax()
                     phiN_pre = p.vstack(
                        (phiN_mag[large_idx,:] - 2*p.pi, phiN_mag, phiN_mag[small_idx,:] + 2*p.pi)
                     )
                     thetaN_pre = p.vstack(
                        (thetaN_mag[large_idx,:], thetaN_mag, thetaN_mag[small_idx,:])
                     )
                     JphiN_pre = p.vstack(
                        (JphiN_mag[large_idx,:], JphiN_mag, JphiN_mag[small_idx,:])
                     )
                     JthetaN_pre = p.vstack(
                        (JthetaN_mag[large_idx,:], JthetaN_mag, JthetaN_mag[small_idx,:])
                     )


                     # rotate SM locations and currents into MAG coordinates
                     ( xS_mag,  yS_mag,  zS_mag) = pyLTR.transform.SMtoMAG(
                        xS_sm, yS_sm, zS_sm, time
                     )
                     (JxS_mag, JyS_mag, JzS_mag) = pyLTR.transform.SMtoMAG(
                        JxS_sm, JyS_sm, JzS_sm, time
                     )

                     # convert back to spherical
                     ( phiS_mag,  thetaS_mag,  rhoS_mag,
                      JphiS_mag, JthetaS_mag, JrhoS_mag) = pyLTR.transform.CARtoSPH(
                         xS_mag,  yS_mag,  zS_mag,
                        JxS_mag, JyS_mag, JzS_mag
                     )
                     
                     # prepare to interpolate onto fixed MAG grid

                     # add 2*pi to smallest phi, concatenate to end;
                     # subtract 2*pi from largest phi, concatenate to begining;
                     # this ensures that there are no gaps in the periodic grid to
                     # to be interpolated
                     small_idx = phiS_mag[:,0].argmin()
                     large_idx = phiS_mag[:,0].argmax()
                     phiS_pre = p.vstack(
                        (phiS_mag[large_idx,:] - 2*p.pi, phiS_mag, phiS_mag[small_idx,:] + 2*p.pi)
                     )
                     thetaS_pre = p.vstack(
                        (thetaS_mag[large_idx,:], thetaS_mag, thetaS_mag[small_idx,:])
                     )
                     JphiS_pre = p.vstack(
                        (JphiS_mag[large_idx,:], JphiS_mag, JphiS_mag[small_idx,:])
                     )
                     JthetaS_pre = p.vstack(
                        (JthetaS_mag[large_idx,:], JthetaS_mag, JthetaS_mag[small_idx,:])
                     )
                  
                  else:

                     # simply copy inputs if requested coordinates are SM
                     phiN_pre = phiN_sm
                     thetaN_pre = thetaN_sm
                     JphiN_pre = JphiN_sm
                     JthetaN_pre = JthetaN_sm
                     
                     phiS_pre = phiS_sm
                     thetaS_pre = thetaS_sm
                     JphiS_pre = JphiS_sm
                     JthetaS_pre = JthetaS_sm
                     
                  
                  # finally, interpolate onto fixed coordinates
                  JphiN_interp = sinterp.griddata(
                     (phiN_pre.reshape(-1),
                      thetaN_pre.reshape(-1)),
                     JphiN_pre.reshape(-1),
                     (phiN_interp.reshape(-1), 
                      thetaN_interp.reshape(-1))
                  ).reshape(phiN_interp.shape)
                  
                  JthetaN_interp = sinterp.griddata(
                     (phiN_pre.reshape(-1),
                      thetaN_pre.reshape(-1)),
                     JthetaN_pre.reshape(-1),
                     (phiN_interp.reshape(-1), 
                      thetaN_interp.reshape(-1))
                  ).reshape(thetaN_interp.shape)

                  
                  JphiS_interp = sinterp.griddata(
                     (phiS_pre.reshape(-1),
                      thetaS_pre.reshape(-1)),
                     JphiS_pre.reshape(-1),
                     (phiS_interp.reshape(-1), 
                      thetaS_interp.reshape(-1))
                  ).reshape(phiS_interp.shape)
                  
                  JthetaS_interp = sinterp.griddata(
                     (phiS_pre.reshape(-1),
                      thetaS_pre.reshape(-1)),
                     JthetaS_pre.reshape(-1),
                     (phiS_interp.reshape(-1), 
                      thetaS_interp.reshape(-1))
                  ).reshape(thetaS_interp.shape)
                  
                  
                  # if DALECS are not initialized yet, do it now
                  if dalecs_N_ion is None:
                     print()
                     print("Initializing Northern MIX DALECS")
                     begin = perf_counter()
                     
                     print("   ...ionospheric currents")
                     # create 1-amp DALECS to be scaled inside the loop
                     dalecs_N_ion = DALECS.dalecs((longNdict['data'], colatNdict['data']),
                                                   ion_rho=ion_rho, fac=False, equator=False)
                           
                     print("   ...field-aligned currents")
                     dalecs_N_fac = DALECS.dalecs((longNdict['data'], colatNdict['data']),
                                                   ion_rho=ion_rho, iono=False)
                           
                     dalecs_N_fac = dalecs_N_fac.trim(rho_max=2.5*ion_rho)
                     
                     # convert _ion and _fac DALECS to Cartesian coordinates to avoid
                     # unnecessary conversions to/from spherical in the loop
                     dalecs_N_ion.cartesian()
                     dalecs_N_fac.cartesian()

                     
                     # push DALECSs to pool of workers
                     pool_dalecs_N_ion = Pool(nprocs, initializer, (dalecs_N_ion,))
                     pool_dalecs_N_fac = Pool(nprocs, initializer, (dalecs_N_fac,))
                     
                     
                     print("done after %f seconds"%(perf_counter() - begin))


                  if dalecs_S_ion is None:
                     print()
                     print("Initializing Southern MIX DALECS")
                     begin = perf_counter()

                     print("   ...ionospheric currents")
                     # create 1-amp DALECS to be scaled inside the loop
                     dalecs_S_ion = DALECS.dalecs((longSdict['data'], colatSdict['data']),
                                                   ion_rho=ion_rho, fac=False, equator=False)
                           
                     print("   ...field-aligned currents")
                     dalecs_S_fac = DALECS.dalecs((longSdict['data'], colatSdict['data']),
                                                   ion_rho=ion_rho, iono=False)
                           
                     dalecs_S_fac = dalecs_S_fac.trim(rho_max=2.5*ion_rho)
                     
                     # convert _ion and _fac DALECS to Cartesian coordinates to avoid
                     # unnecessary conversions to/from spherical in the loop
                     dalecs_S_ion.cartesian()
                     dalecs_S_fac.cartesian()

                     
                     # push DALECSs to pool of workers
                     pool_dalecs_S_ion = Pool(nprocs, initializer, (dalecs_S_ion,))
                     pool_dalecs_S_fac = Pool(nprocs, initializer, (dalecs_S_fac,))
                     
                     
                     print("done after %f seconds"%(perf_counter() - begin))


                  # update Northern MIX DALECS and integrate BS
                  print()
                  print("Scale northern DALECS and integrate Biot-Savart")
                  begin = perf_counter()
                                    
                  
                  if nprocs > 1:
              
                     # scale DALECS in each worker
                     _ = pool_dalecs_N_ion.map(
                        scale_parallel, 
                        ((mixN_weights * JphiN_interp/1e6,
                          mixN_weights * JthetaN_interp/1e6) 
                         for _ in range(nprocs) )
                     )

                     # integrate Biot-Savart in each worker; return results
                     bs_args = [(obs_xyz,) for obs_xyz in 
                                zip(p.array_split(obs_x, nprocs), 
                                    p.array_split(obs_y, nprocs), 
                                    p.array_split(obs_z, nprocs) ) ]
                     (dBxN_ion,
                      dByN_ion,
                      dBzN_ion) = (p.hstack(result) for result in
                                     zip(*pool_dalecs_N_ion.starmap(
                                        bs_parallel, bs_args))
                                    )

                  else:
                     dalecs_N_ion.scale(
                        (mixN_weights * JphiN_interp/1e6, 
                        mixN_weights * JthetaN_interp/1e6)
                     )

                     (dBxN_ion,
                      dByN_ion,
                      dBzN_ion) = dalecs_N_ion.bs_cart((obs_x, obs_y, obs_z),
                                                       matrix=mix_bs_mx)                  
                  
                  
                  if nprocs > 1:
                     
                     # scale DALECS in each worker
                     _ = pool_dalecs_N_fac.map(
                        scale_parallel, 
                        ((mixN_weights * JphiN_interp/1e6,
                          mixN_weights * JthetaN_interp/1e6) 
                         for _ in range(nprocs) )
                     )

                     # integrate Biot-Savart in each worker; return results
                     bs_args = [(obs_xyz,) for obs_xyz in 
                                zip(p.array_split(obs_x, nprocs), 
                                    p.array_split(obs_y, nprocs), 
                                    p.array_split(obs_z, nprocs) ) ]
                     (dBxN_fac,
                      dByN_fac,
                      dBzN_fac) = (p.hstack(result) for result in
                                     zip(*pool_dalecs_N_fac.starmap(
                                        bs_parallel, bs_args))
                                    )

                  else:
                     dalecs_N_fac.scale(
                        (mixN_weights * JphiN_interp/1e6, 
                        mixN_weights * JthetaN_interp/1e6)
                     )
                  
                     (dBxN_fac,
                      dByN_fac,
                      dBzN_fac) = dalecs_N_fac.bs_cart((obs_x, obs_y, obs_z),
                                                       matrix=mix_bs_mx)
                                    
                  
                  print("...done after %f seconds"%(perf_counter() - begin))
                  

                  # update Southern MIX DALECS and integrate BS
                  print()
                  print("Scale southern DALECS and integrate Biot-Savart")
                  begin = perf_counter()
                  


                  if nprocs > 1:
                     
                     # scale DALECS in each worker
                     _ = pool_dalecs_S_ion.map(
                        scale_parallel, 
                        ((mixS_weights * JphiS_interp/1e6,
                          mixS_weights * JthetaS_interp/1e6) 
                         for _ in range(nprocs) )
                     )

                     # integrate Biot-Savart in each worker; return results
                     bs_args = [(obs_xyz,) for obs_xyz in 
                                zip(p.array_split(obs_x, nprocs), 
                                    p.array_split(obs_y, nprocs), 
                                    p.array_split(obs_z, nprocs) ) ]
                     (dBxS_ion,
                      dByS_ion,
                      dBzS_ion) = (p.hstack(result) for result in
                                     zip(*pool_dalecs_S_ion.starmap(
                                        bs_parallel, bs_args))
                                    )

                  else:
                     dalecs_S_ion.scale(
                        (mixS_weights * JphiS_interp/1e6, 
                        mixS_weights * JthetaS_interp/1e6)
                     ) 
                  
                     (dBxS_ion,
                      dByS_ion,
                      dBzS_ion) = dalecs_S_ion.bs_cart((obs_x, obs_y, obs_z),
                                                       matrix=mix_bs_mx)


                  if nprocs > 1:
                     
                     # scale DALECS in each worker
                     _ = pool_dalecs_S_fac.map(
                        scale_parallel, 
                        ((mixS_weights * JphiS_interp/1e6,
                          mixS_weights * JthetaS_interp/1e6) 
                         for _ in range(nprocs) )
                     )

                     # integrate Biot-Savart in each worker; return results
                     bs_args = [(obs_xyz,) for obs_xyz in 
                                zip(p.array_split(obs_x, nprocs), 
                                    p.array_split(obs_y, nprocs), 
                                    p.array_split(obs_z, nprocs) ) ]
                     (dBxS_fac,
                      dByS_fac,
                      dBzS_fac) = (p.hstack(result) for result in
                                     zip(*pool_dalecs_S_fac.starmap(
                                        bs_parallel, bs_args))
                                    )

                  else:
                     dalecs_S_fac.scale(
                        (mixS_weights * JphiS_interp/1e6, 
                        mixS_weights * JthetaS_interp/1e6)
                     )
                     (dBxS_fac,
                      dByS_fac,
                      dBzS_fac) = dalecs_S_fac.bs_cart((obs_x, obs_y, obs_z),
                                                       matrix=mix_bs_mx)
                                    
                  
                  print("...done after %f seconds"%(perf_counter() - begin))
                  
              else:
                  # set dBs to zero if no MIX data is available
                  dBxN_ion = p.zeros(p.array(obs_x).shape)
                  dByN_ion = p.zeros(p.array(obs_y).shape)
                  dBzN_ion = p.zeros(p.array(obs_z).shape)
                  dBxN_fac = p.zeros(p.array(obs_x).shape)
                  dByN_fac = p.zeros(p.array(obs_y).shape)
                  dBzN_fac = p.zeros(p.array(obs_z).shape)

                  dBxS_ion = p.zeros(p.array(obs_x).shape)
                  dByS_ion = p.zeros(p.array(obs_y).shape)
                  dBzS_ion = p.zeros(p.array(obs_z).shape)
                  dBxS_fac = p.zeros(p.array(obs_x).shape)
                  dByS_fac = p.zeros(p.array(obs_y).shape)
                  dBzS_fac = p.zeros(p.array(obs_z).shape)



              if dTIE:
                      
                  print()
                  print("TIEGCM")

                  #
                  # transform and interpolate ionospheric currents to requested coordinates
                  # if not Geomagnetic
                  #

                  # these are the fixed coordinates to which we will interpolate
                  phiT_interp = phiT_mag_interp
                  thetaT_interp = thetaT_mag_interp

                  # retrieve Northern MIX currents in SM coordinates
                  phiT_mag = phiT_mag_interp
                  thetaT_mag = thetaT_mag_interp
                  rhoT_mag = p.full(phiT_mag.shape, ion_rho)
                  JphiT_mag = dTIE.read('KQPHI', time).T.copy()[:-1,:]
                  JthetaT_mag = -dTIE.read('KQLAM', time).T.copy()[:-1,:]
                  JrhoT_mag = p.full(JphiT_mag.shape, 0)
                  
                  # convert to Cartesian coordinates
                  ( xT_mag,  yT_mag,  zT_mag,
                     JxT_mag, JyT_mag, JzT_mag) = pyLTR.transform.SPHtoCAR(
                     phiT_mag, thetaT_mag, rhoT_mag,
                     JphiT_mag, JthetaT_mag, JrhoT_mag
                  )

                                    
                  # rotate into requested coordinates
                  if geoGrid:

                     # rotate MAG locations and currents into GEO coordinates
                     ( xT_geo,  yT_geo,  zT_geo) = pyLTR.transform.MAGtoGEO(
                        xT_mag, yT_mag, zT_mag, time
                     )
                     (JxT_geo, JyT_geo, JzT_geo) = pyLTR.transform.MAGtoGEO(
                        JxT_mag, JyT_mag, JzT_mag, time
                     )

                     # convert back to spherical
                     ( phiT_geo,  thetaT_geo,  rhoT_geo,
                      JphiT_geo, JthetaT_geo, JrhoT_geo) = pyLTR.transform.CARtoSPH(
                         xT_geo,  yT_geo,  zT_geo,
                        JxT_geo, JyT_geo, JzT_geo
                     )
                     
                     # prepare to interpolate onto fixed GEO grid

                     # add 2*pi to smallest phi, concatenate to end;
                     # subtract 2*pi from largest phi, concatenate to begining;
                     # this ensures that there are no gaps in the periodic grid to
                     # to be interpolated
                     small_idx = phiT_geo[:,0].argmin()
                     large_idx = phiT_geo[:,0].argmax()
                     phiT_pre = p.vstack(
                        (phiT_geo[large_idx,:] - 2*p.pi, phiT_geo, phiT_geo[small_idx,:] + 2*p.pi)
                     )
                     thetaT_pre = p.vstack(
                        (thetaT_geo[large_idx,:], thetaT_geo, thetaT_geo[small_idx,:])
                     )
                     JphiT_pre = p.vstack(
                        (JphiT_geo[large_idx,:], JphiT_geo, JphiT_geo[small_idx,:])
                     )
                     JthetaT_pre = p.vstack(
                        (JthetaT_geo[large_idx,:], JthetaT_geo, JthetaT_geo[small_idx,:])
                     )

                  elif smGrid:

                     # rotate MAG locations and currents into SM coordinates
                     ( xT_sm,  yT_sm,  zT_sm) = pyLTR.transform.MAGtoSM(
                        xT_mag, yT_mag, zT_mag, time
                     )
                     (JxT_sm, JyT_sm, JzT_sm) = pyLTR.transform.MAGtoSM(
                        JxT_mag, JyT_mag, JzT_mag, time
                     )

                     # convert back to spherical
                     ( phiT_sm,  thetaT_sm,  rhoT_sm,
                      JphiT_sm, JthetaT_sm, JrhoT_sm) = pyLTR.transform.CARtoSPH(
                         xT_sm,  yT_sm,  zT_sm,
                        JxT_sm, JyT_sm, JzT_sm
                     )
                     
                     # prepare to interpolate onto fixed SM grid

                     # add 2*pi to smallest phi, concatenate to end;
                     # subtract 2*pi from largest phi, concatenate to begining;
                     # this ensures that there are no gaps in the periodic grid to
                     # to be interpolated
                     small_idx = phiN_sm[:,0].argmin()
                     large_idx = phiN_sm[:,0].argmax()
                     phiT_pre = p.vstack(
                        (phiN_sm[large_idx,:] - 2*p.pi, phiN_sm, phiN_sm[small_idx,:] + 2*p.pi)
                     )
                     thetaT_pre = p.vstack(
                        (thetaN_sm[large_idx,:], thetaN_sm, thetaN_sm[small_idx,:])
                     )
                     JphiT_pre = p.vstack(
                        (JphiN_sm[large_idx,:], JphiN_sm, JphiN_sm[small_idx,:])
                     )
                     JthetaT_pre = p.vstack(
                        (JthetaN_sm[large_idx,:], JthetaN_sm, JthetaN_sm[small_idx,:])
                     )

                  else:
                     # simply copy inputs if requested coordinates are MAG
                     phiT_pre = phiT_mag
                     thetaT_pre = thetaT_mag
                     JphiT_pre = JphiT_mag
                     JthetaT_pre = JthetaT_mag
                     
                     phiT_pre = phiT_mag
                     thetaT_pre = thetaT_mag
                     JphiT_pre = JphiT_mag
                     JthetaT_pre = JthetaT_mag
                     
                                   
                  # finally, interpolate onto fixed coordinates
                  JphiT_interp = sinterp.griddata(
                     (phiT_pre.reshape(-1),
                      thetaT_pre.reshape(-1)),
                     JphiT_pre.reshape(-1),
                     (phiT_interp.reshape(-1), 
                      thetaT_interp.reshape(-1))
                  ).reshape(phiT_interp.shape)
                  
                  JthetaT_interp = sinterp.griddata(
                     (phiT_pre.reshape(-1),
                      thetaT_pre.reshape(-1)),
                     JthetaT_pre.reshape(-1),
                     (phiT_interp.reshape(-1), 
                      thetaT_interp.reshape(-1))
                  ).reshape(thetaT_interp.shape)




                  if dalecs_T_ion is None:

                     print()
                     print("Initializing TIEGCM DALECS")
                     begin = perf_counter()

                     # initialize DALECS for ionosphere and field-aligned currents
                     print("   ...ionospheric currents")
                     dalecs_T_ion = DALECS.dalecs((phiT_mag_interp, thetaT_mag_interp),
                                                   ion_rho=ion_rho, fac=False, equator=False)
                           
                     print("   ...field-aligned currents")
                     dalecs_T_fac = DALECS.dalecs((phiT_mag_interp, thetaT_mag_interp),
                                                   ion_rho=ion_rho, iono=False)
                     
                     dalecs_T_fac = dalecs_T_fac.trim(rho_max=2.5*ion_rho)
                        
                     # convert to Cartesian coordinates
                     dalecs_T_ion.cartesian()
                     dalecs_T_fac.cartesian()
                     
                     # push DALECSs to pool of workers
                     pool_dalecs_T_ion = Pool(nprocs, initializer, (dalecs_T_ion,))
                     pool_dalecs_T_fac = Pool(nprocs, initializer, (dalecs_T_fac,))

                     print("done after %f seconds"%(perf_counter() - begin))





                  # update TIEGCM DALECS and integrate BS
                  print()
                  print("Scale DALECS and integrate Biot-Savart")
                  begin = perf_counter()
                  
                  
                  
                  if nprocs > 1:
                     # scale DALECS in each worker
                     _ = pool_dalecs_T_ion.map(
                        scale_parallel, 
                        ((TIE_weights * JphiT_interp,
                          TIE_weights * JthetaT_interp) 
                         for _ in range(nprocs) )
                     )

                     # integrate Biot-Savart in each worker; return results
                     bs_args = [(obs_xyz,) for obs_xyz in 
                                zip(p.array_split(obs_x, nprocs), 
                                    p.array_split(obs_y, nprocs), 
                                    p.array_split(obs_z, nprocs) ) ]
                     (dBxTIE_ion,
                      dByTIE_ion,
                      dBzTIE_ion) = (p.hstack(result) for result in
                                     zip(*pool_dalecs_T_ion.starmap(
                                        bs_parallel, bs_args))
                                    )

                  else:
                     dalecs_T_ion.scale(
                        (TIE_weights * JphiT_interp, 
                        TIE_weights * JthetaT_interp)
                     )
                     (dBxTIE_ion,
                      dByTIE_ion,
                      dBzTIE_ion) = dalecs_T_ion.bs_cart((obs_x, obs_y, obs_z),
                                                         matrix=mix_bs_mx)


                  if nprocs > 1:
                     # scale DALECS in each worker
                     _ = pool_dalecs_T_fac.map(
                        scale_parallel, 
                        ((TIE_weights * JphiT_interp,
                          TIE_weights * JthetaT_interp) 
                         for _ in range(nprocs) )
                     )

                     # integrate Biot-Savart in each worker; return results
                     bs_args = [(obs_xyz,) for obs_xyz in 
                                zip(p.array_split(obs_x, nprocs), 
                                    p.array_split(obs_y, nprocs), 
                                    p.array_split(obs_z, nprocs) ) ]
                     (dBxTIE_fac,
                      dByTIE_fac,
                      dBzTIE_fac) = (p.hstack(result) for result in
                                     zip(*pool_dalecs_T_fac.starmap(
                                        bs_parallel, bs_args))
                                    )
                     
                  else:
                     dalecs_T_fac.scale(
                        (TIE_weights * JphiT_interp, 
                        TIE_weights * JthetaT_interp)
                     )
                     (dBxTIE_fac,
                      dByTIE_fac,
                      dBzTIE_fac) = dalecs_T_fac.bs_cart((obs_x, obs_y, obs_z),
                                                         matrix=mix_bs_mx)
                  
                  
                  print("...done after %f seconds"%(perf_counter() - begin))

              else:
                  # set dBs to zero if no TIEGCM data is available
                  dBxTIE_ion = p.zeros(p.array(obs_x).shape)
                  dByTIE_ion = p.zeros(p.array(obs_y).shape)
                  dBzTIE_ion = p.zeros(p.array(obs_z).shape)
                  dBxTIE_fac = p.zeros(p.array(obs_x).shape)
                  dByTIE_fac = p.zeros(p.array(obs_y).shape)
                  dBzTIE_fac = p.zeros(p.array(obs_z).shape)


              
              if dLFM:
                  
                  print()
                  print("LFM")
                  begin = perf_counter()

                  # retrieve magnetospheric currents from LFM file, then calculate deltaB
                  # in SM coordinates before transforming output to requested coordinates
                  # (matrix trick used above for MIX and TIEGCM will not help us much 
                  #  here because we still have to apply BS to each current element)
                  Bx_sm = dLFM.read('bx_', time) # this is in G
                  By_sm = dLFM.read('by_', time) # this is in G
                  Bz_sm = dLFM.read('bz_', time) # this is in G
                  Jx_sm, Jy_sm, Jz_sm = pyLTR.Physics.LFMCurrent(
                    hgridcc, Bx_sm, By_sm, Bz_sm, rion=1) # ...should be A/m^2 given default input units
                  

                  if nprocs > 1:
                     # create a list of argument lists for bs_cart
                     bs_args = [((xJ_sm, yJ_sm, zJ_sm),
                                 (Jx_sm, Jy_sm, Jz_sm),
                                 dV_sm,
                                 obs_xyz) for obs_xyz in 
                                zip(p.array_split(obs_x_sm, nprocs), 
                                    p.array_split(obs_y_sm, nprocs), 
                                    p.array_split(obs_z_sm, nprocs) ) ]
                                          
                     with Pool(processes=nprocs) as pool:
                        (dBxLFM,
                         dByLFM,
                         dBzLFM) = (p.hstack(out) for out in
                                      zip(*pool.starmap(
                                          DALECS.bs_cart, bs_args
                                          ))
                                      )
                  else:
                     (dBxLFM,
                      dByLFM,
                      dBzLFM) = DALECS.bs_cart(
                         (xJ_sm, yJ_sm, zJ_sm),
                         (Jx_sm, Jy_sm, Jz_sm),
                         dV_sm,
                         (obs_x_sm, obs_y_sm, obs_z_sm)
                     )


                  # rotate dB?LFM into requested coordinates
                  if geoGrid:
                     # rotate output SM locations into GEO coordinates
                     (dBxLFM, dByLFM, dBzLFM) = pyLTR.transform.SMtoGEO(
                         dBxLFM, dByLFM, dBzLFM, time
                      )

                  elif magGrid:
                     # rotate output SM locations into MAG coordinates
                     (dBxLFM, dByLFM, dBzLFM) = pyLTR.transform.SMtoMAG(
                         dBxLFM, dByLFM, dBzLFM, time
                      )
                  
                  print("...done after %f seconds"%(perf_counter() - begin))
              
              else:
                  # set dBs to zero if no LFM data is available
                  dBxLFM = p.zeros(p.array(obs_x).shape)
                  dByLFM = p.zeros(p.array(obs_y).shape)
                  dBzLFM = p.zeros(p.array(obs_z).shape)

            
              # transform Cartesian dBs to spherical coordinates for output
              # (leave position vectors unchanged for subsequent iterations)
              _, _, _, dBphiN_ion, dBthetaN_ion, dBrhoN_ion = pyLTR.transform.CARtoSPH(
                 obs_x, obs_y, obs_z, dBxN_ion, dByN_ion, dBzN_ion
              )
              _, _, _, dBphiN_fac, dBthetaN_fac, dBrhoN_fac = pyLTR.transform.CARtoSPH(
                 obs_x, obs_y, obs_z, dBxN_fac, dByN_fac, dBzN_fac
              )

              _, _, _, dBphiS_ion, dBthetaS_ion, dBrhoS_ion = pyLTR.transform.CARtoSPH(
                 obs_x, obs_y, obs_z, dBxS_ion, dByS_ion, dBzS_ion
              )
              _, _, _, dBphiS_fac, dBthetaS_fac, dBrhoS_fac = pyLTR.transform.CARtoSPH(
                 obs_x, obs_y, obs_z, dBxS_fac, dByS_fac, dBzS_fac
              )

              _, _, _, dBphiTIE_ion, dBthetaTIE_ion, dBrhoTIE_ion = pyLTR.transform.CARtoSPH(
                 obs_x, obs_y, obs_z, dBxTIE_ion, dByTIE_ion, dBzTIE_ion
              )
              _, _, _, dBphiTIE_fac, dBthetaTIE_fac, dBrhoTIE_fac = pyLTR.transform.CARtoSPH(
                 obs_x, obs_y, obs_z, dBxTIE_fac, dByTIE_fac, dBzTIE_fac
              )
              
              _, _, _, dBphiLFM, dBthetaLFM, dBrhoLFM = pyLTR.transform.CARtoSPH(
                 obs_x, obs_y, obs_z, dBxLFM, dByLFM, dBzLFM
              )                                     


              # combine dBs into lists of dicts, then dump to binary file
              toPickle = {}

              toPickle['pov'] = 'north'
              toPickle['dB_obs'] = [{'data':obs_phi,'name':r'$\phi$','units':r'rad'},
                                   {'data':obs_theta,'name':r'$\theta$','units':r'rad'},
                                   {'data':obs_rho,'name':r'$\rho$','units':r'm'}]

              if geoGrid:
                 toPickle['coordinates'] = 'Geographic'
              elif magGrid:
                 toPickle['coordinates'] = 'Geomagnetic'
              else:
                 toPickle['coordinates'] = 'Solar Magnetic'

              dB_ion = [{'data': (dBphiN_ion + dBphiS_ion + dBphiTIE_ion)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\phi}$'},
                        {'data': (dBthetaN_ion + dBthetaS_ion + dBthetaTIE_ion)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\theta}$'},
                        {'data': (dBrhoN_ion + dBrhoS_ion + dBrhoTIE_ion)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\rho}$'}]
              dB_fac = [{'data': (dBphiN_fac + dBphiS_fac + dBphiTIE_fac)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\phi}$'},
                        {'data': (dBthetaN_fac + dBthetaS_fac + dBthetaTIE_fac)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\theta}$'},
                        {'data': (dBrhoN_fac + dBrhoS_fac + dBrhoTIE_fac)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\rho}$'}]
              dB_mag = [{'data': (dBphiLFM)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\phi}$'},
                        {'data': (dBthetaLFM)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\theta}$'},
                        {'data': (dBrhoLFM)*1e9,'units':'nT' ,'name':r'$\Delta \marthrm{B}_{\rho}$'}]

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
           # if geoGrid is True, obsGEO will be constant;
           # if magGrid is True, obsMAG will be constant;
           # if smGrid is True, obsSM will be constant;
           #
           obsGEO[0].append(obs_phi_geo)
           obsGEO[1].append(obs_theta_geo)
           obsGEO[2].append(obs_rho_geo)
           obsMAG[0].append(obs_phi_mag)
           obsMAG[1].append(obs_theta_mag)
           obsMAG[2].append(obs_rho_mag)
           obsSM[0].append(obs_phi_sm)
           obsSM[1].append(obs_theta_sm)
           obsSM[2].append(obs_rho_sm)


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


    

    if nprocs > 1 and dalecs_N_ion is not None:
       pool_dalecs_N_ion.close()
    if nprocs > 1 and dalecs_N_fac is not None:
       pool_dalecs_N_fac.close()
    
    if nprocs > 1 and dalecs_S_ion is not None:
       pool_dalecs_S_ion.close()
    if nprocs > 1 and dalecs_S_fac is not None:
       pool_dalecs_S_fac.close()
    
    if nprocs > 1 and dalecs_T_ion is not None:
       pool_dalecs_T_ion.close()
    if nprocs > 1 and dalecs_T_fac is not None:
       pool_dalecs_T_fac.close()
    
    
    
    
    #
    # generate pyLTR.TimeSeries objects for final output; convert units to nT
    #
    dBObs = [] # output is a list of dictionaries of time series objects
    for obs in range(len(obsList)):

       # initialize pyLTR TimeSeries objects
       dBTot = pyLTR.TimeSeries()

       # insert shared values between different TimeSeries objects
       dBTot.append('datetime', 'Date & Time', '', timeRange[index0:index1+1])
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

    (path, run, 
     t0, t1, 
     obs, 
     mix, tie, lfm, 
     mix_bs_mx, tie_bs_mx, 
     smGrid, geoGrid, magGrid, 
     ignoreBinary, binaryType, 
     multiPlot, outDir) = parseArgs()

    (dBObs) = extractQuantities(
       path, run, 
       t0, t1, 
       obs, 
       mix, tie, lfm,
       mix_bs_mx, tie_bs_mx,
       smGrid, geoGrid, magGrid,
       ignoreBinary, binaryType, 
       outDir
    )


    # convert multiPlot into proper list of indices
    mp_idx = []
    for i in range(len(multiPlot)):
       mp_idx.extend(
          p.flatnonzero(p.array(['tot','ion','fac','mag'])==multiPlot[i].lower()).tolist()
       )


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
          ax = p.gca()
          xlims = ax.get_xlim()
          xticks_doy = ax.get_xticks()
          
          # extra tickmarks
          xticks_phiSM = sinterp.griddata(
            p.array(tsList[0]['doy']['data']), 
            p.array(tsList[0]['phiSM']['data']), 
            p.array(xticks_doy)
          )
          xticks_thetaSM = sinterp.griddata(
            p.array(tsList[0]['doy']['data']), 
            p.array(tsList[0]['thetaSM']['data']), 
            p.array(xticks_doy)
          )
          
          # make multi-line xtick labels
          labels = ["%7.3f\n%4.1f\n%4.1f"%(
             xticks_doy[tick], 
             xticks_phiSM[tick] * 180/p.pi, 
             xticks_thetaSM[tick] * 180/p.pi)
             for tick in range(len(xticks_doy))
          ]
          
          # last xtick label should define xticks
          labels[-2] = r'DoY' + '\n' + r'$\phi_{SM}$' + '\n' + r'$\theta_{SM}$'

          ax.set_xticks(xticks_doy)
          ax.set_xticklabels(labels)
          ax.set_xlim(xlims) # setting xticks changes xlims for some reason
          fig=p.gcf()
          for ax in fig.axes:
             ax.grid()
          p.title(tsList[0]['obs']['name'])
          #fig.tight_layout()
          p.subplots_adjust(top=.92)
          p.subplots_adjust(hspace=0)
          p.subplots_adjust(bottom=.18)
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
