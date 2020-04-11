#!/usr/bin/env python
"""
pyLTR wrapper for deltaB time series generator that derives Dst-like indices
 Execute 'simulateDst.py --help' for more info on command-line usage.
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

#squelch divide by 0 warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def parseArgs():
    """
    Returns simulated Dst, plus its magnetic-poleward and eastword components
    Execute `simulateAE.py --help` for more information.
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
                      metavar='PATH', help='Path, relative to path, to output data direcotry.')

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
    
    parser.add_option("-m", "--multiPlot", dest="multiPlot", default='*',
                      help="Comma-separated list of strings specifying which observatories\n"+
                           "to plot along with indices.\n"+
                           "(note: ts are plotted in order received; repeats are allowed")
    
    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    if options.about:
        print((sys.argv[0] + ' version ' + pyLTR.Release.version))
        print('')
        print('This script is a wrapper for deltaBTimeSeries.py, using the latter\'s')
        print('deltaB output to generate a simulated Dst-like index. The output is')
        print('Dst, SYMH, SYMD, ASYH, and ASYD, broken down into constituents driven')
        print('by magnetosphere, FAC, and ionospheric currents. This script provides')
        print('a default set of observatory inputs that correspond to the standard')
        print('4 Dst stations, but an aribtrary set of observatory coordinates may')
        print('also be specified.')
        print('')
        print('This script may generate the following files, depending on inputs:')
        print('  1a. runPath/figs/obs_deltaB_yyyy-mm-ddTHH-MM-SSZ.pkl - PKL file')
        print('      holding particular time steps\' delta B values, decomposed')
        print('      into ionospheric, FAC, and magnetospheric contributions, in')
        print('      spherical coordinates (i.e., phi,theta,rho).')
        print('  1b. runPath/figs/obs_deltaB_yyyy-mm-ddTHH-MM-SSZ.mat - MAT file')
        print('      holding particular time steps\' delta B values.')
        print('      (These files are mostly to avoid unnecessary recomputation')
        print('       when all that is desired is to replot the data, and are')
        print('       probably not very useful for subsequent analysis)')
        print('  2.  runPath/figs/dBTS_Dst[total|iono|fac|mag].png - PNG graphic,')
        print('      one for each consituent current, plus their total, showing ')
        print('      3 panels: Dst (dBH, sym and asym); dBN_dp (sym and asym);')
        print('                and dBE_dp (sym and asym)')
        print('  3a. runPath/figs/dBTS_Dst[total|iono|fac|mag].pkl - PKL files,')
        print('      one for each constituent current, and their total, holding ')
        print('      dBH_sym, dBH_asy dBN_sym, dBN_asy, dBE_sym, and dBE_asy,')
        print('      plus the N and E components for each observatory....this is')
        print('      is probably most useful for subsequent analysis that is not')
        print('      possible from the time series plots.')
        print('  3b. runPath/figs/dBTS_Dst[total|iono|fac|mag].mat - MAT files,')
        print('      one for each constituent current, and their total, holding ')
        print('      dBH_sym, dBH_asy dBN_sym, dBN_asy, dBE_sym, and dBE_asy,')
        print('      plus the N and E components for each observatory....this is')
        print('      is probably most useful for subsequent analysis that is not')
        print('      possible from the time series plots.')
        print('')
        print('To limit the time range, use the "--first" and/or "--last" flags.')
        print('To specify observatories, use the "--observatories" flag 1 or more times.')
        print('To treat observatories as geographic coordinates, use the "--geoGrid" flag.')
        print(('Execute "' + sys.argv[0] + ' --help" for details.'))
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
    
    
    # generate default list of AE observatories if options.observs is empty or None
    if len(options.observ) == 0:
       obsList = _defaultObs()
       #force geographic coordinates, just in case
       options.geoGrid = True
       
    else:
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
       obsLabels = [obs.split(',')[3] if len(obs.split(',')) > 3 
                    else 'Obs%04d'%i # gotta love Python list comprehension
                    for i,obs in enumerate(obsStrings,1)]
       obsList = [obsList[i]+[obsLabels[i],] for i in range(len(obsList))]
    
       
    
    
    geoGrid = options.geoGrid
    
    ignoreBinary = options.ignoreBinary
    
    binaryType = options.binaryType
    
    if  options.multiPlot == '*':
       multiPlot = [obs[3] for obs in obsList]
    else:
       multiPlot = options.multiPlot.split(',')
    
    outDir = options.outDir
    
    return (path, run, t0, t1, obsList, geoGrid, ignoreBinary, binaryType, multiPlot, outDir)


def _defaultObs():
   """
   Support routine that generates a default list of Dst observatories and their
   geographic coordinates
   """
   obsDst = {}
   obsDst['HON'] = {'name':'Honolulu',     'geoLatitude':21.320, 'geoLongitude':202.00}
   obsDst['KAK'] = {'name':'Kakioka',      'geoLatitude':36.232, 'geoLongitude':140.19}
   obsDst['HER'] = {'name':'Hermanus',     'geoLatitude':-34.425, 'geoLongitude':19.225}
   obsDst['SJG'] = {'name':'San Juan',     'geoLatitude':18.110, 'geoLongitude':293.85}
   
   
   # create list of lists of observatory coordinates to pass to detlaBTimeSeries.py
   obsList = [[elem[1]['geoLongitude']*p.pi/180,
               (90-elem[1]['geoLatitude'])*p.pi/180,
               6378000,
               elem[0]] 
               for elem in list(obsDst.items())]
   
   return(obsList)


def calculateIndex(path='./', run='', 
                   t0='', t1='', 
                   obsList=None, geoGrid=False, 
                   ignoreBinary=False, binaryType='pkl',
                   outDirName='figs'):
   """
   Compute deltaBs at virtual observatories, then generate an Dst-like index,
   given LFM-MIX output files in path.
   
   Computes:
     SymAsyDst - a dict of pyLTR.TimeSeries objects, corresponding to the total,
                 ionospheric current, field-aligned current, and magnetospheric
                 current constituent of dBH (mag dipole northward, scaled by mag
                 latitude), dBD (mag dipole eastward), and dBDst (horizontal, 
                 scaled by mag latitude), including both symmetric and asymmetric 
                 components.
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

   # define and geo-locate virtual observatories
   if obsList==None or len(obsList) == 0:
      obsList = _defaultObs()
      # force geoGrid=True
      geoGrid=True
      
   
   # get delta B time series for Dst stations
   print("start go!")
   dBExtract = pyLTR.Tools.deltaBTimeSeriespara.extractQuantities
   dBObs = dBExtract(path=path, run=run, 
                     t0=t0, t1=t1,
                     obsList=obsList, geoGrid=geoGrid, 
                     ignoreBinary=ignoreBinary, binaryType=binaryType,
                     outDirName=outDirName)
   print("stop now!")
   
   # convert lists of constituent vector components into NumPy arrays
   dBNorthIon = p.array([elem['dBIon']['North']['data'] for elem in dBObs]).T
   dBNorthFAC = p.array([elem['dBFAC']['North']['data'] for elem in dBObs]).T
   dBNorthMag = p.array([elem['dBMag']['North']['data'] for elem in dBObs]).T
   dBNorthTot = p.array([elem['dBTot']['North']['data'] for elem in dBObs]).T

   dBEastIon = p.array([elem['dBIon']['East']['data'] for elem in dBObs]).T
   dBEastFAC = p.array([elem['dBFAC']['East']['data'] for elem in dBObs]).T
   dBEastMag = p.array([elem['dBMag']['East']['data'] for elem in dBObs]).T
   dBEastTot = p.array([elem['dBTot']['East']['data'] for elem in dBObs]).T

   dBDownIon = p.array([elem['dBIon']['Down']['data'] for elem in dBObs]).T
   dBDownFAC = p.array([elem['dBFAC']['Down']['data'] for elem in dBObs]).T
   dBDownMag = p.array([elem['dBMag']['Down']['data'] for elem in dBObs]).T
   dBDownTot = p.array([elem['dBTot']['Down']['data'] for elem in dBObs]).T
   
   
   
   # convert deltaB vectors into local geomagnetic coordinates if they
   # are not there already
   if geoGrid:
      
      # extract geographic spherical coordinates
      phis = p.array([elem['dBTot']['phiGEO']['data'] for elem in dBObs]).T
      thetas = p.array([elem['dBTot']['thetaGEO']['data'] for elem in dBObs]).T
      rhos = p.array([elem['dBTot']['rhoGEO']['data'] for elem in dBObs]).T
      
      # extract datetime stamps
      datetimes = p.array([elem['dBTot']['datetime']['data'] for elem in dBObs]).T
      
      
      # convert Tot constituent to dipole North,East,Down coordinates
      # (must loop over time steps for correct time-dependent transforms)
      for i,ts in enumerate(datetimes):
         x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(phis[i,:], thetas[i,:], rhos[i,:], 
                                                   dBEastTot[i,:], -dBNorthTot[i,:], -dBDownTot[i,:])
         x,y,z = pyLTR.transform.GEOtoSM(x,y,z,ts[0])
         dx,dy,dz = pyLTR.transform.GEOtoSM(dx,dy,dz,ts[0])
         p_ts,t_ts,r_ts,dp_ts,dt_ts,dr_ts = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
         
         dBNorthTot[i,:] = -dt_ts
         dBEastTot[i,:] =  dp_ts
         dBDownTot[i,:] = -dr_ts
      
      
      
      # convert Ion constituent to dipole North,East,Down coordinates
      # (must loop over time steps for correct time-dependent transforms)
      for i,ts in enumerate(datetimes):
         x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(phis[i,:], thetas[i,:], rhos[i,:], 
                                                   dBEastIon[i,:], -dBNorthIon[i,:], -dBDownIon[i,:])
         x,y,z = pyLTR.transform.GEOtoSM(x,y,z,ts[0])
         dx,dy,dz = pyLTR.transform.GEOtoSM(dx,dy,dz,ts[0])
         p_ts,t_ts,r_ts,dp_ts,dt_ts,dr_ts = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
         
         dBNorthIon[i,:] = -dt_ts
         dBEastIon[i,:] =  dp_ts
         dBDownIon[i,:] = -dr_ts
      
      
      # convert FAC constituent to dipole North,East,Down coordinates
      # (must loop over time steps for correct time-dependent transforms)
      for i,ts in enumerate(datetimes):
         x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(phis[i,:], thetas[i,:], rhos[i,:], 
                                                   dBEastFAC[i,:], -dBNorthFAC[i,:], -dBDownFAC[i,:])
         x,y,z = pyLTR.transform.GEOtoSM(x,y,z,ts[0])
         dx,dy,dz = pyLTR.transform.GEOtoSM(dx,dy,dz,ts[0])
         p_ts,t_ts,r_ts,dp_ts,dt_ts,dr_ts = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
         
         dBNorthFAC[i,:] = -dt_ts
         dBEastFAC[i,:] =  dp_ts
         dBDownFAC[i,:] = -dr_ts
      
      
      # convert Mag constituent to dipole North,East,Down coordinates
      # (must loop over time steps for correct time-dependent transforms)
      for i,ts in enumerate(datetimes):
         x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(phis[i,:], thetas[i,:], rhos[i,:], 
                                                   dBEastMag[i,:], -dBNorthMag[i,:], -dBDownMag[i,:])
         x,y,z = pyLTR.transform.GEOtoSM(x,y,z,ts[0])
         dx,dy,dz = pyLTR.transform.GEOtoSM(dx,dy,dz,ts[0])
         p_ts,t_ts,r_ts,dp_ts,dt_ts,dr_ts = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
         
         # note that we didn't update phis,thetas,rhos until now because we needed
         # them to remain in geographic for previous calls to GEOtoSM
         phis[i,:] = p_ts
         thetas[i,:] = t_ts
         rhos[i,:] = r_ts
         
         dBNorthMag[i,:] = -dt_ts
         dBEastMag[i,:] =  dp_ts
         dBDownMag[i,:] = -dr_ts
      
   else:
      
      # extract SM spherical coordinates
      phis = p.array([elem['dBTot']['phiSM']['data'] for elem in dBObs]).T
      thetas = p.array([elem['dBTot']['thetaSM']['data'] for elem in dBObs]).T
      rhos = p.array([elem['dBTot']['rhoSM']['data'] for elem in dBObs]).T
      
      # everything else is already in the necessary coordinates
   
   
   
   # Now to replicate (more-or-less) SYM/ASY/Dst index algorithm(s) from 
   # Kyoto WDC (i.e., http://wdc.kugi.kyoto-u.ac.jp/aeasy/asy.pdf) 
   
   # symH is average of stations northward field normalized by the cosine 
   # of the stations' dipole latitude
   # (this differs slightly from Sugiura's technique, which has never been
   #  mathematically or physically sensible; also, the "paper" cited above
   #  describes some sort of additional normalization for the ASY indices,
   #  but is extremely confusing, so we're leaving that out for now) 
   symHTot = (dBNorthTot / p.cos(p.pi/2 - thetas)).mean(axis=1)
   
   # indices to max/min disturbance are all based on the total current system; 
   # this allows the constituents to sum up to give the total
   HMaxIdx = ((dBNorthTot / p.cos(p.pi/2 - thetas)) ).argmax(axis=1)
   HMinIdx = ((dBNorthTot / p.cos(p.pi/2 - thetas)) ).argmin(axis=1)
   
   # I do not understand how slices and index arrays are combined, so just create
   # an index array for all time steps (rows)
   allT = list(range(dBNorthTot.shape[0]))
   
   # indices from total current system
   #symHTot = (dBNorthTot / p.cos(p.pi/2 - thetas)).mean(axis=1) # redundant
   HMaxTot = ((dBNorthTot / p.cos(p.pi/2 - thetas)) )[allT,HMaxIdx]
   HMinTot = ((dBNorthTot / p.cos(p.pi/2 - thetas)) )[allT,HMinIdx]
   asyHTot = HMaxTot - HMinTot
  
   # indices from ionospheric current system
   symHIon = (dBNorthIon / p.cos(p.pi/2 - thetas)).mean(axis=1)
   HMaxIon = ((dBNorthIon / p.cos(p.pi/2 - thetas)) )[allT,HMaxIdx]
   HMinIon = ((dBNorthIon / p.cos(p.pi/2 - thetas)) )[allT,HMinIdx]
   asyHIon = HMaxIon - HMinIon
  
   # indices from field-aligned current system
   symHFAC = (dBNorthFAC / p.cos(p.pi/2 - thetas)).mean(axis=1)
   HMaxFAC = ((dBNorthFAC / p.cos(p.pi/2 - thetas)) )[allT,HMaxIdx]
   HMinFAC = ((dBNorthFAC / p.cos(p.pi/2 - thetas)) )[allT,HMinIdx]
   asyHFAC = HMaxFAC - HMinFAC
  
   # indices from magnetospheric current system
   symHMag = (dBNorthMag / p.cos(p.pi/2 - thetas)).mean(axis=1)
   HMaxMag = ((dBNorthMag / p.cos(p.pi/2 - thetas)) )[allT,HMaxIdx]
   HMinMag = ((dBNorthMag / p.cos(p.pi/2 - thetas)) )[allT,HMinIdx]
   asyHMag = HMaxMag - HMinMag
   
   
   
   # symD is average of stations eastward field, but NOT normalized by the 
   # cosine of dipole latitude
   # (the "paper" cited above describes some sort of normalization for the ASY
   #  indices, but is extremely confusing, so we're leaving that out for now) 
   symDTot = dBEastTot.mean(axis=1)
   
   # indices to max/min disturbance are all based on the total current system; 
   # this allows the constituents to sum up to give the total
   DMaxIdx = (dBEastTot ).argmax(axis=1)
   DMinIdx = (dBEastTot ).argmin(axis=1)
   
   # I do not understand how slices and index arrays are combined, so just create
   # an index array for all time steps (rows)
   allT = list(range(dBNorthTot.shape[0]))
   
   # indices from total current system
   #symDTot = dBEastTot.mean(axis=1) # redundant
   DMaxTot = (dBEastTot )[allT,DMaxIdx]
   DMinTot = (dBEastTot )[allT,DMinIdx]
   asyDTot = DMaxTot - DMinTot
  
   # indices from ionospheric current system
   symDIon = dBEastIon.mean(axis=1)
   DMaxIon = (dBEastIon )[allT,DMaxIdx]
   DMinIon = (dBEastIon )[allT,DMinIdx]
   asyDIon = DMaxIon - DMinIon
  
   # indices from field-aligned current system
   symDFAC = dBEastFAC.mean(axis=1)
   DMaxFAC = (dBEastFAC )[allT,DMaxIdx]
   DMinFAC = (dBEastFAC )[allT,DMinIdx]
   asyDFAC = DMaxFAC - DMinFAC
  
   # indices from magnetospheric current system
   symDMag = dBEastMag.mean(axis=1)
   DMaxMag = (dBEastMag )[allT,DMaxIdx]
   DMinMag = (dBEastMag )[allT,DMinIdx]
   asyDMag = DMaxMag - DMinMag
   
   """
   FIXME
   
   NEED TO GET DST SIGN RIGHT, JUST LIKE WITH AE...IN OTHER WORDS, ADD BASELINE
   PRIOR TO VECTOR SUMMING, THEN SUBTRACT BASELINE TO ZERO OUT
   
   FIXME
   """
   
   # Dst is average of stations' horizontal field normalized by the cosine 
   # the stations' dipole latitude
   # (this differs slightly from Sugiura's technique, which has never been
   #  mathematically or physically sensible; also, the "paper" cited above
   #  describes some sort of additional normalization for the ASY indices,
   #  but is extremely confusing, so we're leaving that out for now) 
   #dBHorizTot = ( (p.sqrt((dBNorthTot+5e4)**2+dBEastTot**2) - 
   #               (5e4*p.cos(p.arctan2(dBEastTot,(dBNorthTot+5e4)) ) )) / 
   #              p.cos(p.pi/2 - thetas))
   #dBHorizIon = ( (p.sqrt((dBNorthIon+5e4)**2+dBEastIon**2) - 
   #               (5e4*p.cos(p.arctan2(dBEastIon,(dBNorthIon+5e4)) ) )) / 
   #              p.cos(p.pi/2 - thetas))
   #dBHorizFAC = ( (p.sqrt((dBNorthFAC+5e4)**2+dBEastFAC**2) - 
   #               (5e4*p.cos(p.arctan2(dBEastFAC,(dBNorthFAC+5e4)) ) )) / 
   #              p.cos(p.pi/2 - thetas))
   #dBHorizMag = ( (p.sqrt((dBNorthMag+5e4)**2+dBEastMag**2) - 
   #               (5e4*p.cos(p.arctan2(dBEastMag,(dBNorthMag+5e4)) ) )) / 
   #              p.cos(p.pi/2 - thetas))
   
   
   
   # calculate main field to add to deltaB before calculating horizontal field,
   # then subtract quadratic sum of main field components, assuming this is the
   # equivalent to removing a quiet baseline...I know this seems like a strange
   # thing to do, but real-world indices have positive and negative values,
   # which would be impossible if vector-component baselines were removed
   # *before* adding up the disturbance vectors; so, we add plausible vector
   # component baselines to our disturbance vectors to emulate how real-world
   # observations are collected, determine the horizontal field, then remove
   # the horizontal baseline.
   dipoleMF = pyLTR.Tools.deltaBTimeSeriespara._dipoleMF
   mfObs = [dipoleMF(obs['dBTot'], geoGrid) for obs in dBObs]
   BNorth = p.array([obs['North']['data'] for obs in mfObs]).T
   BEast = p.array([obs['East']['data'] for obs in mfObs]).T
   BDown = p.array([obs['Down']['data'] for obs in mfObs]).T
      
   dBHorizIon = p.sqrt((dBNorthIon + BNorth)**2 + (dBEastIon + BEast)**2) # add main field
   dBHorizIon = dBHorizIon - p.sqrt(BNorth**2 + BEast**2) # remove horiz baseline
   dBHorizFAC = p.sqrt((dBNorthFAC + BNorth)**2 + (dBEastFAC + BEast)**2) # add main field
   dBHorizFAC = dBHorizFAC - p.sqrt(BNorth**2 + BEast**2) # remove horiz baseline
   dBHorizMag = p.sqrt((dBNorthMag + BNorth)**2 + (dBEastMag + BEast)**2) # add main field
   dBHorizMag = dBHorizMag - p.sqrt(BNorth**2 + BEast**2) # remove horiz baseline
   dBHorizTot = p.sqrt((dBNorthTot + BNorth)**2 + (dBEastTot + BEast)**2) # add main field
   dBHorizTot = dBHorizTot - p.sqrt(BNorth**2 + BEast**2) # remove horiz baseline
   
   
                 
   
   # indices to max/min disturbance are all based on the total current system; 
   # this allows the constituents to sum up to give the total
   DstMaxIdx = dBHorizTot.argmax(axis=1)
   DstMinIdx = dBHorizTot.argmin(axis=1)
   
   
   # I do not understand how slices and index arrays are combined, so just create
   # an index array for all time steps (rows)
   allT = list(range(dBNorthTot.shape[0]))
   
   # indices from total current system
   symDstTot = dBHorizTot.mean(axis=1)
   DstMaxTot = dBHorizTot[allT,DstMaxIdx]
   DstMinTot = dBHorizTot[allT,DstMinIdx]
   asyDstTot = DstMaxTot - DstMinTot
   
   # indices from ionospheric current system
   symDstIon = dBHorizIon.mean(axis=1)
   DstMaxIon = dBHorizIon[allT,DstMaxIdx]
   DstMinIon = dBHorizIon[allT,DstMinIdx]
   asyDstIon = DstMaxIon - DstMinIon
   
   # indices from field-aligned current system
   symDstFAC = dBHorizFAC.mean(axis=1)
   DstMaxFAC = dBHorizFAC[allT,DstMaxIdx]
   DstMinFAC = dBHorizFAC[allT,DstMinIdx]
   asyDstFAC = DstMaxFAC - DstMinFAC
   
   # indices from magnetospheric current system
   symDstMag = dBHorizMag.mean(axis=1)
   DstMaxMag = dBHorizMag[allT,DstMaxIdx]
   DstMinMag = dBHorizMag[allT,DstMinIdx]
   asyDstMag = DstMaxMag - DstMinMag
   
   
   
   
   
   
   
   # create time series object to hold sum total of constituents
   dBTot = pyLTR.TimeSeries()
   dBTot.append('datetime', 'Date & Time', '', dBObs[0]['dBTot']['datetime']['data'])
   dBTot.append('doy', 'Day of Year', 'days', dBObs[0]['dBTot']['doy']['data'])
   dBTot.append('SYMH', r'$\Delta B_{average}$', 'nT', symHTot.tolist())
   dBTot.append('ASYH', r'$\Delta B_{envelope}$', 'nT', asyHTot.tolist())
   dBTot.append('MaxH', r'$\Delta B_{upper}$', 'nT', HMaxTot.tolist())
   dBTot.append('MinH', r'$\Delta B_{lower}$', 'nT', HMinTot.tolist())
   dBTot.append('dBH', r'$\Delta B$', 'nT', (dBNorthTot / p.cos(p.pi/2 - thetas)).tolist())
   
   dBTot.append('SYMD', r'$\Delta B_{average}$', 'nT', symDTot.tolist())
   dBTot.append('ASYD', r'$\Delta B_{envelope}$', 'nT', asyDTot.tolist())
   dBTot.append('MaxD', r'$\Delta B_{upper}$', 'nT', DMaxTot.tolist())
   dBTot.append('MinD', r'$\Delta B_{lower}$', 'nT', DMinTot.tolist())
   dBTot.append('dBD', r'$\Delta B$', 'nT', dBEastTot.tolist())
   
   dBTot.append('SYMDst', r'$\Delta B_{average}$', 'nT', symDstTot.tolist())
   dBTot.append('ASYDst', r'$\Delta B_{envelope}$', 'nT', asyDstTot.tolist())
   dBTot.append('MaxDst', r'$\Delta B_{upper}$', 'nT', DstMaxTot.tolist())
   dBTot.append('MinDst', r'$\Delta B_{lower}$', 'nT', DstMinTot.tolist())
   #dBTot.append('dBDst', r'$\Delta B$', 'nT', p.sqrt(dBNorthTot**2+dBEastTot**2) / p.cos(p.pi/2 - thetas).tolist())
   dBTot.append('dBDst', r'$\Delta B$', 'nT', dBHorizTot.tolist())
   
   
   
   # create time series object to hold ionospheric current constituent
   dBIon = pyLTR.TimeSeries()
   dBIon.append('datetime', 'Date & Time', '', dBObs[0]['dBIon']['datetime']['data'])
   dBIon.append('doy', 'Day of Year', 'days', dBObs[0]['dBIon']['doy']['data'])
   dBIon.append('SYMH', r'$\Delta B_{average}$', 'nT', symHIon.tolist())
   dBIon.append('ASYH', r'$\Delta B_{envelope}$', 'nT', asyHIon.tolist())
   dBIon.append('MaxH', r'$\Delta B_{upper}$', 'nT', HMaxIon.tolist())
   dBIon.append('MinH', r'$\Delta B_{lower}$', 'nT', HMinIon.tolist())
   dBIon.append('dBH', r'$\Delta B$', 'nT', (dBNorthIon / p.cos(p.pi/2 - thetas)).tolist())
   
   dBIon.append('SYMD', r'$\Delta B_{average}$', 'nT', symDIon.tolist())
   dBIon.append('ASYD', r'$\Delta B_{envelope}$', 'nT', asyDIon.tolist())
   dBIon.append('MaxD', r'$\Delta B_{upper}$', 'nT', DMaxIon.tolist())
   dBIon.append('MinD', r'$\Delta B_{lower}$', 'nT', DMinIon.tolist())
   dBIon.append('dBD', r'$\Delta B$', 'nT', dBEastIon.tolist())
   
   dBIon.append('SYMDst', r'$\Delta B_{average}$', 'nT', symDstIon.tolist())
   dBIon.append('ASYDst', r'$\Delta B_{envelope}$', 'nT', asyDstIon.tolist())
   dBIon.append('MaxDst', r'$\Delta B_{upper}$', 'nT', DstMaxIon.tolist())
   dBIon.append('MinDst', r'$\Delta B_{lower}$', 'nT', DstMinIon.tolist())
   #dBIon.append('dBDst', r'$\Delta B$', 'nT', p.sqrt(dBNorthIon**2+dBEastIon**2) / p.cos(p.pi/2 - thetas).tolist())
   dBIon.append('dBDst', r'$\Delta B$', 'nT', dBHorizIon.tolist())
   
   
   
   # create time series object to hold field-aligned current constituent
   dBFAC = pyLTR.TimeSeries()
   dBFAC.append('datetime', 'Date & Time', '', dBObs[0]['dBFAC']['datetime']['data'])
   dBFAC.append('doy', 'Day of Year', 'days', dBObs[0]['dBFAC']['doy']['data'])
   dBFAC.append('SYMH', r'$\Delta B_{average}$', 'nT', symHFAC.tolist())
   dBFAC.append('ASYH', r'$\Delta B_{envelope}$', 'nT', asyHFAC.tolist())
   dBFAC.append('MaxH', r'$\Delta B_{upper}$', 'nT', HMaxFAC.tolist())
   dBFAC.append('MinH', r'$\Delta B_{lower}$', 'nT', HMinFAC.tolist())
   dBFAC.append('dBH', r'$\Delta B$', 'nT', (dBNorthFAC / p.cos(p.pi/2 - thetas)).tolist())
   
   dBFAC.append('SYMD', r'$\Delta B_{average}$', 'nT', symDFAC.tolist())
   dBFAC.append('ASYD', r'$\Delta B_{envelope}$', 'nT', asyDFAC.tolist())
   dBFAC.append('MaxD', r'$\Delta B_{upper}$', 'nT', DMaxFAC.tolist())
   dBFAC.append('MinD', r'$\Delta B_{lower}$', 'nT', DMinFAC.tolist())
   dBFAC.append('dBD', r'$\Delta B$', 'nT', dBEastFAC.tolist())
   
   dBFAC.append('SYMDst', r'$\Delta B_{average}$', 'nT', symDstFAC.tolist())
   dBFAC.append('ASYDst', r'$\Delta B_{envelope}$', 'nT', asyDstFAC.tolist())
   dBFAC.append('MaxDst', r'$\Delta B_{upper}$', 'nT', DstMaxFAC.tolist())
   dBFAC.append('MinDst', r'$\Delta B_{lower}$', 'nT', DstMinFAC.tolist())
   #dBFAC.append('dBDst', r'$\Delta B$', 'nT', p.sqrt(dBNorthFAC**2+dBEastFAC**2) / p.cos(p.pi/2 - thetas).tolist())
   dBFAC.append('dBDst', r'$\Delta B$', 'nT', dBHorizFAC.tolist())
   
   
   
   # create time series object to hold magnetospheric constituent
   dBMag = pyLTR.TimeSeries()
   dBMag.append('datetime', 'Date & Time', '', dBObs[0]['dBMag']['datetime']['data'])
   dBMag.append('doy', 'Day of Year', 'days', dBObs[0]['dBMag']['doy']['data'])
   dBMag.append('SYMH', r'$\Delta B_{average}$', 'nT', symHMag.tolist())
   dBMag.append('ASYH', r'$\Delta B_{envelope}$', 'nT', asyHMag.tolist())
   dBMag.append('MaxH', r'$\Delta B_{upper}$', 'nT', HMaxMag.tolist())
   dBMag.append('MinH', r'$\Delta B_{lower}$', 'nT', HMinMag.tolist())
   dBMag.append('dBH', r'$\Delta B$', 'nT', (dBNorthMag / p.cos(p.pi/2 - thetas)).tolist())
   
   dBMag.append('SYMD', r'$\Delta B_{average}$', 'nT', symDMag.tolist())
   dBMag.append('ASYD', r'$\Delta B_{envelope}$', 'nT', asyDMag.tolist())
   dBMag.append('MaxD', r'$\Delta B_{upper}$', 'nT', DMaxMag.tolist())
   dBMag.append('MinD', r'$\Delta B_{lower}$', 'nT', DMinMag.tolist())
   dBMag.append('dBD', r'$\Delta B$', 'nT', dBEastMag.tolist())
   
   dBMag.append('SYMDst', r'$\Delta B_{average}$', 'nT', symDstMag.tolist())
   dBMag.append('ASYDst', r'$\Delta B_{envelope}$', 'nT', asyDstMag.tolist())
   dBMag.append('MaxDst', r'$\Delta B_{upper}$', 'nT', DstMaxMag.tolist())
   dBMag.append('MinDst', r'$\Delta B_{lower}$', 'nT', DstMinMag.tolist())
   #dBMag.append('dBDst', r'$\Delta B$', 'nT', p.sqrt(dBNorthMag**2+dBEastMag**2) / p.cos(p.pi/2 - thetas).tolist())
   dBMag.append('dBDst', r'$\Delta B$', 'nT', dBHorizMag.tolist())
      
   
   
   SymAsyDst = {'dBTot':dBTot, 'dBIon':dBIon, 'dBFAC':dBFAC, 'dBMag':dBMag}
   
   return (SymAsyDst)


if __name__ == '__main__':

   (path, run, t0, t1, obs, geoGrid, ignoreBinary, binaryType, multiPlot, outDir) = parseArgs()
   (dBall) = calculateIndex(path, run, 
                            t0, t1, obs, 
                            geoGrid, ignoreBinary, 
                            binaryType, outDir)
   
   # convert multiPlot into proper list of indices
   mp_idx = []
   for i in range(len(multiPlot)):
      #mp_idx.extend(n.nonzero(n.array([o[3].lower() for o in obs])==multiPlot[i].lower()).tolist() )
      mp_idx.extend(e for e, in n.nonzero(n.array([o[3].lower() for o in obs])==multiPlot[i].lower()) )
   
   #
   # --- Make plots of everything
   #
   
   # Make sure the output directory exisits if not make it
   dirname = os.path.join(path, outDir)
   if not os.path.exists(dirname):
       os.makedirs(dirname)
       
   # Associate output filename bases with expected dBall keys
   fnames = {'dBTot':'Dsttotal', 'dBIon':'Dstiono', 'dBFAC':'Dstfac', 'dBMag':'Dstmag'}
   
   # Associate plot title strings with expected dBall keys
   ptitles = {'dBTot':'All Currents', 'dBIon':'Ionospheric Currents', 
              'dBFAC':'Field-aligned Currents', 'dBMag':'Magnetospheric Currents'}
   
   # plot/save file for each constituent
   for key,item in list(dBall.items()):
      
      if not (multiPlot[0].lower() == 'none' and len(multiPlot) == 1):
         
         # set filename for png files
         filename = os.path.join(dirname, 'dBTS_'+fnames[key]+'.png')
         
         # define figure dimensions
         p.figure(1,figsize=(16,18))
         
         # define list of unique colors, one for each observatory, from current colormap
         # (this strange calling sequence seems to be required by mpl colormaps)
         cmap = p.get_cmap(lut=len(obs))
         colors = cmap(list(range(len(obs)))).tolist()
         
         # plot 3 panels (i.e., H,D,Dst), each with time series specified by multiPlot
         
         p.subplot(311)
         # plot dBh time series specified by multiPlot
         if mp_idx:
            pyLTR.Graphics.TimeSeries.BasicPlot(item, 'datetime', 'dBH',
                                               color=[colors[mp] for mp in mp_idx])
            legendLabels=[[o[3] for o in obs][mp] for mp in mp_idx]
            p.legend(legendLabels, loc='upper left')
         # overplot SYM-H, HMin, HMax, and ASY-H on panel 1
         p.plot(dBall[key]['datetime']['data'], dBall[key]['MinH']['data'], '--k', linewidth=2,dashes=(6,4))
         p.plot(dBall[key]['datetime']['data'], dBall[key]['MaxH']['data'], '--k', linewidth=2,dashes=(6,4))
         p.plot(dBall[key]['datetime']['data'], dBall[key]['SYMH']['data'], '-k', linewidth=3)
         #p.plot(dBall[key]['datetime']['data'], dBall[key]['ASYH']['data'], '-k', linewidth=3)
         p.title('SYM-H and ASY-H ('+ptitles[key]+')')
         p.gca().set_xticklabels('')
         p.xlabel('')
         
         p.subplot(312)
         # plot dBh time series specified by multiPlot
         if mp_idx:
            pyLTR.Graphics.TimeSeries.BasicPlot(item, 'datetime', 'dBD',
                                               color=[colors[mp] for mp in mp_idx])
            legendLabels=[[o[3] for o in obs][mp] for mp in mp_idx]
            p.legend(legendLabels, loc='upper left')
         # overplot SYM-D, MinD, MaxD, and ASY-D on panel 2
         p.plot(dBall[key]['datetime']['data'], dBall[key]['MinD']['data'], '--k', linewidth=2,dashes=(6,4))
         p.plot(dBall[key]['datetime']['data'], dBall[key]['MaxD']['data'], '--k', linewidth=2,dashes=(6,4))
         p.plot(dBall[key]['datetime']['data'], dBall[key]['SYMD']['data'], '-k', linewidth=3)
         #p.plot(dBall[key]['datetime']['data'], dBall[key]['ASYD']['data'], '-k', linewidth=3)
         p.title('SYM-D and ASY-D ('+ptitles[key]+')')
         p.gca().set_xticklabels('')
         p.xlabel('')
         
         p.subplot(313)
         # plot dBh time series specified by multiPlot
         if mp_idx:
            pyLTR.Graphics.TimeSeries.BasicPlot(item, 'datetime', 'dBDst',
                                               color=[colors[mp] for mp in mp_idx])
            legendLabels=[[o[3] for o in obs][mp] for mp in mp_idx]
            p.legend(legendLabels, loc='upper left')
         # overplot SYM-Dst, MinDst, MaxDst, and ASY-Dst on panel 3
         p.plot(dBall[key]['datetime']['data'], dBall[key]['MinDst']['data'], '--k', linewidth=2,dashes=(6,4))
         p.plot(dBall[key]['datetime']['data'], dBall[key]['MaxDst']['data'], '--k', linewidth=2,dashes=(6,4))
         p.plot(dBall[key]['datetime']['data'], dBall[key]['SYMDst']['data'], '-k', linewidth=3)
         #p.plot(dBall[key]['datetime']['data'], dBall[key]['ASYDst']['data'], '-k', linewidth=3)
         p.title('SYM-Dst and ASY-Dst ('+ptitles[key]+')')
         
         
         #p.subplots_adjust(hspace=0)
         #p.subplots_adjust(bottom=.16)
         p.savefig(filename, dpi=150)
         p.clf()
         
         
         
         
         # save observatory time series to a binary file
         # note: this is the full time series, not a single time-step like what
         #       is saved in deltaBTimeSeries.extractQuantities()...it is not used
         #       to avoid re-computations in any way, but is intended to facilitate
         #       subsequent time-series analysis, whether in Python, or some other
         #       analysis tool.
         if binaryType.lower() == 'pkl' or binaryType.lower() == '.pkl' or binaryType.lower() == 'pickle':
            # --- Dump a pickle!
            pklFilename = os.path.join(dirname, 'dBTS_'+fnames[key]+'.pkl')
            try:
              fh = open(pklFilename, 'wb')
              pickle.dump(dBall[key], fh, protocol=2)
              fh.close()
            except:
              print(('Warning: Unable to write binary file: '+pklFilename))
         elif binaryType.lower() == 'mat' or binaryType.lower() == '.mat' or binaryType.lower() == 'matlab':
            # --- Dump a .mat file!
            matFilename = os.path.join(dirname, 'dBTS_'+fnames[key]+'.mat')
            
            # savemat() cannot handle datetimes, so convert datetimes in output to
            # ML-compatible "days-since-epoch (31December0000)"
            dBall[key]['datetime']['data'] = p.date2num(dBall[key]['datetime']['data'])
            
            try:
              sio.savemat(matFilename, dBall[key])
            except:
              print(('Warning: Unable to write binary file: '+matFilename))
         elif binaryType.lower() == 'none':
            pass
         else:
            print(('Unrecognized binary type '+binaryType+' requested'))
            raise Exception
      
   else:
      pass

