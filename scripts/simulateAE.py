#!/usr/bin/env python
"""
pyLTR wrapper for deltaB time series generator that derives AE-like indices.
 Execute 'simulateAE.py --help' for more info on command-line usage.
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
    Returns simulated AU, AL, and AE time series.
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
                           "to plot along with AU, AL, and AE.\n"+
                           "(note: ts are plotted in order received; repeats are allowed")
    
    parser.add_option('-k', '--cpu', dest='ncpus',
                        default=1, action='store',
                        help='Number of cpus for the parallelism. Note default is 8')
    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    if options.about:
        print((sys.argv[0] + ' version ' + pyLTR.Release.version))
        print('')
        print('This script is a wrapper for deltaBTimeSeries.py, using the latter\'s')
        print('deltaB output to generate a simulated AE index. The output of this')
        print('script is AU, AL, and AE, broken down into constituents driven by')
        print('magnetosphere, FAC, and ionospheric currents. This script provides')
        print('a default set of observatory inputs that correspond to the standard')
        print('12 AE stations, but an aribtrary set of observatory coordinates may')
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
        print('  2.  runPath/figs/dBTS_AE[total|iono|fac|mag].png - PNG graphic,')
        print('      one for each consituent current, plus their total, showing ')
        print('      delta B_h constituents, and the AU, AL, and AE indices.')
        print('  3a. runPath/figs/dBTS_AE[total|iono|fac|mag].pkl - PKL files,')
        print('      one for each constituent current, and their total, holding ')
        print('      delta B_h constituents, and the AU, AL, and AE indices.')
        print('      this is probably most useful for subsequent analysis that')
        print('      is not possible from the time series plots.')
        print('  3b. runPath/figs/dBTS_AE[total|iono|fac|mag].mat - MAT files,')
        print('      one for each constituent current, and their total, holding ')
        print('      delta B_h constituents, and the AU, AL, and AE indices.')
        print('      this is probably most useful for subsequent analysis that')
        print('      is not possible from the time series plots.')
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
    
    return (path, run, t0, t1, obsList, geoGrid, ignoreBinary, binaryType, multiPlot, options.ncpus, outDir)


def _defaultObs():
   """
   Support routine that generates a default list of AE observatories and their
   geographic coordinates
   """
   obsAE = {}
   obsAE['ABK'] = {'name':'Abisko',              'geoLatitude':68.36, 'geoLongitude':18.82}
   obsAE['DIK'] = {'name':'Dixon Island',        'geoLatitude':73.55, 'geoLongitude':80.57}
   obsAE['CCS'] = {'name':'Cape Chelyuskin',     'geoLatitude':77.72, 'geoLongitude':104.28}

   obsAE['TIK'] = {'name':'Tixie Bay',           'geoLatitude':71.58, 'geoLongitude':129.00}
   obsAE['CWE'] = {'name':'Cape Wellen',         'geoLatitude':66.17, 'geoLongitude':190.17}
   obsAE['BRW'] = {'name':'Barrow',              'geoLatitude':71.30, 'geoLongitude':203.25}

   obsAE['CMO'] = {'name':'College',             'geoLatitude':64.87, 'geoLongitude':212.17}
   obsAE['YKC'] = {'name':'Yellowknife',         'geoLatitude':62.40, 'geoLongitude':245.60}
   obsAE['FCC'] = {'name':'Fort Churchill',      'geoLatitude':58.80, 'geoLongitude':265.90}

   obsAE['PBQ'] = {'name':'Poste-de-la-Baleine', 'geoLatitude':55.27, 'geoLongitude':282.22}
   obsAE['NAQ'] = {'name':'Narsarsuaq',          'geoLatitude':61.20, 'geoLongitude':314.16}
   obsAE['LRV'] = {'name':'Leirvogur',           'geoLatitude':64.18, 'geoLongitude':338.30}
   
   
   # create list of lists of observatory coordinates to pass to detlaBTimeSeries.py
   obsList = [[elem[1]['geoLongitude']*p.pi/180,
               (90-elem[1]['geoLatitude'])*p.pi/180,
               6378000,
               elem[0]] 
               for elem in list(obsAE.items())]
   
   return(obsList)



def calculateIndex(path='./', run='', 
                   t0='', t1='', 
                   obsList=None, geoGrid=False, 
                   ignoreBinary=False, binaryType='pkl',
                   outDirName='figs',ncpus=1):
   """
   Compute deltaBs at virtual observatories, then generate an AE-like index,
   given LFM-MIX output files in path.
   
   Computes:
     AEdB      - a dict of pyLTR.TimeSeries objects, corresponding to the total,
                 ionospheric current, field-aligned current, and magnetospheric
                 current constituent of the AE, AU, AL, and the delta B_h's
                 used to derived these indices.
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
      
   print(('obsList: ',obsList))
   # get delta B time series for AE stations
   dBExtract = pyLTR.Tools.deltaBTimeSeriespara.extractQuantities
   dBObs = dBExtract(path=path, run=run, 
                     t0=t0, t1=t1,
                     obsList=obsList, geoGrid=geoGrid, 
                     ignoreBinary=ignoreBinary, binaryType=binaryType,
                     outDirName=outDirName,ncpus=ncpus)
   
   
   # convert lists of constituent vector components into NumPy arrays
   dBNorthIon = p.array([obs['dBIon']['North']['data'] for obs in dBObs]).T
   dBNorthFAC = p.array([obs['dBFAC']['North']['data'] for obs in dBObs]).T
   dBNorthMag = p.array([obs['dBMag']['North']['data'] for obs in dBObs]).T
   dBNorthTot = p.array([obs['dBTot']['North']['data'] for obs in dBObs]).T

   dBEastIon = p.array([obs['dBIon']['East']['data'] for obs in dBObs]).T
   dBEastFAC = p.array([obs['dBFAC']['East']['data'] for obs in dBObs]).T
   dBEastMag = p.array([obs['dBMag']['East']['data'] for obs in dBObs]).T
   dBEastTot = p.array([obs['dBTot']['East']['data'] for obs in dBObs]).T

   dBDownIon = p.array([obs['dBIon']['Down']['data'] for obs in dBObs]).T
   dBDownFAC = p.array([obs['dBFAC']['Down']['data'] for obs in dBObs]).T
   dBDownMag = p.array([obs['dBMag']['Down']['data'] for obs in dBObs]).T
   dBDownTot = p.array([obs['dBTot']['Down']['data'] for obs in dBObs]).T
   
   
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
   
   

   # compute simulated AU and AL based on the total field, then apply this to
   # mag, ion, and fac constituents to get constituent AEs that sum to the total. 
   # NOTE: this is NOT the same as calculating envelopes for each constituent
   AUidx = dBHorizTot.argmax(axis=1)
   ALidx = dBHorizTot.argmin(axis=1)

   # I do not understand how slices and index arrays are combined, so just create
   # an index array for all time steps (rows)
   allT = list(range(dBHorizTot.shape[0]))

   AUTot = dBHorizTot[allT,AUidx]
   ALTot = dBHorizTot[allT,ALidx]
   AETot = AUTot - ALTot

   AUIon = dBHorizIon[allT,AUidx]
   ALIon = dBHorizIon[allT,ALidx]
   AEIon = AUIon - ALIon

   AUFAC = dBHorizFAC[allT,AUidx]
   ALFAC = dBHorizFAC[allT,ALidx]
   AEFAC = AUFAC - ALFAC

   AUMag = dBHorizMag[allT,AUidx]
   ALMag = dBHorizMag[allT,ALidx]
   AEMag = AUMag - ALMag
   
   
   
   # create time series object to hold sum total of constituents
   dBTot = pyLTR.TimeSeries()
   dBTot.append('datetime', 'Date & Time', '', dBObs[0]['dBTot']['datetime']['data'])
   dBTot.append('doy', 'Day of Year', 'days', dBObs[0]['dBTot']['doy']['data'])
   dBTot.append('AE', r'$\Delta B_{envelope}$', 'nT', AETot.tolist())
   dBTot.append('AU', r'$\Delta B_{upper}$', 'nT', AUTot.tolist())
   dBTot.append('AL', r'$\Delta B_{lower}$', 'nT', ALTot.tolist())
   #dBTot.append('dBh', r'$\Delta B$', 'nT', dBHorizTot.tolist())
   dBTot.append('dBh', [r'$\Delta B_{%s}'%o[3] for o in obsList], 'nT', dBHorizTot.tolist())
      
   # create time series object to hold ionospheric current constituent
   dBIon = pyLTR.TimeSeries()
   dBIon.append('datetime', 'Date & Time', '', dBObs[0]['dBIon']['datetime']['data'])
   dBIon.append('doy', 'Day of Year', 'days', dBObs[0]['dBIon']['doy']['data'])
   dBIon.append('AE', r'$\Delta B_{envelope}$', 'nT', AEIon.tolist())
   dBIon.append('AU', r'$\Delta B_{upper}$', 'nT', AUIon.tolist())
   dBIon.append('AL', r'$\Delta B_{lower}$', 'nT', ALIon.tolist())
   #dBIon.append('dBh', r'$\Delta B$', 'nT', dBHorizIon.tolist())
   dBIon.append('dBh', [r'$\Delta B_{%s}$'%o[3] for o in obsList], 'nT', dBHorizIon.tolist())
   
   # create time series object to hold field aligned current constituent
   dBFAC = pyLTR.TimeSeries()
   dBFAC.append('datetime', 'Date & Time', '', dBObs[0]['dBFAC']['datetime']['data'])
   dBFAC.append('doy', 'Day of Year', 'days', dBObs[0]['dBFAC']['doy']['data'])
   dBFAC.append('AE', r'$\Delta B_{envelope}$', 'nT', AEFAC.tolist())
   dBFAC.append('AU', r'$\Delta B_{upper}$', 'nT', AUFAC.tolist())
   dBFAC.append('AL', r'$\Delta B_{lower}$', 'nT', ALFAC.tolist())
   #dBFAC.append('dBh', r'$\Delta B$', 'nT', dBHorizFAC.tolist())
   dBFAC.append('dBh', [r'$\Delta B_{%s}$'%o[3] for o in obsList], 'nT', dBHorizFAC.tolist())
   
   # create time series object to hold magnetospheric current constituent
   dBMag = pyLTR.TimeSeries()
   dBMag.append('datetime', 'Date & Time', '', dBObs[0]['dBMag']['datetime']['data'])
   dBMag.append('doy', 'Day of Year', 'days', dBObs[0]['dBMag']['doy']['data'])
   dBMag.append('AE', r'$\Delta B_{envelope}$', 'nT', AEMag.tolist())
   dBMag.append('AU', r'$\Delta B_{upper}$', 'nT', AUMag.tolist())
   dBMag.append('AL', r'$\Delta B_{lower}$', 'nT', ALMag.tolist())
   #dBMag.append('dBh', r'$\Delta B$', 'nT', dBHorizMag.tolist())
   dBMag.append('dBh', [r'$\Delta B_{%s}$'%o[3] for o in obsList], 'nT', dBHorizMag.tolist())
   
   
   
   # output is a dictionary of time series objects
   AEdB = {'dBTot':dBTot, 'dBIon':dBIon, 'dBFAC':dBFAC, 'dBMag':dBMag}
   return (AEdB)


if __name__ == '__main__':

   (path, run, t0, t1, obs, geoGrid, ignoreBinary, binaryType, multiPlot, ncpus, outDir) = parseArgs()
   (dBall) = calculateIndex(path, run, 
                            t0, t1, obs, 
                            geoGrid, ignoreBinary, 
                            binaryType, outDir, ncpus)
   
   # convert multiPlot into proper list of indices
   mp_idx = []
   for i in range(len(multiPlot)):
      #mp_idx.extend(p.find(p.array([o[3].lower() for o in obs])==multiPlot[i].lower()).tolist() )
      mp_idx.extend(e for e, in n.nonzero(n.array([o[3].lower() for o in obs])==multiPlot[i].lower()) )
   
   #
   # --- Make plots of everything
   #
   
   # Make sure the output directory exisits if not make it
   dirname = os.path.join(path, outDir)
   if not os.path.exists(dirname):
       os.makedirs(dirname)
   
   # Associate output filename bases with expected dBall keys
   fnames = {'dBTot':'AEtotal', 'dBIon':'AEiono', 'dBFAC':'AEfac', 'dBMag':'AEmag'}
   
   # Associate plot title strings with expected dBall keys
   ptitles = {'dBTot':'All Currents', 'dBIon':'Ionospheric Currents', 
              'dBFAC':'Field-aligned Currents', 'dBMag':'Magnetospheric Currents'}
   
   # plot/save file for each constituent
   for key,item in list(dBall.items()):
      
      if not (multiPlot[0].lower() == 'none' and len(multiPlot) == 1):
         
         # set filename for png files
         filename = os.path.join(dirname, 'dBTS_'+fnames[key]+'.png')
         
         # define figure dimensions
         p.figure(1,figsize=(16,6))
         
         # define list of unique colors, one for each observatory, from current colormap
         # (this strange calling sequence seems to be required by mpl colormaps)
         cmap = p.get_cmap(lut=len(obs))
         colors = cmap(list(range(len(obs)))).tolist()
         
         # plot dBh time series specified by multiPlot
         if mp_idx:
            
            pyLTR.Graphics.TimeSeries.BasicPlot(item, 'datetime', 'dBh',
                                                 color=[colors[mp] for mp in mp_idx])
            legendLabels=[[o[3] for o in obs][mp] for mp in mp_idx]
            p.legend(legendLabels, loc='upper left')
         
         # overplot AU, AL, and AE
         p.plot(item['datetime']['data'], item['AU']['data'], '--k', linewidth=2,dashes=(6,4))
         p.plot(item['datetime']['data'], item['AL']['data'], '--k', linewidth=2,dashes=(6,4))
         p.plot(item['datetime']['data'], item['AE']['data'], '-k', linewidth=3)
         
         p.title('AE, AU, and AL ('+ptitles[key]+')')
         p.subplots_adjust(hspace=0)
         p.subplots_adjust(bottom=.16)
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
              pickle.dump(item, fh, protocol=2)
              fh.close()
            except:
              print(('Warning: Unable to write binary file: '+pklFilename))
         elif binaryType.lower() == 'mat' or binaryType.lower() == '.mat' or binaryType.lower() == 'matlab':
            # --- Dump a .mat file!
            matFilename = os.path.join(dirname, 'dBTS_'+fnames[key]+'.mat')
            
            # savemat() cannot handle datetimes, so convert datetimes in output to
            # ML-compatible "days-since-epoch (31December0000)"
            item['datetime']['data'] = p.date2num(item['datetime']['data'])
            
            try:
              sio.savemat(matFilename, item)
            except:
              print(('Warning: Unable to write binary file: '+matFilename))
         elif binaryType.lower() == 'none':
            pass
         else:
            print(('Unrecognized binary type '+binaryType+' requested'))
            raise Exception
      
   else:
      pass

