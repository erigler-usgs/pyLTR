#!/usr/bin/env python
"""
pyLTR MIX Time Series

Execute './mixTimeSeries.py --help' for more information.
"""

# Custom
import pyLTR

# 3rd-party
import pylab
import numpy

# Standard
import pickle
import datetime
import math
import optparse
import os
import re
import sys

def parseArgs():
    """
    Returns standard parameters used for extracting time series data from MIX output.
      - path: Path to a directory containing a MIX run.
      - run: Name of run (ie. 'RUN-IDENTIFIER')
      - t0: datetime object of first step to be used in time series
      - t1: datetime object of last step to be used in time series
    Execute `mixTimeSeries.py --help` for more information.
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

    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    if options.about:
        print((sys.argv[0] + ' version ' + pyLTR.Release.version))
        print('')
        print('This script searches a path for any MIX ionosphere output files and ')
        print('computes useful metrics:')
        print('  - Cross Polar Cap Potential (cpcp)')
        print('  - Hemispheric Power (hp)')
        print('  - Positive current density (ipfac)')
        print('for both the Northern and Southern hemispheres.')
        print('')
        print('This script outputs two files:')
        print('  1.  "runPath/mixTimeSeries.png" - summary plot of variables')
        print('  2.  "runPath/mixTimeSeries.pkl" - pyLTR.TimeSeries file ')
        print('      serialized with the Python Pickle package.  See the ')
        print('      Python cPickle documentation for details on extracting ')
        print('      the data.')
        print('To limit the time range, use the "--first" and/or "--last" flags.')
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
            
    return (path, run, t0, t1)

def extractQuantities(path, run, t0, t1):
    """
    Extract MIX quantities from the input path.
    
    Execute `mixTimeSeries.py --help` for details on the function parameters (path,run,t0,t1).
    
    Outputs a pyLTR.TimeSeries object containing the data.
    """
    data = pyLTR.Models.MIX(path, run)

    # hard-coded input for testing & debugging:
    #data = pyLTR.Models.LFM('/hao/aim2/schmitt/data/LTR-2_0_1b/r1432/March1995/LR/single', 'LRs')
        
    #Make sure variables are defined in the model.
    modelVars = data.getVarNames()
    for v in ['Grid X', 'Grid Y', 
              'Poynting flux North [???]', 'Poynting flux South [???]', 
              'FAC North [A/m^2]', 'FAC South [A/m^2]',
              'Number flux North [1/cm^2 s]', 'Number flux South [1/cm^2 s]', 
              'BBE average energy North [keV]', 'BBE average energy South [keV]',
              'BBE number flux North [1/cm^2 s]', 'BBE number flux South [1/cm^2 s]', 
              'Cusp average energy North [keV]', 'Cusp average energy South [keV]',
              'Cusp number flux North [1/cm^2 s]', 'Cusp number flux South [1/cm^2 s]']:
#        print v
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

    t_doy   = []
    isparNorth  = []
    isparSouth  = []
    hpNorth    = []
    hpSouth    = []
    asparNorth = []
    asparSouth = []
    bbe = 0
    cusp = 0

    # Pre-compute area of the grid.
    x = data.read('Grid X', timeRange[index0])
    y = data.read('Grid Y', timeRange[index0])
    # Fix singularity at the pole
    x[:,0] = 0.0
    y[:,0] = 0.0
    z = numpy.sqrt(1.0-x**2-y**2)
    ri = 6500.0e3  # Radius of ionosphere in meters
    areaMixGrid = pyLTR.math.integrate.calcFaceAreas(x,y,z)*ri*ri

    for i,time in enumerate(timeRange[index0:index1]):
        try:
            # -- Day of Year
            tt = time.timetuple()
            t_doy.append(tt.tm_yday+tt.tm_hour/24.0+tt.tm_min/1440.0+tt.tm_sec/86400.0)

            # --- Cross Polar Cap Potential
            psi = data.read('Poynting flux North [???]', time)
            psi[psi<0] = 0.
            spar = areaMixGrid*psi[:-1,:-1] 
            # is MIX grid in m^2? spar is in mW/m^2
            # mW to GW
            isparNorth.append(spar.sum() * 1.e-12)
            spar = psi[psi>0.1].mean()
            asparNorth.append(spar.sum())

            psi = data.read('Poynting flux South [???]', time)
            psi[psi>0] = 0.
            spar = areaMixGrid*psi[:-1,:-1] 
            # is MIX grid in m^2? spar is in mW/m^2
            # mW to GW
            isparSouth.append(spar.sum()*-1.e-12)
            spar = psi[psi<-0.1].mean()
            asparSouth.append(spar.sum() * -1.)
            
            # --- Hemispheric Power
            energy = data.read('Average energy North [keV]', time)
            flux = data.read('Number flux North [1/cm^2 s]', time)
            mhp = areaMixGrid*energy[:-1,:-1] * flux[:-1,:-1]
            flux = data.read('BBE number flux North [1/cm^2 s]', time)
            energy = data.read('BBE average energy North [keV]', time)
            bbe = areaMixGrid*energy[:-1,:-1] * flux[:-1,:-1]
            flux = data.read('Cusp number flux North [1/cm^2 s]', time)
            energy = data.read('Cusp average energy North [keV]', time)
            cusp = areaMixGrid*energy[:-1,:-1] * flux[:-1,:-1]
            hp = mhp + bbe + cusp
            # KeV/cm^2s to mW/m^2 to GW
            hpNorth.append(hp.sum() * 1.6e-21) 

            energy = data.read('Average energy South [keV]', time)
            flux = data.read('Number flux South [1/cm^2 s]', time)
            mhp = areaMixGrid*energy[:-1,:-1] * flux[:-1,:-1]
            flux = data.read('BBE number flux South [1/cm^2 s]', time)
            energy = data.read('BBE average energy South [keV]', time)
            bbe = areaMixGrid*energy[:-1,:-1] * flux[:-1,:-1]
            flux = data.read('Cusp number flux South [1/cm^2 s]', time)
            energy = data.read('Cusp average energy South [keV]', time)
            cusp = areaMixGrid*energy[:-1,:-1] * flux[:-1,:-1]
            hp = mhp + bbe + cusp
            # KeV/cm^2s to mW/m^2 to GW
            hpSouth.append(hp.sum() * 1.6e-21)

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

    dataNorth = pyLTR.TimeSeries()
    dataSouth = pyLTR.TimeSeries()
    dataNorth.append('datetime', 'Date & Time', '', timeRange[index0:index1])
    dataSouth.append('datetime', 'Date & Time', '', timeRange[index0:index1])
    dataNorth.append('doy', 'Day of Year', '', t_doy)
    dataSouth.append('doy', 'Day of Year', '', t_doy)
    
    # "N" and "S" label subscripts are redundant here, potentially leading to
    # mis-labeling of plots
    #dataNorth.append('cpcp', r'$\Phi_N$', 'kV', cpcpNorth)
    #dataSouth.append('cpcp', r'$\Phi_S$', 'kV', cpcpSouth)
    #
    #dataNorth.append('hp', r'$HP_N$', 'GW', hpNorth)
    #dataSouth.append('hp', r'$HP_S$', 'GW', hpSouth)
    #
    #dataNorth.append('ipfac', r'$FAC_N$', 'MA', ipfacNorth)
    #dataSouth.append('ipfac', r'$FAC_S$', 'MA', ipfacSouth)
    
    dataNorth.append('ispar', r'$Integrated S||$', 'GW', isparNorth)
    dataSouth.append('ispar', r'$Integrated S||$', 'GW', isparSouth)
    
    dataNorth.append('hp', r'$THP$', 'GW', hpNorth)
    dataSouth.append('hp', r'$THP$', 'GW', hpSouth)
    
    dataNorth.append('aspar', r'$Mean S||r$', 'mW/m^2', asparNorth)
    dataSouth.append('aspar', r'$Mean S||$', 'mW/m^2', asparSouth)

    return (dataNorth, dataSouth)

if __name__ == '__main__':

    (path, run, t0, t1) = parseArgs()

    (dataNorth, dataSouth) = extractQuantities(path, run, t0, t1)
    
    # --- Dump a  pickle!
    print('Serializing pyLTR.TimeSeries object of MIX data using the Python Pickle package.')
    filename = os.path.join(path, 'mixSparTimeSeries.pkl')
    print(('Writing ' + filename))
    fh = open(filename, 'wb')
    pickle.dump([dataNorth, dataSouth], fh, protocol=2)
    fh.close()

    # --- Make a plot of everything
    print('Creating summary plot of MIX time series data.')
    filename = os.path.join(path, 'mixSparTimeSeries.png')
    print(('Writing ' + filename))
    pyLTR.Graphics.TimeSeries.MultiPlotN([dataNorth, dataSouth], 
                                         'datetime', ['ispar', 'hp', 'aspar'], 
                                         ['r','b'], ['North', 'South'])
    pylab.gcf().autofmt_xdate()
    pylab.title(os.path.join(path, run))
    pylab.savefig(filename,dpi=1200)
