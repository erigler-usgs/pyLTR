#!/usr/bin/env python
"""
pyLTR Solar Wind Processing.

Execute './solarWind.py --help' for more information.
"""

# Custom
import pyLTR

# 3rd-party
import pylab

# Standard
import pickle
import optparse
import os
import sys

def parseArgs():
    """
    Returns solar wind filename & whether or not to run interactively
    """
    # additional optparse help available at:
    # http://docs.python.org/library/optparse.html
    # http://docs.python.org/lib/optparse-generating-help.html    
    parser = optparse.OptionParser(usage='usage: %prog -f [FILE] -o [FORMAT] [options]',
                                   version=pyLTR.Release.version)
    parser.add_option('-f', '--filename', dest='filename', default='SW-SM-DAT', metavar='FILE',
                      help='Path to solar wind file to convert (i.e. CCMC, '
                      'ENLIL, LFM, MAS, OMNI, etc.).')
    parser.add_option('-F', '--force', dest='isInteractive', default=True, action='store_false',
                      help='Force solar wind conversion without asking the'
                      'user for input.')
    parser.add_option('-o', '--output', dest='format', default='LFM', metavar='[LFM | TIEGCM]',
                      help='Format to write.  Options are "LFM" or "TIEGCM".')
    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                      help='About this program.')

    (options, args) = parser.parse_args()

    if options.about:
        print((sys.argv[0] + ' version ' + pyLTR.Release.version))
        print('')
        print('This script does several things:')
        print('  1. Read solar wind data (several formats supported: CCMC, ENLIL, LFM, MAS, OMNI)')
        print('  2. Generate standard plots of solar wind data')
        print('  3. Save all solar wind data as a pyLTR.TimeSeries via a Python Pickle file.')
        print('  4. Write output in a model file format.')
        print('     - "LFM" format will:')
        print('         a. Generate coefficients for Bx Fit')
        print('         b. Save a SW-SM-DAT file')
        print('         c. Generate pyLTR.TimeSeries of LFM Solar winddata.')
        print('     - "TIEGCM" format will:')
        print('         a. Compute 15-minute boxcar average lagged by 5 minutes')
        print('         b. Sub-sample at 5-minutes')
        print('         c. Write NetCDF IMF data file')
        print('         d. Write pyLTR.TimeSeries (as a Python Pickle file) containing TIEGCM IMF data.')
        sys.exit()

    # Do some error checking
    if not options.isInteractive:
        if not os.path.isfile(options.filename):
            raise Exception('Error:  Could not find file "%s" and attempting to run without user input (--force)!  At least give me the right file to read!' % options.filename)
    else:
        # Make sure we get a file on the disk.
        if not os.path.isfile(options.filename):
            while True:                
                options.filename = input('Could not find "%s".  Please enter the path to a solar wind file:\n\t' % (options.filename))
                if os.path.isfile(options.filename):
                    break

    if ( (options.format != 'LFM') &
         (options.format != 'TIEGCM') ):
        raise Exception('Did not understand output format "%s".  Valid options are either "LFM" or "TIEGCM".' % options.format)

    return (options.filename, options.isInteractive, options.format)

def bxFit(sw, fileType, filename, isInteractive):
    def bxFitPlot(bxFit_array):
        pyLTR.Graphics.TimeSeries.BasicPlot(sw.data, 'time_doy', 'bx', color='k')
        pylab.plot(sw.data.getData('time_doy'), bxFit_array, 'g')
        pylab.title('Bx Fit Coefficients ('+fileType+'):\n$Bx_{fit}(0)$=%f      $By_{coef}$=%f      $Bz_{coef}$=%f' % (coef[0], coef[1], coef[2]) )
        pylab.legend(('$Bx$','$Bx_{fit}$'))

    while isInteractive:
        continueStr = input( 'Calculate Bx Fit? (y/n): ' )
        if (continueStr.lower() == 'y') | (continueStr.lower() == 'yes'):
            break
        elif (continueStr.lower() == 'n') | (continueStr.lower() == 'no'):
            return

    coef = sw.bxFit()

    print(('Bx Fit Coefficients are ', coef))
    by = sw.data.getData('by')
    bz = sw.data.getData('bz')
    bxFit = coef[0] + coef[1] * by + coef[2] * bz

    if isInteractive:
        bxFitPlot(bxFit)
        print('Close "Bx Fit Coefficients" plot window to continue.')
        pylab.show()

        while True:
            continueStr = input( 'Export to LFM solar wind file & generate plots? (y/n): ' )
            if (continueStr.lower() == 'y') | (continueStr.lower() == 'yes'):
                break
            elif (continueStr.lower() == 'n') | (continueStr.lower() == 'no'):
                print('You told me not to continue. Exiting.')
                sys.exit()

    # Save plot
    bxFitPlot(bxFit)
    bxPlotFilename = os.path.basename(filename) + '_bxFit.png'
    print(('Saving "%s"' % bxPlotFilename))
    pylab.savefig(bxPlotFilename)

if __name__ == '__main__':
    (filename, isInteractive, outputFormat) = parseArgs()
    fileType = pyLTR.SolarWind.fileType(filename)

    # Read solar wind data into 'sw' object.
    sw = eval('pyLTR.SolarWind.'+fileType)(filename)

    # Do output format-specific tasks:
    if (outputFormat == 'LFM'):
        # Bx Fit & prompt user for details.
        bxFit( sw, fileType, filename, isInteractive )
        
        # Write LFM solar wind file
        pyLTR.SolarWind.Writer.LFM(sw, os.path.basename(filename))
    elif (outputFormat == 'TIEGCM'):
        # Write TIEGCM IMF solar wind file
        pyLTR.SolarWind.Writer.TIEGCM(sw, os.path.basename(filename))
    else:
        raise Exception('Error:  Misunderstood output file format.')

    # If we made it this far, save some standard items:

    # Save a plot of the solar wind data.
    pyLTR.Graphics.TimeSeries.MultiPlot(sw.data, 'time_doy', ['n', 'vx','vy','vz','t','bx','by','bz'])
    pylab.title('Solar Wind data for\n %s' % filename)
    swPlotFilename = os.path.basename(filename) + '.png'
    print(('Saving "%s"' % swPlotFilename))
    pylab.savefig(swPlotFilename)

    # Save pyLTR.TimeSeries object as a Pickle.
    pklFilename = os.path.basename(filename) + '.pkl'
    print(('Saving "%s"' % pklFilename))
    fh = open(pklFilename, 'wb')
    pickle.dump(sw.data, fh, protocol=2)
    fh.close()
