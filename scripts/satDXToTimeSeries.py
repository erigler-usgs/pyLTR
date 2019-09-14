#!/usr/bin/env python
# Custom
import pyLTR
import FileTools as ft
# 3rd Party
import numpy as n
# Standard
import datetime
import pickle
import optparse
import os.path
import sys

def satDXToTimeSeries(filename):
    file = open(filename)
    data = ft.read(file)
    (rows,cols) = data.shape
    dict=pyLTR.TimeSeries()
    mon=[]
    mday=[]
    dayFraction=[]
    doy=[]
    atime=[]
    for i in range(rows):
      d = datetime.date(int(data[i,0]),1,1)+datetime.timedelta(int(data[i,1])-1)
      mon.append(d.month)
      mday.append(d.day)
      dayFraction=(data[i,2]+data[i,3]/60.0+data[i,4]/3600.0)/24.0
      doy.append(float(data[i,1])+dayFraction)   
      #atime.append(datetime.datetime(d.year,d.month,d.day,data[i,2],data[i,3],data[i,4]))
    dict.append('time_doy','Day of Year',' ',doy)
    dict.append('density','Den',r'$\mathrm{1/cm^3}$',data[:,5])
    dict.append('vx','Vx',r'$\mathrm{km/s}$',data[:,6])
    dict.append('vy','Vy',r'$\mathrm{km/s}$',data[:,7])
    dict.append('vz','Vz',r'$\mathrm{km/s}$',data[:,8])
    dict.append('p','P',r'$\mathrm{keV/cm^3}$',data[:,9])
    dict.append('bx','Bx',r'$\mathrm{nT}$',data[:,10])
    dict.append('by','By',r'$\mathrm{nT}$',data[:,11])
    dict.append('bz','Bz',r'$\mathrm{nT}$',data[:,12])
    dict.append('ex','Ex',r'$\mathrm{V/m}$',data[:,13])
    dict.append('ey','Ey',r'$\mathrm{V/m}$',data[:,14])
    dict.append('ez','Ez',r'$\mathrm{V/m}$',data[:,15])
    
    return dict

def parseArgs():
    """
    Returns parameters used for DX-Satellite-Interp-to-pyLTR.TimeSeries conversion
      goesFile
    Execute `satDXToTimeSeries.py --help` for details explaining these variables.
    """
    # additional optparse help available at:
    # http://docs.python.org/library/optparse.html
    # http://docs.python.org/lib/optparse-generating-help.html
    parser = optparse.OptionParser()

    parser.add_option('-f', '--filename', dest='filename',
                      default='LFM', metavar='FILE',
                      help='Path to DX satellite file.')

    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    if options.about:
        print((sys.argv[0] + " version " + pyLTR.Release.version))
        print("")
        print("This script reads MHD variables (in GSM coordinates) from a DX satellite")
        print("interpolation data file and saves the data to FILENAME.pkl as a ")
        print("pyLTR.TimeSeries object using Python's Pickle module.")
        print("http://docs.python.org/library/pickle.html")
        print("")
        print("To load the pyLTR.TimeSeries object into memory, use Python's cPickle:")
        print("")
        print("      import cPickle")
        print("      fh = open(filename, 'rb')")
        print("      obj = cPickle.load(fh)")
        print("      fh.close()")
        print("")
        print("Note that pkl files may be machine-dependent thanks to big/little ")
        print("endian machines.")
                
        sys.exit()

    # Sanitize inputs
    assert( os.path.exists(options.filename) )

    return (options.filename)

if __name__ == '__main__':
    filename = parseArgs()
    data=satDXToTimeSeries(filename)
    out=os.path.splitext(filename)[0]+'.pkl'
    f=open(out,'wb') 
    pickle.dump(data,f)
    f.close()

