#!/usr/bin/env python
"""
Extract Disturbance Storm Time (DST) index from LFM HDF files.  Store
results in pyLTR.TimeSeries object.

Execute `./dst.py --help` for more information.
"""

# Custom
import pyLTR
# 3rd Party
import numpy as n
import pylab
from pyhdf.SD import HDF4Error
# Standard
import pickle
import datetime
import optparse
import os.path
import sys
import re

def parseArgs():
   """
   Returns parameters used for DST extraction:
     - runDir: Path to a directory containing a MIX run.
     - runName: Name of run (ie. 'RUN-IDENTIFIER')
     - t0: datetime object of first step to be used in time series
     - t1: datetime object of last step to be used in time series
   Execute `dst.py --help` for details explaining these variables.
   """
   # additional optparse help available at:
   # http://docs.python.org/library/optparse.html
   # http://docs.python.org/lib/optparse-generating-help.html
   parser = optparse.OptionParser(usage='usage: %prog -p [PATH] [options]',
                                  version=pyLTR.Release.version)
   
   parser.add_option('-p', '--path', dest='path',
                     default='/Users/schmitt/paraview/testData/March1995_LM_r1432_single', metavar='PATH',
                     help='Path to LFM run directory.')

   parser.add_option('-r', '--runName', dest='runName',
                     default='', metavar='RUN_IDENTIFIER',
                     help='Name of run to extract DST from.  Leave empty unless there is more than one run in a directory.')    

   parser.add_option('-f', '--first', dest='t0',
                     default='', metavar='YYYY-MM-DD-HH-MM-SS', help='Date & Time that should be the first element of the time series')
   
   parser.add_option('-l', '--last', dest='t1',
                     default='', metavar='YYYY-MM-DD-HH-MM-SS', help='Date & Time of last element for the time series')

   parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                      help='About this program.')

   (options, args) = parser.parse_args()
   if options.about:
       print((sys.argv[0] + " version " + pyLTR.Release.version))
       print("")
       print("This script extracts the Disturbance Storm Time (dst) index ")
       print("from an LFM run.  Results are saved to FILENAME.pkl as a")
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
   assert( os.path.exists(options.path) )

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
           
   return (options.path, options.runName, t0, t1)

def dipole(x, dipoleMoment = 3.0e4):
   """
   Returns dipole at location [x,y,z] in nT.

   >>> dipole([3.0,4.0,5.0]) # doctest:+ELLIPSIS
   array([ -76.3675..., -101.823...,  -42.426...])

   >>> dipole([3.0,4.0,5.0], 3.01e4) # doctest:+ELLIPSIS
   array([ -76.6220..., -102.162...,  -42.5678...])
   """
   if len(x) != 3:
      raise Exception('Input vector must be length 3.')
   
   r = n.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

   b = n.zeros(3)
   if r > 0:
      r5=r**5
      b[0] = -3*dipoleMoment*x[0]*x[2]/r5
      b[1] = -3*dipoleMoment*x[1]*x[2]/r5
      b[2] =    dipoleMoment*(x[0]**2+x[1]**2-2.0*x[2]**2)/r5

   return b

def find3dmin(array):
   """
   Find the minimum (and indices) value in the 3d input array.

   FIXME:  Should this be a sub-function of getDstIndices?
   >>> import numpy
   >>> find3dmin(numpy.array([[[ 5.],[ 10.]],  [[ -99.],[ 20.]]]))
   (-99.0, 1, 0, 0)
   """
   (ni,nj,nk) = array.shape
   
   minI = 0
   minJ = 0
   minK = 0
   minVal = array[minI, minJ, minK]

   for i in range(ni):
      for j in range(nj):
         for k in range(nk):
            if array[i,j,k] < minVal:
               minVal = array[i,j,k]
               minI = i
               minJ = j
               minK = k

   return (minVal, minI, minJ, minK)


def getCellCenters(x,y,z):
   """
   Returns cell centers of x,y,z float Arrays
   """
   # Calculate cell centers:
   (ni,nj,nk) = x.shape
   xc = n.empty((ni-1,nj-1,nk-1))
   yc = n.empty((ni-1,nj-1,nk-1))
   zc = n.empty((ni-1,nj-1,nk-1))
   for i in range(ni-1):
      for j in range(nj-1):
         for k in range(nk-1):
            xc[i,j,k] = 0.125*(x[i,j,k]+x[i,j+1,k]+x[i,j,k+1]+x[i,j+1,k+1]+
                               x[i+1,j,k]+x[i+1,j+1,k]+x[i+1,j,k+1]+x[i+1,j+1,k+1]);
            yc[i,j,k] = 0.125*(y[i,j,k]+y[i,j+1,k]+y[i,j,k+1]+y[i,j+1,k+1]+
                               y[i+1,j,k]+y[i+1,j+1,k]+y[i+1,j,k+1]+y[i+1,j+1,k+1]);
            zc[i,j,k] = 0.125*(z[i,j,k]+z[i,j+1,k]+z[i,j,k+1]+z[i,j+1,k+1]+
                               z[i+1,j,k]+z[i+1,j+1,k]+z[i+1,j,k+1]+z[i+1,j+1,k+1]);

   return (xc, yc, zc)

def getDstIndices(xc,yc,zc, nPoints=4, radius=3.0):
   """
   Returns a list of (i,j,k) indices corresponding to (x,y,z)
   locations to be used for DST computation.

   Parameters: 
     xc,yc,zc: position of cell centers
     nPoints:  How many indices do we want?
     radius: Distance from center of earth to calculate 
   """                    
   # Points around the Earth we'll calculate (i,j,k) locations:
   dphi = 2.0*n.pi / float(nPoints)
   phi = n.linspace(0.0, 2.0*n.pi - dphi, nPoints)
   x2 = radius * n.cos(phi)
   y2 = radius * n.sin(phi)
   z2 = n.zeros(phi.shape)

   # Find the nearest (i,j,k) cells to the nPoints at a distance of radius 
   ijk = []
   (ni,nj,nk) = xc.shape
   for idx, angle in enumerate(phi):
      distance = n.empty(xc.shape)
      for i in range(ni):
         for j in range(nj):
            for k in range(nk):
               distance[i,j,k] = n.sqrt( (x2[idx]-xc[i,j,k])**2 + 
                                         (y2[idx]-yc[i,j,k])**2 + 
                                         (z2[idx]-zc[i,j,k])**2 )
      
      (val, idxI, idxJ, idxK) = find3dmin(distance)
      ijk.append((idxI, idxJ, idxK))

   return ijk

def getDst(path, runName, t0, t1):
   """
   Returns a pyLTR.TimeSeries object containing the DST index
   
   Parameters:
     path: path to run 
     runName: name of run we wish to compute DST for.
   """
   data = pyLTR.Models.LFM(path, runName)
   
   # Hard-code a subset (useful for testing & debugging):
   #data = pyLTR.Models.LFM('/Users/schmitt/paraview/testData/March1995_LM_r1432_single', 'LMs')

   # Make sure variables are defined in the model.
   modelVars = data.getVarNames()
   for v in ['X_grid', 'Y_grid', 'Z_grid',
             'bi_', 'bj_', 'bk_', 
             'bx_', 'by_', 'bz_', 
             'c_', 
             'ei_', 'ej_', 'ek_', 
             'rho_', 'vx_', 'vy_', 'vz_']:
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

   t_doy = []
   dst = []

   # Pre-compute some quantities
   #FIXME:  Should this be a configurable parameter?
   earthRadius = 6.38e8
   #earthRadius = 6340e5
   x = data.read('X_grid', timeRange[index0]) / earthRadius
   y = data.read('Y_grid', timeRange[index0]) / earthRadius
   z = data.read('Z_grid', timeRange[index0]) / earthRadius

   dipStr = data.readAttribute('dipole_moment', timeRange[index0])
   dipStr = dipStr.split('=')[1].strip().split()[0]
   gauss_2_nT = 1.0e5
   dipoleMoment = float(dipStr) / (earthRadius**3) * gauss_2_nT

   print('Reticulating splines')
   #FIXME:  nPoints and radius should be configurable
   nPoints = 4
   radius = 3.0
   (xc, yc, zc) = getCellCenters(x,y,z)
   ijk = getDstIndices(xc,yc,zc, nPoints, radius)

   print(( 'Extracting DST from MHD files for time series over %d time steps.' % (index1-index0) ))
   
   # Output a status bar displaying how far along the computation is.   
   progress = pyLTR.StatusBar(0, index1-index0)
   progress.start()
   for i,time in enumerate(timeRange[index0:index1]):
      try:
         tt = time.timetuple()
         dayFraction = (tt.tm_hour+tt.tm_min/60.+tt.tm_sec/(60.*60.))/24.
         t_doy.append( float(tt.tm_yday) + dayFraction )

         ## Skip the file if it can't be read.
         try:
            data.readAttribute('time', time)
         except HDF4Error as msg:
            sys.stderr.write('HDF4Error: Trouble "' + data._LFM__io.filename + '"\n')
            sys.stderr.write('HDF4 Error message: "' + str(msg) + '"\n')
            sys.stderr.write('Skipping file.\n')
            sys.stderr.flush()
         
            #FIXME: What's the correct behavior here for the dst
            #       value? Set to last known good value?  Pop array values?
            dst.append(dst[len(dst)-1])
            continue      

         # Calculate DST
         dstVal = 0.0
         for idx in ijk:               
            #FIXME: when count=[1,1,1], data.read(...) returns a 3d
            #       array with just one element!            
            #FIXME: setting a particular start index doesn't work
            #       because dimensions are transposed because Fortran
            #       does the I/O... so just read all the data :-(
            #bz = data.read('bz_', time, start=idx, count=(1,1,1))
            bz = data.read('bz_', time)
                              
            # scale to nT
            bz = bz[idx[0], idx[1], idx[2]] * gauss_2_nT
            
            dip = dipole( (xc[idx[0],idx[1],idx[2]],
                           yc[idx[0],idx[1],idx[2]],
                           zc[idx[0],idx[1],idx[2]]),
                          dipoleMoment)
            
            dstVal += bz - dip[2]

         dstVal /= len(ijk)
         dst.append( dstVal )

         progress.increment()

      except KeyboardInterrupt:
         # Exit when the user hits CTRL+C.
         progress.stop()
         progress.join()            
         print('Exiting.')
         sys.exit(0)
      except:
         # Cleanup progress bar if something bad happened.
         progress.stop()
         progress.join()
         raise
   progress.stop()
   progress.join()

   if len(dst) == 0:
      raise Exception(('No data computed.  Did you specify a valid run path?'))    
   
   data = pyLTR.TimeSeries()
   data.append('datetime',     'Date & Time', '', timeRange[index0:index1])
   data.append('doy', 'Day of Year', '',      t_doy)
   data.append('dst', 'Disturbance Storm Time Index', 'nT', dst)

   return data

if __name__ == '__main__':
#   import doctest
#   doctest.testmod()
#else:
   (path, run, t0, t1) = parseArgs()
   
   dstTimeSeries = getDst(path, run, t0, t1)
   
   # --- Dump a  pickle!
   print('Serializing pyLTR.TimeSeries object of DST data using the Python Pickle package.')
   filename = os.path.join(path, 'dst.pkl')
   print(('Writing ' + filename))

   fh=open(filename,'wb')
   pickle.dump(dstTimeSeries,fh,protocol=2)
   fh.close()

   # --- Make a plot of everything
   print('Creating plot of LFM DST.')
   filename = os.path.join(path, 'dst.png')
   print(('Writing ' + filename))
   pyLTR.Graphics.TimeSeries.MultiPlot(dstTimeSeries, 'doy', ['dst'])
   pylab.title(os.path.join(path, run))
   pylab.savefig(filename)
