from pyLTR.Models import Model
from pyLTR.Models.Hdf4Io import Hdf4Io
from pyLTR.Models.Hdf5Io import Hdf5Io

import numpy

import datetime
import glob
import os
import re

class LFM(Model):
    """
    Implementation class for Model I/O.  Typical variables used are:
      - X_grid, Y_grid, Z_grid
      - bx_, by_, bz_
      - c_
      - rho_
      - vx_, vy_, vz_
    Not as frequently used/may require processing:
      - ei_, ej_, ek_, bi_, bj_, bk_ (edge-centered data)
    """

    def __init__(self, runPath, runName,ext='.hdf'):
        """
        Parameters:
          runPath: path to directory containing MIX data files.
          runName: Optional parameter.  When specified, it searches
          runPath/runName* for files.  This is useful for single
          directories containing multiple runs.
        """
        Model.__init__(self, runPath, runName)

        filePrefix = os.path.join(self.runPath, self.runName)
        self.__fileList = glob.glob(filePrefix + '*_mhd_*'+ext)
        self.__fileList.sort()
        
        #
        # regex is used to determine filename convention (eg. UT
        # '2012-07-13T02-15-00' vs Timestep '0008000') and to obtain
        # runName if it is undefined.
        # 
        regex = (r'^([\S\-]+)' + # \S is "digits, letters and underscore".
                 r'_mhd_' + 
                 r'('
                 r'\d{4}\-\d{2}\-\d{2}T\d{2}\-\d{2}\-\d{2}Z' +  # UT-based filename
                 r'|' + 
                 r'\d{7}'  # timestep-based filename
                 r')' +
                 ext +                     
                 r'$')
        r = re.match(regex, os.path.basename(self.__fileList[0]))
        if not r:
            raise Exception('Having trouble identifying LFM MHD files.  Are you looking at the correct directory?')
        if (len(r.groups()) != 2):
            raise Exception('Having trouble identifying LFM MHD files.  Are you looking at the correct directory?')
        
        if not self.runName:
            self.runName = r.groups()[0]
        
        if len(r.groups()[1]) == 7:
            # Timestep filenames (eg. 'RunName_mhd_0008000.hdf')
            self.__timestepFilenames = True
        else:
            # UT-based filenames (eg. 'RunName_mhd_2012-07-13T02-47-00Z.hdf')
            self.__timestepFilenames = False
        
        # Pick the HDF4 or HDF5 IO Object based upon file extension
        if (ext == '.hdf'):
            self.__io = Hdf4Io( self.__fileList[0] )
        else:
            self.__io = Hdf5Io( self.__fileList[0] )
        self.runExt = ext

        # Set list of datetime objects corresponding to self.__fileList
        self.__timeRange = []
        self.getTimeRange()


    def getTimeRange(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available.
        
        LFM has files named like
        '[runName]_mhd_1995-03-21T16-00-00Z.hdf', one time step per
        file.

        Assumes self.__timestepFilenames boolean is set.
        """
        if not self.__timeRange:
            if self.__timestepFilenames:
                self.__timeRange = self.__getTimeRange_timestep()
            else:
                self.__timeRange = self.__getTimeRange_ut()

        return self.__timeRange


    def __getTimeRange_ut(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available.
        
        LFM has files named like
        '[runName]_mhd_1995-03-21T16-00-00Z.hdf', one time step per
        file.
        """        
        for f in self.__fileList:
            regex = ( r'^' + self.runName + '_mhd_' +
                      r'(\d{4})\-(\d{2})\-(\d{2})' +
                      r'T' +
                      r'(\d{2})\-(\d{2})-(\d{2})' +
                      r'Z'+
                      self.runExt + r'$' )
            
            r = re.match(regex, os.path.basename(f))
            
            assert( len(r.groups()) == 6 )
            t = [ int(match) for match in r.groups() ]
            self.__timeRange.append( datetime.datetime(year=t[0],
                                                       month=t[1],
                                                       day=t[2],
                                                       hour=t[3],
                                                       minute=t[4],
                                                       second=t[5]) )
            
        return self.__timeRange

    def __getTimeRange_timestep(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available.
        
        LFM has files named like 
        '[runName]_mhd_069000.hdf',when in step dump mode.
        """

        # Ask user to input starting time...
        format_regex = "^([\s]*\d{4})[\s]+(0?[1-9]|1[012])[\s]+(0?[1-9]|[12]\d|3[01])[\s]+(0?\d|1\d|2[0-4])[\s]+([0-5]?\d?|60)[\s]+([0-5]?\d?|60)[\s]*$"
        while True:
            line = eval(input('Enter start time of run (YYYY MM DD HH MM SS):'))
            r = re.match(format_regex,line)
            if r:
                vals = [int(s) for s in line.split()]
                runStart=datetime.datetime(year=vals[0],month=vals[1],day=vals[2],
                                           hour=vals[3],minute=vals[4],second=vals[5])
                break
            else:
                print('Invaild entry')
        
        # Calculate the time range...
        for f in self.__fileList:

            self.__io.setFilename(f)
            #hdf = self.__getHdfObj()
            #assert(hdf.attributes().has_key('time'))
            #seconds = hdf.attributes()['time']

            #try:
            elapsedSeconds = self.__io.readAttribute('time_8byte', None)
            #except hdferror:
            #    elapsedSeconds = self.__io.readAttribute('time', None)
            date_time = runStart+datetime.timedelta(seconds=elapsedSeconds-3000.0)
            self.__timeRange.append(date_time)
        
        return self.__timeRange
        

    def __setTimeValue(self, time):
        """
        Select a particular time slice.  Use getTimeRange() to see all
        available data.
        """
        # Check the input:
        assert( isinstance(time, datetime.datetime) )
        
        # This will raise an exception if the time slice is missing:
        i = (self.__timeRange).index(time)
       
        filename=self.__fileList[i]
        
        # Set this filename for IO.
        self.__io.setFilename(filename)

    def getAttributeNames(self):
        """
        Returns a list of attributes available in the dataset.    
        """
        return( self.__io.getAttributeNames() )

    def readAttribute(self, attrName, time):
        """
        Returns an attribute at a particular time.  Use an item in the
        lists returned by getAttributeNames() and getTimeRange() to
        obtain valid data.
        """
        self.__setTimeValue(time)
        return self.__io.readAttribute(attrName, None)
    def getVarNames(self):
        """
        Returns a list of variables available in the dataset.    
        """
        return( self.__io.getVarNames() )

    def read(self, varName, time, start=None, count=None, stride=None):
        """
        Returns a variable (varName) at a particular time.  Use an
        item in the lists returned by getVarNames() and getTimeRange()
        to obtain valid data.

        WARNING:  start,count,stride hasn't been tested extensively!
        
        FIXME: LFM MHD HDF4 files are written in Fortran ordering.
        Accessing a single element via start index is unpredictable.
        Need to read *all* the data and numpy.reshape it before
        accessing individual elements!
        """

        self.__setTimeValue(time)

        data = self.__io.read(varName, None, start, count, stride)

        # Arrays in LFM HDF4 files have several strange conventions:
        # 
        #   1.  Data is stored in C-order, but shape is transposed in
        #       Fortran order!  Why?  Because LFM stores its arrays using
        #       the rather odd A++/P++ library.  Let's reverse this evil:
        s = data.shape
        data = numpy.reshape( data.ravel(), (s[2], s[1], s[0]), order='F' )

        #   2.  Cell, Face & Edge-centered arrays have extra unused indices.
        #       Let's return data only at valid points.        

        # If the user specifies particular indices, we probably shouldn't remove data!
        if start:
            #FIXME:  If count=[1,1,1], this returns a 3d array with just one element!
            return data
        if (varName in  ['bx_', 'by_', 'bz_',  'c_', 'rho_', 'vx_', 'vy_', 'vz_']):
            # Cell-centers
            return data[:-1,:-1,:-1]
        elif (varName is 'bi_'):
            # i faces
            return data[:,:-1,:-1]
        elif (varName is 'bj_'):
            # j faces
            return data[:-1,:,:-1]
        elif (varName is 'bk_'):
            # k faces
            return data[:-1,:-1,:]
        elif (varName is 'ei_'):
            # i edges
            return data[:-1,:,:]
        elif (varName is 'ej_'):
            # j edges
            return data[:,:-1,:]
        elif (varName is 'ek_'):
            # k edges
            return data[:,:,:-1]
        else:
            # Assume data defined everywhere (ie. for XYZ grid).
            return data

    def getEqSlice(self,varName,time):
        """
        Returns and Equatorial slice of the given variable
        """
        data = self.read(varName,time)
        if (varName in ['X_grid','Y_grid','Z_grid']):
            nk=data.shape[2]-1
            # Equatorial slice
            dusk=data[:,:,0]
            dawn=data[:,:,nk/2]
            dawn=dawn[:,::-1] # reverse the j-index
            # Now stack everything together
            # remove the axis from the  dawn array and stack
            eq=numpy.hstack( (dusk,dawn[:,1:]) )
            eq_c = 0.25*( eq[:-1,:-1]+eq[:-1,1:]+eq[1:,:-1]+eq[1:,1:] )
            eq_c = numpy.append(eq_c.transpose(),[eq_c[:,0]],axis=0).transpose()
            return eq_c
        else:
            nk=data.shape[2]
            dusk = .5*(data[:,:,0]+data[:,:,nk-1])
            dawn = .5*(data[:,:,nk/2]+data[:,:,nk/2-1])
            eq = numpy.hstack( (dusk,dawn[:,::-1]) )
            eq = numpy.append(eq.transpose(),[eq[:,0]],axis=0).transpose()
            return eq

    def getMerSlice(self,varName,time):
        """
        Returns and Meridonal slice of the given variable
        """

        data = self.read(varName,time)
        if (varName in ['X_grid','Y_grid','Z_grid']):
            nk=data.shape[2]-1
            # Meridional slice
            north=data[:,:,nk/4]
            south=data[:,:,3*nk/4]
            south=south[:,::-1] # reverse the j-index
            # remove the axis from the dawn array and stack to get one plane
            mer=numpy.hstack( (north,south[:,1:]) )
            mer_c = 0.25*( mer[:-1,:-1]+mer[:-1,1:]+mer[1:,:-1]+mer[1:,1:])
            mer_c = numpy.append(mer_c.transpose(),[mer_c[:,0]],axis=0).transpose()
            return mer_c
        else:
            nk=data.shape[2]
            north = .5*(data[:,:,nk/4]+data[:,:,nk/4-1])
            south = .5*(data[:,:,3*nk/4,]+data[:,:,3*nk/4-1])
            mer = numpy.hstack( (north,south[:,::-1]) )
            mer = numpy.append(mer.transpose(),[mer[:,0]],axis=0).transpose()
            return mer

    def getKPlane(self,varName,time,k):
        """
        Returns and Meridonal slice of the given variable
        """
        
        data = self.read(varName,time)
        if (varName in ['X_grid','Y_grid','Z_grid']):
            nk=data.shape[2]-1
            if (k > nk/2): 
                k = k-nk/2
            # Meridional slice
            north=data[:,:,k]
            south=data[:,:,k+nk/2]
            south=south[:,::-1] # reverse the j-index
            # remove the axis from the dawn array and stack to get one plane
            mer=numpy.hstack( (north,south[:,1:]) )
            mer_c = 0.25*( mer[:-1,:-1]+mer[:-1,1:]+mer[1:,:-1]+mer[1:,1:])
            mer_c = numpy.append(mer_c.transpose(),[mer_c[:,0]],axis=0).transpose()
            return mer_c
        else:
            nk=data.shape[2]
            if (k > nk/2): 
                k = k-nk/2
            north = .5*(data[:,:,k]+data[:,:,k-1])
            south = .5*(data[:,:,k+nk/2]+data[:,:,k+nk/2-1])
            mer = numpy.hstack( (north,south[:,::-1]) )
            mer = numpy.append(mer.transpose(),[mer[:,0]],axis=0).transpose()
            return mer

if (__name__ == '__main__'):
    l = LFM('/hao/aim2/schmitt/data/July_Merge/LR_CMIT_merge/May_19-20_1996', 'CMIT')
    print('Time Range:')
    print((l.getTimeRange()))
    print()    
    print('Attributes:')
    print((l.getAttributeNames()))
    print()        
    print('Modified Julian Date:')
    print((l.readAttribute('mjd', datetime.datetime(1996, 5, 19, 16, 2))))
    print()    
    print('Variables:')
    print((l.getVarNames()))
    print()
    print('Density: ')
    print((l.getVarNames()))
    var = l.read('rho_', datetime.datetime(1996,5,19,16,0,0))


