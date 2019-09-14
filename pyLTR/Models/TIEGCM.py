from pyLTR.Models import Model
from pyLTR.Models.NetcdfIo import NetcdfIo

import datetime
import glob
import os
import re

class TIEGCM(Model):
    """
    Implementation class for TIEGCM I/O.

    Standard variables:
      - Geographic grid: 'lat', 'lon', 'lev'
      - Magnetic grid: 'mlat', 'mlon', 'mlev'
      - time:  'mtime' (ignore 'time'... use mtime!)
    Typical primary history parameters for TIEGCM are:
      - 'N2D', 'N4S', 'N4S_NM', 'NE', 'NO', 'NO_NM',
      - 'O1', 'O1_NM', 'O2', 'O2P', 'O2_NM', 'OMEGA',
      - 'OP', 'OP_NM', 'POTEN', 'TE', 'TI', 'TLBC', 'TLBC_NM',
      - 'TN', 'TN_NM', 'ULBC', 'ULBC_NM', 'UN', 'UN_NM',
      - 'VLBC', 'VLBC_NM', 'VN', 'VN_NM', 'Z'    
    Typical secondary history parameters for CMIT are:
      - Standard vars:
         - 'TN','UN','VN','O2','O1','N2','NO','NO_COOL',
         - 'OP','TI','TE','NE','O2P','OMEGA','Z','POTEN',
         - 'UI_VEL','VI_VEL','WI_VEL','TEC','QJI_TN',
      - CMIT exchange vars:
         - From LFM/MIX:
            - Potential (geographic): 'gpot','gpotm'
            - Energy: 'geng'
            - Flux: 'gflx'
         - Send to LFM/MIX:
            - Conductances: 'gzigm1','gzigm2'
            - Current: 'gnsrhs'

    FIXME: This class is functional to read data, but the whole I/O
       system probably needs to be refactored.  TIEGCM stores multiple
       timesteps per file.  The way we treat that in this package is a
       bit kludgy: You must have a datetime.datetime object
       representing the date & time you care about.  Then we search
       through the list of timesteps to find a match... This is
       probably slow.  Not sure of the best way to deal with this.

       It's also not obvious if the NetcdfIo class is doing full I/O
       when reading a single Netcdf file to grab attribute data.

       Finally, TIEGCM variables have a bit more going on than other
       models I'm used to (LFM and MIX).  Some are in geomagnetic,
       others are in geographic coordinates.  Some have pressure
       levels, etc.  Need a better way to deal with dimensions in an
       abstract way.  Right now we don't provide many mechanisms to
       read metadata in TIEGCM::read(...).  This is a major drawback
       of working with this IO package.

       Ultimately I'm not sure of the best way to fix this.  I don't
       really want to write a genericwrapper around pyhdf (hdf4),
       scipy.io (netcdf) and future formats (Hdf5, etc)!
    """

    def __init__(self, runPath, runName):
        """
        Parameters:
          runPath: path to directory containing TIEGCM data files.
          runName: Optional parameter.  When specified, it searches
          runPath/runName* for files.  This is useful for single
          directories containing multiple runs.
        """
        Model.__init__(self, runPath, runName)

        filePrefix = os.path.join(self.runPath, self.runName)
        self.__fileList = glob.glob(filePrefix + '*_sech_tie_*nc')
        self.__fileList.sort()
        
        if not self.runName:
            regex = (r'^([\S\-]+)' + # \S is "digits, letters and underscore".
                     r'_sech_tie_\d{4}\-\d{2}\-\d{2}T\d{2}.nc$')
            r = re.match(regex, os.path.basename(self.__fileList[0]))
            if len(r.groups()) < 1:
                raise Exception('Could not determine run name.  Are you looking at the correct directory?')
            self.runName = r.groups()[0]

        self.__io = NetcdfIo( self.__fileList[0] )
        
        self.__timeRange = []
        self.getTimeRange()


    def getTimeRange(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available.
        
        TIEGCM might have files named like
          '[runName]_tie_1995-03-21T16.nc'
          '[runName]_sech_tie_1995-03-21T16.nc'
        with multiple time steps per file.
        """
        if not self.__timeRange:
            for f in self.__fileList:
                regex = ( r'^' + self.runName + '_sech_tie_' +
                          r'(\d{4})\-(\d{2})\-(\d{2})' +
                          r'T' +
                          r'(\d{2})\-(\d{2})\-(\d{2})' +
                          r'.nc$' )
            
                r = re.match(regex, os.path.basename(f))
            
                assert( len(r.groups()) == 4 )
                t = [ int(match) for match in r.groups() ]

                # Found a file containing Year, Month, Day, Hour.
                # Now determine all the time steps in the file.

                self.__io.setFilename(f)
                years = self.__io.read('year').data
                times = self.__io.read('mtime').data
                assert( len(years) == len(times) )

                for i in range(len(years)):
                    # set year
                    d  = datetime.datetime( years[i],1,1 )
                    d += datetime.timedelta( times[i][0] - 1, # month & day of year
                                             times[i][1]*60.0*60.0 + # hour of day
                                             times[i][2]*60.0 # minute of day.
                                             )
                    self.__timeRange.append( d )
                                
        return self.__timeRange
        

    def __setTimeValue(self, time):
        """
        Select a particular time slice.  Use getTimeRange() to see all
        available data.
        """
        # Check the input:
        assert( isinstance(time, datetime.datetime) )
        
        # Find the index corresponding to this time value
        filename  = os.path.join(self.runPath, self.runName)
        filename += ( '_sech_tie_%04d-%02d-%02dT%02d.nc' % (time.year, time.month, time.day,
                                                            time.hour) )
        
        # This will raise an exception if the time slice is missing:        
        (self.__timeRange).index(time)
        
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

        FIXME: 'time' parameter does nothing.
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
        """

        self.__setTimeValue(time)
        # get scipy.io.netcdf object mapped to this variable
        (v, timeIdx) = self.__io.read(varName, time, start,count,stride)

        if time:
            if (v.dimensions[0] == 'time'):
                return v.data[timeIdx]
            else:
                return v.data
        else:
            return v.data
        
        
if (__name__ == '__main__'):
    l = TIEGCM('/home/schmitt/src/LTR-para/branches/sandbox/python/pyLTR', '')
    #l = TIEGCM('/home/schmitt/src/LTR-para/branches/sandbox/python/pyLTR', 'WHI-CMIT-final')
    print('Time Range:')
    print((l.getTimeRange()))
    print()    
    print('Attributes:')
    print((l.getAttributeNames()))
    #print l.readAttribute('model_version', datetime.datetime(2008,3,21,2,30,0))
    print()        
    print('Modified Julian Date:')
    #print l.readAttribute('mjd', datetime.datetime(1996, 5, 19, 16, 2))
    print()    
    print('Variables:')
    print((l.getVarNames()))
    print()
    print('Reading "UI_VEL"')
    var = l.read('UI_VEL', datetime.datetime(2008,3,21,2,30,0))
    print(('Type: ', type(var)))
    print(('Shape: ', str(var.shape)))
    print('fin.')
