from pyLTR.Models import Model
from pyLTR.Models.Hdf4Io import Hdf4Io
from pyLTR.Models.Hdf5Io import Hdf5Io

import datetime
import glob
import os
import re

class LFMION(Model):
    """
    Implementation class for LFMION I/O.  Typical variables used are:
      - 'x_interp','y_interp'
      - 'potnorth','potsouth'
      - 'SigmaH_north','SigmaH_south'
      - 'fluxnorth','fluxsouth'
      - 'avE_north','avE_south'
      - 'SigmaP_north','SigmaP_south'
      - 'curnorth','cursouth'
    """

    def __init__(self, runPath, runName,ext='.hdf'):
        """
        Parameters:
          runPath: path to directory containing LFMION data files.
          runName: Optional parameter.  When specified, it searches
          runPath/runName* for files.  This is useful for single
          directories containing multiple runs.
        """
        Model.__init__(self, runPath, runName)

        filePrefix = os.path.join(self.runPath, self.runName)
        self.__fileList = glob.glob(filePrefix + '*_ion_*'+ext)
        self.__fileList.sort()

        #
        # regex is used to determine filename convention (eg. UT
        # '2012-07-13T02-15-00' vs Timestep '0008000') and to obtain
        # runName if it is undefined.
        # 
        regex = (r'^([\S\-]+)' + # \S is "digits, letters and underscore".
                 r'_ion_' + 
                 r'('
                 r'\d{4}\-\d{2}\-\d{2}T\d{2}\-\d{2}\-\d{2}Z' +  # UT-based filename
                 r'|' + 
                 r'\d{7}'  # timestep-based filename
                 r')' + 
                 ext +                    
                 r'$')
        r = re.match(regex, os.path.basename(self.__fileList[0]))
        if (len(r.groups()) != 2):
            raise Exception('Having trouble identifying LFM ION files.  Are you looking at the correct directory?')
        
        if not self.runName:
            self.runName = r.groups()[0]
        
        if len(r.groups()[1]) == 7:
            # Timestep filenames (eg. 'RunName_ion_0008000.hdf')
            self.__timestepFilenames = True
        else:
            # UT-based filenames (eg. 'RunName_ion_2012-07-13T02-47-00Z.hdf')
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
        '[runName]_ion_1995-03-21T16-00-00Z.hdf', one time step per
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
        '[runName]_ion_1995-03-21T16-00-00Z.hdf', one time step per
        file.
        """        
        for f in self.__fileList:
            regex = ( r'^' + self.runName + '_ion_' +
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
        '[runName]_ion_069000.hdf',when in step dump mode.
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
        """

        self.__setTimeValue(time)
        return self.__io.read(varName, None, start,count,stride)


if (__name__ == '__main__'):
    l = LFMION('/hao/aim2/schmitt/data/July_Merge/LR_CMIT_merge/May_19-20_1996', 'CMIT')
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
    #var = l.read('rho_', datetime.datetime(1996,5,19,16,0,0))


