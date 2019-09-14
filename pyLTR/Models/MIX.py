from pyLTR.Models import Model
from pyLTR.Models.Hdf4Io import Hdf4Io

import datetime
import glob
import os
import re

class MIX(Model):
    """
    Implementation class for MIX I/O.  Typical variables used are:
      - 'Grid X', 'Grid Y'
      - 'Potential North [V]', 'Potential South [V]'
      - 'Hall conductance North [S]', 'Hall conductance South [S]'
      - 'Number flux North [1/cm^2 s]', 'Number flux South [1/cm^2 s]'      
      - 'Neutral wind speed North [m/s]', 'Neutral wind speed South [m/s]'           
      - 'Average energy North [keV]', 'Average energy South [keV]'            
      - 'Pedersen conductance North [S]', 'Pedersen conductance South [S]'
      - 'FAC North [A/m^2]'], 'FAC South [A/m^2]'
    """

    def __init__(self, runPath, runName):
        """
        Parameters:
          runPath: path to directory containing MIX data files.
          runName: Optional parameter.  When specified, it searches
          runPath/runName* for files.  This is useful for single
          directories containing multiple runs.
        """
        Model.__init__(self, runPath, runName)

        filePrefix = os.path.join(self.runPath, self.runName)
        self.__fileList = glob.glob(filePrefix + '*_mix_*.hdf')
        self.__fileList.sort()

        if not self.runName:
            regex = (r'^([\S\-]+)' + # \S is "digits, letters and underscore".
                     r'_mix_\d{4}\-\d{2}\-\d{2}T\d{2}\-\d{2}\-\d{2}Z.hdf$')
            r = re.match(regex, os.path.basename(self.__fileList[0]))
            if len(r.groups()) < 1:
                raise Exception('Could not determine run name.  Are you looking at the correct directory?')
            self.runName = r.groups()[0]

        self.__io = Hdf4Io( self.__fileList[0] )

        self.__timeRange = []
        self.getTimeRange()


    def getTimeRange(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available.
        
        MIX has files named like
        '[runName]_mix_1995-03-21T16-00-00Z.hdf', one time step per
        file.
        """
        if not self.__timeRange:
            
            for f in self.__fileList:
                regex = ( r'^' + self.runName + '_mix_' +
                          r'(\d{4})\-(\d{2})\-(\d{2})' +
                          r'T' +
                          r'(\d{2})\-(\d{2})-(\d{2})' +
                          r'Z.hdf$' )
            
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
        

    def __setTimeValue(self, time):
        """
        Select a particular time slice.  Use getTimeRange() to see all
        available data.
        """
        # Check the input:
        assert( isinstance(time, datetime.datetime) )
        
        # Find the index corresponding to this time value
        filename  = os.path.join(self.runPath, self.runName)
        filename += ( '_mix_%04d-%02d-%02dT%02d-%02d-%02dZ.hdf' % (time.year, time.month, time.day,
                                                                   time.hour, time.minute, time.second) )

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
    l = MIX('/hao/aim2/schmitt/data/July_Merge/LR_CMIT_merge/May_19-20_1996', 'CMIT')
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


