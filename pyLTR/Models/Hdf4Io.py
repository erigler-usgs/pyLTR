import sys

from pyLTR.Models.Io import Io

try:
    from pyhdf.SD import SD, SDC, HDF4Error
except ImportError:
    # mock hdf class
    import numpy
    class SDC(object):
        def READ(self):
            pass
    class dataset(object):
        def get(self, base, bound, stride):
            print((SD(None, None).error))
            return numpy.empty(1)
            
    class SD(object):
        def __init__(self, filename, access_mode):
            self.error = 'pyHDF not installed!  Strange things will happen.'
            print((self.error))
        def attributes(self):
            print((self.error))
            return {}
        def datasets(self):
            print((self.error))
            return {}
        def select(self, variable_name):
            print((self.error))
            return dataset()
    class HDF4Error(Exception):
        pass
        

class Hdf4Io(Io):
    """
    HDF4 I/O.
    
    Note: Assumes only one timestep per file!
    """

    def __init__(self, filename):
        Io.__init__(self, filename)
        self.__keys = []
        self.__attributes = []

#    def getTimeRange(self):
#        """
#        Returns a list of datetime objects corresponding to all the timesteps in the current file
#        """
#        raise NotImplementedError

#    def __setTimeValue(self):
#        """
#        Select a particular time slice.  Use getTimeRange() to see all
#        available data.
#        """
#        raise NotImplementedError
#

    def __getHdfObj(self):
        """
        Return an insance of pyhdf.SD for read
        """
        try:
            hdf = SD(self.filename, SDC.READ)
        except HDF4Error as msg:
            sys.stderr.write('HDF4Error opening "' + self.filename +
                             '" for read.  Error message follows:' + str(msg) )
            sys.stderr.write('Trying operation again.')
            sys.stderr.flush()

            # Maybe the I/O was busy.  Sleep for a bit & try again.
            import time
            time.sleep(1)

            hdf = SD(self.filename, SDC.READ)

        return hdf

    def getAttributeNames(self):
        """
        Returns a list of attributes available in the dataset.        
        """
        if not self.__attributes:
            hdf = self.__getHdfObj()            
            self.__attributes = list(hdf.attributes().keys())

        return self.__attributes

    def readAttribute(self, attrName, time):
        """
        Returns an attribute at a particular time.  Use an item in the
        lists returned by getAttributeNames() and getTimeRange() to
        obtain valid data.
        """
        hdf = self.__getHdfObj()        
        assert( attrName in hdf.attributes() )

        
        return hdf.attributes()[attrName]
        

    def getVarNames(self):
        """
        Returns a list of variables available in the dataset.    
        """
        if not self.__keys:
            hdf = self.__getHdfObj()
            self.__keys = list(hdf.datasets().keys())

        return self.__keys
        

    def read(self, varName, time, start=None, count=None, stride=None):
        """
        Returns a variable (varName) at a particular time.  Use an
        item in the lists returned by getVarNames() and getTimeRange()
        to obtain valid data.
        """
        hdf = self.__getHdfObj()
        return hdf.select(varName).get(start,count,stride)

