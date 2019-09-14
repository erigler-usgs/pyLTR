import sys

from pyLTR.Models.Io import Io

import scipy.io

class NetcdfIo(Io):
    """
    NetCDF I/O.  Implemented for TIEGCM compatibility.  See TIEGCM.py
    for details (along with a long 'FIXME' that discusses refactoring
    this Python I/O wrapper class structure).
    """

    def __init__(self, filename):
        Io.__init__(self, filename)
        self.__keys = []
        self.__attributes = []
        self.__timeIndex = 0

#    def getTimeRange(self):
#        """
#        Returns a list of datetime objects corresponding to all the timesteps in the current file
#        """
#        raise NotImplementedError

#    def __setTimeValue(self, time):
#        """
#        Select a particular time slice.  Use getTimeRange() to see all
#        available data.
#        """
#        raise NotImplementedError
    


    def __getNetcdfObj(self):
        """
        Return an instance of scipy.io.netcdf_file for read
        """
        nc = scipy.io.netcdf_file(self.filename, 'r')

        return nc

    def getAttributeNames(self):
        """
        Returns a list of attributes available in the dataset.        
        """
        if not self.__attributes:
            nc = self.__getNetcdfObj()            
            self.__attributes = list(nc._attributes.keys())

        return self.__attributes

    def readAttribute(self, attrName, time):
        """
        Returns an attribute at a particular time.  Use an item in the
        lists returned by getAttributeNames() and getTimeRange() to
        obtain valid data.

        Note:  'time' parameter does nothing.
        """
        raise NotImplementedError
        nc = self.__getNetcdfObj()
        assert( attrName in nc._attributes )

        return nc._attributes[attrName]
        

    def getVarNames(self):
        """
        Returns a list of variables available in the dataset.    
        """
        if not self.__keys:
            nc = self.__getNetcdfObj()
            self.__keys = list(nc.variables.keys())

        return self.__keys
        

    def read(self, varName, time=None, start=None, count=None, stride=None):
        """
        Returns a scipy.io.netcdf_file variable (varName) at a particular time.  Use an
        item in the lists returned by getVarNames() and getTimeRange()
        to obtain valid data.

        FIXME:  hard-coded to support time indexing for TIEGCM data.
        """
        nc = self.__getNetcdfObj()
        assert( varName in nc.variables )
        v = nc.variables[varName]

        #FIXME: this is TIEGCM-specific!
        if time:
            # Get the requested day of year, hour and minute
            (doy, hour, minute) = [int(i) for i in time.strftime('%j %H %M').split() ]
            # Find the time index this maps to
            mtimes = self.read('mtime')

            for idx,value in enumerate(mtimes.data):
                if ((value[0] == doy) &
                    (value[1] == hour) &
                    (value[2] == minute)):
                    timeIdx = idx
                    break                

            return (v, timeIdx)                                                                    
        else:
            return v               
