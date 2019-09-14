import sys

from pyLTR.Models.Io import Io

try:
    import h5py
except ImportError:
    print("Failed to import h5py module.")
    print("HDF5 support not enabled.")


class Hdf5Io(Io):
    """
    HDF5 I/O.
    
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
        Return an insance of h5py object for read
        """
        hdf = h5py.File(self.filename,"r")

        return hdf

    def getAttributeNames(self):
        """
        Returns a list of attributes available in the dataset.        
        """
        if not self.__attributes:
            hdf = self.__getHdfObj()            
            self.__attributes = list(hdf.attrs.keys())

        return self.__attributes

    def readAttribute(self, attrName, time):
        """
        Returns an attribute at a particular time.  Use an item in the
        lists returned by getAttributeNames() and getTimeRange() to
        obtain valid data.
        """
        hdf = self.__getHdfObj()        
                
        return hdf.attrs.get(attrName)
        

    def getVarNames(self):
        """
        Returns a list of variables available in the dataset.    
        """
        if not self.__keys:
            hdf = self.__getHdfObj()
            self.__keys = list(hdf.keys())

        return self.__keys
        

    def read(self, varName, time, start=None, count=None, stride=None):
        """
        Returns a variable (varName) at a particular time.  Use an
        item in the lists returned by getVarNames() and getTimeRange()
        to obtain valid data.
        """
        hdf = self.__getHdfObj()
        data = hdf.get(varName)
        return data.value

