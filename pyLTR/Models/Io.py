class Io(object):
    """
    Abstract base class for Model I/O
    """

    def __init__(self, filename):
        self.setFilename(filename)

    def setFilename(self, filename):
        self.filename = filename

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
    def getAttributeNames(self):
        """
        Returns a list of attributes available in the dataset.    
        """
        raise NotImplementedError

    def readAttribute(self, attrName, time):
        """
        Returns an attribute at a particular time.  Use an item in the
        lists returned by getAttributeNames() and getTimeRange() to
        obtain valid data.
        """
        raise NotImplementedError
    
    def getVarNames(self):
        """
        Returns a list of variables available in the dataset.    
        """
        raise NotImplementedError

    def read(self, varName):
        """
        Returns a variable (varName) at a particular time.  Use an
        item in the lists returned by getVarNames() and getTimeRange()
        to obtain valid data.
        """
        raise NotImplementedError
