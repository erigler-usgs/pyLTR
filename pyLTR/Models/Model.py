class Model(object):
    """
    Abstract base class for Model I/O

    See rms.py for an example use of the Model class.
    """

    def __init__(self, runPath, runName):
        self.runPath = runPath.strip()
        self.runName = runName.strip()

    def getTimeRange(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available
        """
        raise NotImplementedError

    def __setTimeValue(self, time):
        """
        Select a particular time slice.  Use getTimeRange() to see all
        available data.
        """
        raise NotImplementedError

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

    def read(self, varName, time):
        """
        Returns a variable (varName) at a particular time.  Use an
        item in the lists returned by getVarNames() and getTimeRange()
        to obtain valid data.
        """
        raise NotImplementedError
