import datetime
import re, os, sys, pickle
import numpy as n

from pyLTR.TimeSeries import TimeSeries

class cdaweb():
    """
    Class designed to import an arbirtary CDAWeb ASCII file and create a pyLTR
    TimeSeries Object
    """
    def __init__(self,fileName):
        """
        Requires name of CDAWeb ASCII file
        """
        self.data=TimeSeries()
        self.dateRegEx = r=re.compile(r'^\d{2}\-\d{2}-\d{4} \d{2}\:\d{2}\:\d{2}\.\d{3}')
        if os.path.exists(fileName):
            self.__read(fileName)
        else:
            print("Error: the file does not exist")
            sys.exit(1)
            
    def __read(self,fileName):
        """
        Read in the CDAWeb ASCII file and create TimeSeries Object stored in .data
        """
        fh=open(fileName)
        (self.__labels,self.__units) = self.__parseMetadata(fh)
        (self.__dates, self.__data) = self.__readData(fh)
        self.__dataArray = n.array(self.__data,n.float)
        self.__storeDataDict(self.__dates,self.__dataArray)
        
    def __parseMetadata(self, fh):
        # Eat headers (Yum!)
        while True:
            if 'RECORD VARYING VARIABLES' in fh.readline():
                fh.readline()
                fh.readline()
                break
        
        (labels, units) = self.__getHeaders(fh)
        return (labels,units)
  
    def __getHeaders(self, fh):
        # Get the list of variables
        self.__variables = []
        while True:
            line = fh.readline().strip('#').strip()
            # Read until blank line
            if not line:
                break
            self.__variables.append(line)

        labels = fh.readline().strip().split()
        # some files had (_@_x) line or others before the units so skip them
        while (True):
            line = fh.readline().strip().split()
            if 'dd-mm-yyyy' in line[0]:
                units = line[1:]
                break
        if (len(labels) != len(units)):
            for i in n.arange(n.abs(len(labels)-len(units))):
                units.append(' ')
        return (labels, units)
        
    def __readData(self, fh):
        """
        Read & return 2d array (of strings) containing data from file
        """
        dates = []
        rows = []
        for line in fh:
            if self.__isBadData(line):
                continue
            dates.append( self.__parseDate(line.split()[0], line.split()[1]) )
            rows.append( line.split()[2:] )

        return (dates, rows)      
       
    def __parseDate(self, dateStr, timeStr):
        """
        Convert from [year, doy, hour, minte] to datetime object

        >>> sw = OMNI('examples/data/solarWind/OMNI_HRO_1MIN_14180.txt1')
        >>> sw._OMNI__parseDate('20-03-2008', '04:03:00.000')
        datetime.datetime(2008, 3, 20, 4, 3)
        """
        date = [int(s) for s in dateStr.strip().split('-')]
        # Note: Double type cast required for converting '00.000' into an integer
        #time = [int(float(s)) for s in timeStr.strip().split(':')]
        timeSplit = timeStr.strip().split(':')
        hour = int(timeSplit[0])
        minute = int(timeSplit[1])
        secondSplit = timeSplit[2].split('.')
        second = int(secondSplit[0])
        if (len(secondSplit) == 2):
            microsecond = int(secondSplit[1])*10**3
        elif (len(secondSplit) ==3):
            microsecond = int(secondSplit[1])*10**3 + int(secondSplit[2])
        else:
            microsecond = 0
        return datetime.datetime(year=date[2], month=date[1], day=date[0],
                                 hour=hour, minute=minute, second=second,
                                 microsecond=microsecond)
                                 
    def __isBadData(self, str):
        """ Returns True if str contains bad data."""
        # Valid data must begin with a date & time for example, the
        # last line or two may contain strings describing metadata.
        if not self.dateRegEx.match(str):
            return True
        
        return False
    
    def __storeDataDict(self,dates,dataArray):
        """
        Populate self.data TimeSeries object via the 2d dataArray read from file.
        """
        (rows,cols)=dataArray.shape
        for i in range(cols):
            self.data.append(self.__labels[i+1],self.__labels[i+1],self.__units[i+1],dataArray[:,i])
        self.data.append('time','Date and Time','Date/Time',dates)    
        return
        
    def Writer(self,fileName):
        pklFilename = fileName + '.pkl'
        print(('Saving "%s"' % pklFilename))
        fh = open(pklFilename, 'wb')
        pickle.dump(self.data, fh, protocol=2)
        fh.close()
        

