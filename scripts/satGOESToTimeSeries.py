#!/usr/bin/env python
"""
Converts GOES magnetic field data (in GSM coordinates) to a pyLTR TimeSeries object.

Execute `./satGOESToTimeSeries.py --help` for more information.
"""

# Custom
import pyLTR

# 3rd party
import numpy

# Standard
import datetime
import pickle
import optparse
import os.path
import re
import sys

class GOES(object):
    """
    GOES Satellite Wind file from CDAweb [http://cdaweb.gsfc.nasa.gov/].
    Data stored in GSM coordinates.
    """

    def __init__(self, filename = None):        

        self.data = pyLTR.TimeSeries()
        self.bad_data = ['-1.00000E+31']
        # Match the beginning of a line with a date & time (i.e. 16-04-2008 00:00:00.000)
        self.dateRegEx = r=re.compile(r'^\d{2}\-\d{2}-\d{4} \d{2}\:\d{2}\:\d{2}\.\d{3}')        

        self.__read(filename)
        self.__appendDerivedQuantities()

    def __read(self, filename):
        """
        Read the solar wind file & store results in self.data TimeSeries object.
        """
        fh = open(filename)

        startDate = self.__parseMetadata(fh)

        (dates, dataArray) = self.__readData(fh, startDate)
        self.__storeDataDict(dates, dataArray)
        self.__appendMetaData(startDate, filename)

        
    def __readData(self, fh, startDate):
        """
        Read & return 2d NumPy array of data from file
        """
        dates = []
        rows = []
        for line in fh:
            # Skip bad data rows
            if self.__isBadData(line):
                continue
            date = self.__deltaMinutes(line.split()[0], line.split()[1], startDate)
            data = [ float(s) for s in line.split()[2:] ]
            
            dates.append( self.__parseDate(line.split()[0], line.split()[1]) )
            rows.append( [date]+data )

        return (dates, numpy.array(rows, numpy.float))
        

    def __storeDataDict(self, dates, dataArray):
        """
        Populate self.data TimeSeries object via the 2d dataArray read from file.
        """

        self.data.append('time_min', 'Time (Minutes since start)', 'min', dataArray[:,0])
        self.data.append('bx', 'Bx (gsm)', r'$\mathrm{nT}$', dataArray[:,1])
        self.data.append('by', 'By (gsm)', r'$\mathrm{nT}$', dataArray[:,2])
        self.data.append('bz', 'Bz (gsm)', r'$\mathrm{nT}$', dataArray[:,3])

        
    def __appendMetaData(self, date, filename):
        """
        Add standard metadata to the data dictionary.
        """
        metadata = {'Model': 'GOES',
                    'Source': filename,
                    'Date processed': datetime.datetime.now(),
                    'Start date': date
                    }
        
        self.data.append(key='meta',
                         name='Metadata for GOES file',
                         units='n/a',
                         data=metadata)
                                 

    
    def __parseDate(self, dateStr, timeStr):
        """
        Convert from [year, doy, hour, minte] to datetime object

        >>> sw = GOES('examples/data/solarWind/GOES_HRO_1MIN_14180.txt1')
        >>> sw._GOES__parseDate('20-03-2008', '04:03:00.000')
        datetime.datetime(2008, 3, 20, 4, 3)
        """
        date = [int(s) for s in dateStr.strip().split('-')]
        # Note: Double type cast required for converting '00.000' into an integer
        time = [int(float(s)) for s in timeStr.strip().split(':')]

        return datetime.datetime(year=date[2], month=date[1], day=date[0],
                                 hour=time[0], minute=time[1], second=time[2])

    def __deltaMinutes(self, dateStr, timeStr, startDate):
        """
        Parameters:
          date: date string YYYY-MM-DD
          time: Time string HH:MM
          startDate: datetime object
        Returns: Number of minutes elapsed between (date, time) and startDate.
        """
        curDate = self.__parseDate(dateStr, timeStr)
        diff = curDate - startDate

        return (diff.days*24.0*60.0 + diff.seconds/60.0)

    def __parseMetadata(self, fh):
        # Eat headers (Yum!)
        while True:
            if 'RECORD VARYING VARIABLES' in fh.readline():
                fh.readline()
                fh.readline()
                break
        
        (labels, units) = self.__getHeaders(fh)
        startDate = self.__getStartDate(fh)

        return startDate

    def __getHeaders(self, fh):
        # Get the list of variables
        variables = []
        while True:
            line = fh.readline().strip()
            # Read until blank line
            if not line:
                break
            variables.append(line)

        labels = fh.readline().strip().split()
        fh.readline() # read (@_x_) line
        units = fh.readline().strip().split()

        return (labels, units)

    def __getStartDate(self, fh):
        pos = fh.tell()
        
        line = fh.readline()
        dateTime = self.__parseDate(line.split()[0], line.split()[1])
        
        fh.seek(pos)
        
        return dateTime        


    def __isBadData(self, str):
        """ Returns True if str contains bad data."""
        # First definition of bad data:  self.bad_data set in constructor
        for bad in self.bad_data:
            if bad in str:
                return True

        # 2nd definition of bad data:  Valid data must begin with a date & time
        # for example, the last line or two may contain strings describing metadata.
        if not self.dateRegEx.match(str):
            return True
        
        return False

    def __appendDerivedQuantities(self):
        """Calculate & append standard derived quantities to the data dictionary """    

        # --- Magnetic Field Magnitude
        if 'b' not in self.data:
            b = numpy.sqrt(self.data.getData('bx')**2 +
                           self.data.getData('by')**2 +
                           self.data.getData('bz')**2)
            self.data.append('b', 'Magnitude of Magnetic Field', r'$\mathrm{nT}$', b)
        # --- Hours since start
        if 'time_hr' not in self.data:
            hr = self.data.getData('time_min')/60.0
            self.data.append('time_hr', 'Time (hours since start)', r'$\mathrm{hour}$', hr)

        # --- datetime
        if 'time' not in self.data:
            time = []
            for minute in self.data.getData('time_min'):
                time.append( self.data.getData('meta')['Start date'] + datetime.timedelta(minutes=minute) )
            self.data.append('time', 'Date and Time', r'$\mathrm{Date/Time}$', time)
        # --- Compute & store day of year
        if 'doy' not in self.data:
            doy = []
            for dt in self.data.getData('time'):
                tt = dt.timetuple()
                dayFraction = (tt.tm_hour+tt.tm_min/60.+tt.tm_sec/(60.*60.))/24.
                doy.append( float(tt.tm_yday) + dayFraction )
            self.data.append('time_doy', 'Day of Year', r'$\mathrm{Day}$', doy)

def parseArgs():
    """
    Returns parameters used for GOES-to-pyLTR.TimeSeries conversion
      goesFile
    Execute `satGOESToTimeSeries.py --help` for details explaining these variables.
    """
    # additional optparse help available at:
    # http://docs.python.org/library/optparse.html
    # http://docs.python.org/lib/optparse-generating-help.html
    parser = optparse.OptionParser()

    parser.add_option('-f', '--filename', dest='filename',
                      default='LFM', metavar='MODEL',
                      help='Path to GOES satellite file.')

    parser.add_option('-a', '--about', dest='about', default=False, action='store_true',
                       help='About this program.')

    (options, args) = parser.parse_args()
    if options.about:
        print((sys.argv[0] + " version " + pyLTR.Release.version))
        print("")
        print("This script reads Bx,By,Bz (in GMS coordinates) from a GOES satellite")
        print("data file and saves the data to FILENAME.pkl as a pyLTR.TimeSeries")
        print("object using Python's Pickle module.")
        print("http://docs.python.org/library/pickle.html")
        print("")
        print("To load the pyLTR.TimeSeries object into memory, use Python's cPickle:")
        print("")
        print("      import cPickle")
        print("      fh = open(filename, 'rb')")
        print("      obj = cPickle.load(fh)")
        print("      fh.close()")
        print("")
        print("Note that pkl files may be machine-dependent thanks to big/little ")
        print("endian machines.")
                
        sys.exit()

    # Sanitize inputs
    assert( os.path.exists(options.filename) )

    return (options.filename)

 
if __name__ == '__main__':

    filename = parseArgs()
    
    g10=GOES(filename)

    (basename, extension) = os.path.splitext(filename)
    fh = open(basename + '.pkl','wb')
    pickle.dump(g10.data,fh,protocol=2)
    fh.close()

