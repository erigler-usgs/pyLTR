# Custom
import pyLTR
from .SolarWind import SolarWind

# 3rd party
import numpy

# Standard
import datetime
import re

class OMNI(SolarWind):
    """
    OMNI Solar Wind file from CDAweb [http://cdaweb.gsfc.nasa.gov/].
    Data stored in GSE coordinates.
    """

    def __init__(self, filename = None):        
        SolarWind.__init__(self)

        self.bad_data = ['-999.900', 
                         '0E+31', 
                         '99999.9', # V
                         '9999.99', # B
                         '999.990', # density
                         '1.00000E+07' # Temperature
                         ]
        # Match the beginning of a line with a date & time (i.e. 16-04-2008 00:00:00.000)
        self.dateRegEx = r=re.compile(r'^\d{2}\-\d{2}-\d{4} \d{2}\:\d{2}\:\d{2}\.\d{3}')

        self.__read(filename)

    def __read(self, filename):
        """
        Read the solar wind file & store results in self.data TimeSeries object.
        """
        fh = open(filename)

        startDate = self.__parseMetadata(fh)

        (dates, data) = self.__readData(fh, startDate)
        (dataArray, hasBeenInterpolated) = self.__removeBadData(data)
        (dataArray, hasBeenInterpolated) = self.__coarseFilter(dataArray, hasBeenInterpolated)
        self.__storeDataDict(dates, dataArray, hasBeenInterpolated)
        self.__appendMetaData(startDate, filename)
        self._appendDerivedQuantities()

        
    def __readData(self, fh, startDate):
        """
        Read & return 2d array (of strings) containing data from file
        """
        dates = []
        rows = []
        for line in fh:
            # Skip bad data rows
            if self.__isBadData(line):
                continue
            
            date = self.__deltaMinutes(line.split()[0], line.split()[1], startDate)
            data = line.split()[2:]
            
            dates.append( self.__parseDate(line.split()[0], line.split()[1]) )
            rows.append( [date]+data )

        return (dates, rows)

    def __removeBadData(self, data):
        """
        Linearly interpolate over bad data (defined by self.bad_data
        list) for each variable in dataStrs.
        
        data: 2d list.  Each row is a list containing:
          [float(nMinutes), str(Bx), str(By), str(Bz), str(Vx), str(Vy), str(Vz), str(rho), str(temp)]

        Returns:
          data: interpolated floating-point numpy array
          hasBeenInterpolated: 2d array that identifies if bad values were removed/interpolated.

        NOTE: This is remarkably similar to __coarseFilter!
          Refactoring to keep it DRY wouldn't be a bad idea. . .
        """
        assert( len(data[0]) == 9 )

        hasBeenInterpolated = numpy.empty((len(data), 8))
        hasBeenInterpolated.fill(False)

        for varIdx in range(1,9):

            lastValidIndex = -1
            for curIndex,row in enumerate(data):
                if row[varIdx] in self.bad_data:
                    # This item has bad data.
                    hasBeenInterpolated[curIndex, varIdx-1] = True
                    if (lastValidIndex == -1) & (curIndex == len(data)-1):
                        # Data must have at least one valid element!
                        raise Exception("First & Last datapoint(s) in OMNI "+
                                          "solar wind file are invalid.  Not sure "+
                                          "how to interpolate across bad data.")
                    elif (curIndex == len(data)-1):
                        # Clamp last bad data to previous known good data.
                        data[curIndex][varIdx] = data[lastValidIndex][varIdx]
                    else:
                        # Note the bad data & skip this element for now.
                        # We will linearly interpolate between valid data
                        continue

                # At this point, curIndex has good data.
                if (lastValidIndex+1) == curIndex:
                    # Set current element containing good data.
                    data[curIndex][varIdx] = float( row[varIdx] )
                else:
                    # If first index is invalid, clamp to first good value.
                    if lastValidIndex == -1:
                        lastValidIndex = 0
                        data[lastValidIndex][varIdx] = data[curIndex][varIdx]

                    # Linearly interpolate over bad data.
                    interpolated = numpy.interp(list(range(lastValidIndex, curIndex)), # x-coords of interpolated values
                                                [lastValidIndex, curIndex],  # x-coords of data.
                                                [float(data[lastValidIndex][varIdx]), float(data[curIndex][varIdx])]) # y-coords of data.
                    # Store the results.
                    for j,val in enumerate(interpolated):
                        data[lastValidIndex+j][varIdx] = val
                lastValidIndex = curIndex

        return (numpy.array(data, numpy.float), hasBeenInterpolated)

    def __coarseFilter(self, dataArray, hasBeenInterpolated):
        """
         Use coarse noise filtering to remove values outside 3
         deviations from mean of all values in the plotted time
         interval.

         Parameters:

           dataArray: 2d numpy array.  Each row is a list
             containing [nMinutes, Bx, By, Bz, Vx, Vy, Vz, rho, temp]

           hasBeenInterpolated: 2d boolean list.  Each row is a list
             of boolean values denoting whether dataArray[:,1:9] was
             derived/interpolated from the raw data (ie. bad points
             removed).

         Output:
           dataArray:  same structure as input array with bad elements removed
           hasBeenInterpolated: same as input array with interpolated values stored.

        NOTE: This is remarkably similar to __removeBadData!
          Refactoring to keep it DRY wouldn't be a bad idea. . .
        """
        
        stds = []
        means = []
        for varIdx in range(1,9):
            stds.append( dataArray[:,varIdx].std() )
            means.append( dataArray[:,varIdx].mean() )
            
            # Linearly interpolate over data that exceeds 3 standard
            # deviations from the mean
            lastValidIndex = -1
            for curIndex,row in enumerate(dataArray):
                # Are we outside 3 sigma from mean?
                if abs(means[varIdx-1] - row[varIdx]) > 3*stds[varIdx-1]:
                    hasBeenInterpolated[curIndex, varIdx-1] = True
                    if (curIndex == len(dataArray)-1):
                        # Clamp last bad data to previous known good data.
                        dataArray[curIndex][varIdx] = dataArray[lastValidIndex][varIdx]
                    else:
                        # Note the bad data & skip this element for now.
                        # We will linearly interpolate between valid data
                        continue

                if (lastValidIndex+1) != curIndex:
                    # If first index is invalid, clamp to first good value.
                    if lastValidIndex == -1:
                        lastValidIndex = 0
                        dataArray[lastValidIndex][varIdx] = dataArray[curIndex][varIdx]

                    # Linearly interpolate over bad data.
                    interpolated = numpy.interp(list(range(lastValidIndex, curIndex)), # x-coords of interpolated values
                                                [lastValidIndex, curIndex],  # x-coords of data.
                                                [float(dataArray[lastValidIndex][varIdx]), float(dataArray[curIndex][varIdx])]) # y-coords of data.
                    # Store the results.
                    for j,val in enumerate(interpolated):
                        dataArray[lastValidIndex+j][varIdx] = val
                lastValidIndex = curIndex

        return (dataArray, hasBeenInterpolated)

    def __storeDataDict(self, dates, dataArray, hasBeenInterpolated):
        """
        Populate self.data TimeSeries object via the 2d dataArray read from file.
        """
        self.__gse2gsm(dates, dataArray)

        self.data.append('time_min', 'Time (Minutes since start)', 'min', dataArray[:,0])

        # Magnetic field
        self.data.append('bx', 'Bx (gsm)', r'$\mathrm{nT}$', dataArray[:,1])
        self.data.append('isBxInterped', 'Is index i of By interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,0])
        
        self.data.append('by', 'By (gsm)', r'$\mathrm{nT}$', dataArray[:,2])
        self.data.append('isByInterped', 'Is index i of By interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,1])

        self.data.append('bz', 'Bz (gsm)', r'$\mathrm{nT}$', dataArray[:,3])
        self.data.append('isBzInterped', 'Is index i of Bz interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,2])

        # Velocity
        self.data.append('vx', 'Vx (gsm)', r'$\mathrm{km/s}$', dataArray[:,4])
        self.data.append('isVxInterped', 'Is index i of Vx interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,3])

        self.data.append('vy', 'Vy (gsm)', r'$\mathrm{km/s}$', dataArray[:,5])
        self.data.append('isVyInterped', 'Is index i of Vy interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,4])

        self.data.append('vz', 'Vz (gsm)', r'$\mathrm{km/s}$', dataArray[:,6])
        self.data.append('isVzInterped', 'Is index i of Vz interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,5])

        # Density
        self.data.append('n', 'Density', r'$\mathrm{1/cm^3}$', dataArray[:,7])
        self.data.append('isNInterped', 'Is index i of N interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,6])

        # Temperature
        self.data.append('t', 'Temperature', r'$\mathrm{kK}$', dataArray[:,8]*1e-3)
        self.data.append('isTInterped', 'Is index i of T interpolated from bad data?', r'$\mathrm{boolean}$', hasBeenInterpolated[:,7])

        
    def __appendMetaData(self, date, filename):
        """
        Add standard metadata to the data dictionary.
        """
        metadata = {'Model': 'OMNI',
                    'Source': filename,
                    'Date processed': datetime.datetime.now(),
                    'Start date': date
                    }
        
        self.data.append(key='meta',
                         name='Metadata for OMNI Solar Wind file',
                         units='n/a',
                         data=metadata)
                                 

    
    def __parseDate(self, dateStr, timeStr):
        """
        Convert from [year, doy, hour, minte] to datetime object

        >>> sw = OMNI('examples/data/solarWind/OMNI_HRO_1MIN_14180.txt1')
        >>> sw._OMNI__parseDate('20-03-2008', '04:03:00.000')
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
            line = fh.readline().strip('#').strip()
            # Read until blank line
            if not line:
                break
            variables.append(line)

        labels = fh.readline().strip().split()
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
        # Valid data must begin with a date & time for example, the
        # last line or two may contain strings describing metadata.
        if not self.dateRegEx.match(str):
            return True
        
        return False

    def __gse2gsm(self, dates, dataArray):
        """
        Transform magnetic field B and velocity V from GSE to GSM
        coordinates.  Store results by overwriting dataArray contents.
        """
        for i,data in enumerate(dataArray):
            d = dates[i]

            # Update magnetic field
            b_gsm = pyLTR.transform.GSEtoGSM(data[1], data[2], data[3], d)        
            data[1] = b_gsm[0]
            data[2] = b_gsm[1]
            data[3] = b_gsm[2]

            # Update Velocity
            v_gsm = pyLTR.transform.GSEtoGSM(data[4], data[5], data[6], d)
            data[4] = v_gsm[0]
            data[5] = v_gsm[1]
            data[6] = v_gsm[2]

        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
