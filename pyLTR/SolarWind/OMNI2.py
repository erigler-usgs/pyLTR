# Custom
import pyLTR
from .SolarWind import SolarWind

# 3rd party
import numpy

# Standard
import datetime
import re

class OMNI2(SolarWind):
    """
    OMNI Solar Wind file from SPDF OMNIWeb Plus Service.
    Data stored in GSE coordinates.
    """

    def __init__(self, filename = None):        
        SolarWind.__init__(self)

        self.bad_data = ['-999.900', 
                         '0E+31', 
                         '99999.9', # V
                         '9999.99', # B
                         '999.99', # density
                         '9999999.' # Temperature
                         ]
        # Match the beginning of a line with a date & time (i.e. 16-04-2008 00:00:00.000)
        self.dateRegEx=re.compile(r'^\d{4}\s+\d+\s+\d+\s+\d+')
        self.__read(filename)

    def __read(self, filename):
        """
        Read the solar wind file & store results in self.data TimeSeries object.
        """
        fh = open(filename)

        (dates, data) = self.__readData(fh)
        startDate = dates[0]
        (dataArray, hasBeenInterpolated) = self.__removeBadData(data)
        (dataArray, hasBeenInterpolated) = self.__coarseFilter(dataArray, hasBeenInterpolated)
        self.__storeDataDict(dates, dataArray, hasBeenInterpolated)
        self.__appendMetaData(startDate, filename)
        self._appendDerivedQuantities()

        
    def __readData(self, fh):
        """
        Read & return 2d array (of strings) containing data from file
        """
        dates = []
        rows = []
        index = 0
        for line in fh:
            # Skip bad data rows
            if self.__isBadData(line):
                print('Bad Data Found')
                continue
            
            (yr, doy, hr, mn, data) = (int(line.split()[0]),int(line.split()[1]),
                                      int(line.split()[2]),int(line.split()[3]),
                                      line.split()[4:])
 
            date = datetime.datetime(yr,1,1) + datetime.timedelta(doy) + datetime.timedelta(hours=hr) + datetime.timedelta(minutes=mn) - datetime.timedelta(1)
            dates.append( date )
            diff = (dates[index] - dates[0])
            rows.append( [diff.total_seconds()/60.0]+data )
            index = index + 1
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
