import datetime
import numpy

import pyLTR
from .SolarWind import SolarWind

class LFM(SolarWind):
    """
    LFM Solar Wind file.  Data is in either GSM (10 columns) or SM (11
    columns) coordinates.
    """

    def __init__(self, filename = None):        
        SolarWind.__init__(self)

        self.startDate = None
        self.__read(filename)

    def __read(self, filename):
        """
        Read the solar wind file & store results in self.data TimeSeries object.
        """
        f = open(filename)

        self.startDate = self.__parseDate(f.readline())
        (nRows, nCols) = [int(s) for s in f.readline().split() ]

        dataArray = self.__readData(f, nRows, nCols)
        self.__storeDataDict(dataArray)
        self.__appendMetaData(filename)
        self._appendDerivedQuantities()

        
    def __readData(self, f, nRows, nCols):
        """
        Read & return 2d NumPy array of data from file
        """
        # Efficiently allocate all the memory we'll need.
        data = numpy.empty( (nCols, nRows), float )

        # Import  data from the LFM Solar Wind file
        rowIndex = 0
        for row in f.readlines():
            if len(row.split()) != nCols: continue

            for col, field in enumerate(row.split()):
                data[col, rowIndex] = field

            rowIndex += 1

        # Bad things can happen if the file header says there is more
        # (or less) data than there actually is within the file!
        assert(rowIndex == nRows)

        return data

    def __storeDataDict(self, dataArray):
        """
        Populte self.data TimeSeries object via the 2d dataArray read
        from file. Make sure to store data in GSM coordinates (the
        default used by pyLTR.SolarWind).
        """
        (nCols, nRows) = dataArray.shape

        keys = ['time_min', 'n', 'vx','vy','vz', 'cs', 'bx','by','bz','b']
        names = ['Time (Minutes since start)','Density','Vx (gsm)','Vy (gsm)','Vz (gsm)','Sound Speed','Bx (gsm)','By (gsm)','Bz (gsm)','B']
        units = ['min',r'$\mathrm{1/cm^3}$',
                   r'$\mathrm{km/s}$', r'$\mathrm{km/s}$', r'$\mathrm{km/s}$',
                   r'$\mathrm{km/s}$',
                   r'$\mathrm{nT}$',r'$\mathrm{nT}$',r'$\mathrm{nT}$',
                   r'$\mathrm{nT}$'
                   ]

        # Fill in the data dictionary
        for i, key in enumerate(keys):
            self.data.append(key, names[i], units[i], dataArray[i,:])

        # If a non-zero tilt angle is in the solar wind file, data is
        # stored in SM coordinates.
        if ( nCols > 10 ):
            if (not self.__isZeroEverywhere(dataArray[10,:])):
                keys.append('tilt')
                names.append('SM Tilt Angle')
                units.append('Rad')
                self.__sm2gsm(self.data)

    def __isZeroEverywhere(self, array):
        """
        Returns True if the input array is zero (smaller than machine
        precision) everywhere.  Useful for determining if tilt angle
        is zero everywhere (i.e. LFM file is in GSM coordinates).
        """
        epsilon = numpy.finfo( type(array[0]) ).eps
        boolList = numpy.less_equal(numpy.abs(array), epsilon)

        for b in boolList:
            if not b:
                return False
        return True


    def __sm2gsm(self, dataDict):
        """
        Transform all magnetic field B and velocity V values from SM
        to GSM coordinates.  Store results by overwriting dataDict
        contents.
        """

        b = (dataDict.getData('bx'),dataDict.getData('by'),dataDict.getData('bz'))
        v = (dataDict.getData('vx'),dataDict.getData('vy'),dataDict.getData('vz'))

        for i,time in enumerate(dataDict.getData('time_min')):
            d = self.startDate + datetime.timedelta(minutes=time)

            # Update magnetic field
            b_gsm = pyLTR.transform.SMtoGSM(b[0][i], b[1][i], b[2][i], d)
            
            dataDict.setData('bx', b_gsm[0], i)
            dataDict.setData('by', b_gsm[1], i)
            dataDict.setData('bz', b_gsm[2], i)

            # Update Velocity
            v_gsm = pyLTR.transform.SMtoGSM(v[0][i], v[1][i], v[2][i], d)

            dataDict.setData('vx', v_gsm[0], i)
            dataDict.setData('vy', v_gsm[1], i)
            dataDict.setData('vz', v_gsm[2], i)

    def __appendMetaData(self, filename):
        """
        Add standard metadata to the data dictionary.
        """
        metadata = {'Model': 'LFM',
                    'Source': filename,
                    'Date processed': datetime.datetime.now(),
                    'Start date': self.startDate
                    }
        
        self.data.append(key='meta',
                         name='Metadata for LFM Solar Wind file',
                         units='n/a',
                         data=metadata)
                                 
    
    def __parseDate(self, dateStr):
        """
        Convert from [year, doy, hour, minte] to datetime object

        >>> sw = LFM('examples/data/solarWind/LFM_SW-SM-DAT')
        >>> sw._LFM__parseDate('1995 80 0 1')
        datetime.datetime(1995, 3, 21, 0, 1)
        """
        fields = [int(s) for s in dateStr.split() ]

        date = ( datetime.datetime(year=fields[0], month=1, day=1,
                                   hour=fields[2], minute=fields[3]) +
                 datetime.timedelta(fields[1] - 1) )

        return date
    

if __name__ == '__main__':
    import doctest
    doctest.testmod()
