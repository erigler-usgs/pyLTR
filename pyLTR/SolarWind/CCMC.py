import datetime
import numpy

import pyLTR.transform
from pyLTR.TimeSeries import TimeSeries
from .SolarWind import SolarWind

class CCMC(SolarWind):
    """
    CCMC Solar Wind file.
    http://ccmc.gsfc.nasa.gov/
    """

    def __init__(self, filename = None):        
        SolarWind.__init__(self)
        self.startDate = None

        self.__read(filename)

    def __read(self, filename):
        """
        Read the solar wind file & store results in self.data TimeSeries object.
        """
        fh = open(filename)

        (labels, units) = self.__parseMetadata(fh)

        dataArray = self.__readData(fh)
        self.__storeDataDict(dataArray, labels, units)
        self.__appendMetaData(filename)
        self._appendDerivedQuantities()

        
    def __readData(self, fh, commentChar='#'):
        """
        Read & return 2d NumPy array of data from file
        """
        rows = []
        for line in fh:
            # Skip comments
            if line[0] == commentChar:
                continue
            rows.append( [ float(s) for s in line.split() ] )

        return numpy.array(rows, numpy.float)

    def __storeDataDict(self, dataDict, labels, units):
        """
        Populte self.data TimeSeries object via the data dict
        containing 2d arrays read from file.
        """
        dataDict = self.__array2dict(dataDict, labels, units)

        self.data.append('time_min', 'Time (Minutes since start)', 'min',
                         self.__day2min( dataDict.getData('Time') ) )

        self.data.append('n', 'Density', r'$\mathrm{1/cm^3}$',                        
                         dataDict.getData('N') )
                         
        
        (vx,vy,vz, bx,by,bz) = self.__applyCoordTransf(dataDict)
        self.data.append('vx', 'Vx (gsm)', r'$\mathrm{km/s}$', vx)
        self.data.append('vy', 'Vy (gsm)', r'$\mathrm{km/s}$', vy)
        self.data.append('vz', 'Vz (gsm)', r'$\mathrm{km/s}$', vz)
        self.data.append('v',   'V', r'$\mathrm{km/s}$',
                         dataDict.getData('V'))

        self.data.append('t', 'Temperature', r'$\mathrm{kK}$',
                         dataDict.getData('T')*1e-3)

        self.data.append('bx', 'Bx (gsm)', r'$\mathrm{nT}$', bx)
        self.data.append('by', 'By (gsm)', r'$\mathrm{nT}$', by)
        self.data.append('bz', 'Bz (gsm)', r'$\mathrm{nT}$', bz)
        self.data.append('b',  'B',  r'$\mathrm{nT}$',
                         dataDict.getData('B'))

        #self.data.append('tilt', 'Tilt angle', r'$\mathrm{radians}$',
        #                 self.__getTiltAngle(self.data.getData('time_min')))

    def __day2min(self, dayArray):
        """ Convert array of percentage of day elapsed since start date to array of minute in day """
        startPercentage = ( self.startDate.hour + self.startDate.minute/60. + self.startDate.second/(60.**2) ) / 24.0
        
        minuteArray = numpy.empty( len(dayArray), float )
        for i, d in enumerate(dayArray):
            minuteArray[i] = (d+1 - startPercentage) * 1440.0 # 1440=60 minutes * 24 hours
        return minuteArray            
        
        
    def __array2dict(self, dataArray, labels, units):
        """ Convert 2d dataArray (with labels & units) into a
        self-describing TimeSeries object"""        
        ccmcDict = TimeSeries()
        
        for i,key in enumerate(labels):
            ccmcDict.append(key, key, units[i], dataArray[:,i])

        return ccmcDict

    def __applyCoordTransf(self, dataDict):
        """ convert (r,theta,phi) velocity and magnetic field into cartesian """
        try:
            theta = numpy.deg2rad(90.0-dataDict['Lat']['data'])
            phi = numpy.deg2rad(dataDict['Lon']['data'])
        except AttributeError:
            theta = numpy.radians(90.0-dataDict['Lat']['data'])
            phi = numpy.radians(dataDict['Lon']['data'])

        vr = dataDict['V_r']
        vtheta = dataDict['V_lat']
        vphi = dataDict['V_lon']

        (vx,vy,vz) = self.__spherical2cartesian(vr['data'], vtheta['data'], vphi['data'],
                                                theta, phi)
        br = dataDict['B_r']
        btheta = dataDict['B_lat']
        bphi = dataDict['B_lon']

        (bx,by,bz) = self.__spherical2cartesian(br['data'], btheta['data'], bphi['data'],
                                                theta, phi)

        return (vx,vy,vz, bx,by,bz)
                                                
        
    def __spherical2cartesian(self, srcRad, srcTheta, srcPhi,
                              theta, phi):
        """
        In order to convert from r,theta,phi to x,y,z it is helpful to recall
        the unit vector definitions:
        
           r = i sin(theta)cos(phi) + j sin(theta)sin(phi) + k cos(theta)
           theta = i cos(theta)cos(phi) + j cos(theta)sin(phi) - k sin(theta)
           phi = -i sin(phi) + j cos(phi)
        """        
        return ( (srcRad   * numpy.sin(theta)*numpy.cos(phi) +
                  srcTheta * numpy.cos(theta)*numpy.cos(phi) +
                  srcPhi   * numpy.sin(phi)),
                 (srcRad   * numpy.sin(theta)*numpy.sin(phi) +
                  srcTheta * numpy.cos(theta)*numpy.sin(phi) +
                  srcPhi   * numpy.cos(phi)),
                 (srcRad   * numpy.cos(theta) -
                  srcTheta * numpy.sin(theta)) )

    def __getTiltAngle(self, time_minutes):
        """
        Returns an array of the tilt angle varying in time.
        """
        tiltAngle = []
        for minute in time_minutes:
            d = self.startDate + datetime.timedelta(minutes=minute)
            (x,y,z) = pyLTR.transform.SMtoGSM(0,0,1, d)
            tiltAngle.append( numpy.arctan2(x,z) )

        return numpy.array(tiltAngle, numpy.float)

            
    def __appendMetaData(self, filename):
        """
        Add standard metadata to the data dictionary.
        """
        metadata = {'Model': 'CCMC',
                    'Source': filename,
                    'Date processed': datetime.datetime.now(),
                    'Start date': self.startDate
                    }
        
        self.data.append(key='meta',
                         name='Metadata for CCMC Solar Wind file',
                         units='n/a',
                         data=metadata)

    def __parseMetadata(self, fh):
        """
        Read & parse CCMC metadata (header), which should look something like:

          # Data printout from CCMC-simulation: version 1.1
          # Data type:  ENLIL  Heliosphere
          # Run name:   Christina_Lee_030310_SH_3 Missing data:  -1.09951e+12
          # Start Date, time: 2007/08/14  18:00:00
        
        Parameters:
          fh: an open file handle
        """
        version = fh.readline()
        dataType = fh.readline()
        descriptor = fh.readline()
        missingData = float(descriptor.split('Missing data:')[1].strip())
        runName = descriptor.split('Missing data:')[0].strip()
        self.startDate = self.__parseDate(fh.readline())

        labels = fh.readline().strip('#').split()
        units = fh.readline().strip('#').split()

        return (labels, units)
        

    def __parseDate(self, dateStr):
        """
        Convert from 'YYYY/MM/DD HH:MM:SS' to datetime object

        >>> sw = CCMC('examples/data/solarWind/CCMC_wsa-enlil.dat')
        >>> sw._CCMC__parseDate('# Start Date, time: 2007/08/14  18:00:00')
        datetime.datetime(2007, 8, 14, 18, 0)
        """
        dateStr = dateStr[19:].strip()
        (yr,mon,day) = [int(s) for s in dateStr.split()[0].split('/')]
        (hr,min,sec) = [int(s) for s in dateStr.split()[1].split(':')]
        
        return datetime.datetime(year=yr, month=mon, day=day,
                                 hour=hr, minute=min, second=sec)
        
        
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
