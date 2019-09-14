import datetime
import time
import numpy

import pyLTR
from .SolarWind import SolarWind

class ENLIL(SolarWind):
    """
    ENLIL Solar Wind file. Data is stored in spherical HEEQ coordinates.

    FIXME: Some ENLIL private member functions duplicate functionality
    of the OMNI and CCMC classes (date/time parsing, coordinate
    transforms, etc.).  Maybe I should refactor to protected member
    functions of the SolarWind base class?
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

        enlilVersion = self.__readMetaData(fh)
        (headerDict, units) = self.__readHeaders(fh)

        (dates, dataArray) = self.__readData(fh)
        
        self.__storeDataDict(dates, dataArray, headerDict)
        self.__appendMetaData(filename)
        self._appendDerivedQuantities()
        
    def __parseDateTime(self, inDate, inTime):
        """
        Returns a datetime.datetime object given two strings containing
        date & time formatted as 'YYYY-MM-DD HH:MM'.

        >>> sw = ENLIL('examples/data/solarWind/ENLIL-cr2068-a3b2.Earth.dat')
        >>> sw._ENLIL__parseDateTime('2008-03-19', '22:55')
        datetime.datetime(2008, 3, 19, 22, 55)
        """
        dateTimeStr = inDate  + ' ' + inTime
        # Python 2.5 implements a handy one-liner to do this:
        #return datetime.datetime.strptime(dateTimeStr,        '%Y-%m-%d %H:%M')
        # Python 2.4 doesn't have this... more convoluted appoach that seems to work:
        return datetime.datetime( *time.strptime(dateTimeStr, '%Y-%m-%d %H:%M')[0:5] )

    def __readMetaData(self, fh):
        """
        Eat metadata ENLIL solar wind metadata & return model name.
        """
        path = fh.readline()[1:].strip()
        description =fh.readline()[1:].strip()
        fh.readline() # blank line
        program = fh.readline().split('=')[1].strip()
        version = fh.readline().split('=')[1].strip()
        observatory = fh.readline().split('=')[1].strip()
        corona = fh.readline().split('=')[1].strip()
        shift_deg = fh.readline().split('=')[1].strip()
        coordinates = fh.readline().split('=')[1].strip()
        run_descriptor = fh.readline().split('=')[1].strip()
        
        return program + '_' + str(version) + ' ' + corona

    def __readHeaders(self, fh):
        """
        Read headers & units for each column.  Pop off the unused 'date' column.
        """
        fh.readline()
        fh.readline()
        
        headersStr = fh.readline()
        headers = [ s.strip() for s in headersStr[1:].split() ]
        unitsStr = fh.readline()
        units = [ s.strip() for s in unitsStr[1:].split() ]
        
        fh.readline()
        
        headers.pop(1)
        units[0] = 'mjd'
        units[1] = 'seconds'

        self.startDate = self.__getStartDate(fh)

        # Get a mapping of header names to column index
        headerDict = dict(list(zip(headers,list(range(len(headers))))))
        return (headerDict, units)

    def __getStartDate(self, fh):
        """
        Get the current date & time as a datetime.datetime object.
        
        Note: Assumes the current line is a data entry
        """
        pos = fh.tell()
        
        line = fh.readline()
        dateTime = self.__parseDateTime(line.split()[1].strip(), line.split()[2].strip())
        
        fh.seek(pos)
        
        return dateTime

    def __deltaMinutes(self, date, time, startDate):
        """
        Parameters:
          date: date string YYYY-MM-DD
          time: Time string HH:MM
          startDate: datetime object
        Returns: Number of minutes elapsed between (date, time) and startDate.

        >>> sw = ENLIL('examples/data/solarWind/ENLIL-cr2068-a3b2.Earth.dat')
        >>> sw._ENLIL__deltaMinutes('2008-03-19','22:58', datetime.datetime(2008,3,19,22,55,0))
        3
        """
        curDate = self.__parseDateTime(date, time)
        diff = curDate - startDate
        
        return (diff.days*24*60 + diff.seconds/60)
        
    def __readData(self, fh, commentChar='#'):
        """
        Read & return 2d NumPy array of data from file
        """
        dates = []
        rows = []

        for line in fh:
            if line[0] == commentChar:
                continue
            fields = line.split()
            dates.append( self.__parseDateTime(fields[1], fields[2]) )
            # Replace YYYY-MM-DD date with number of minutes since start
            fields[1] = self.__deltaMinutes(fields[1].strip(), fields[2].strip(), self.startDate)
            # Pop the HH:MM time off the list, since we wont use it any more.
            fields.pop(2)
            # Fields now contains entirely floating point numbers.
            rows.append( [float(s) for s in fields] )

        return (dates, numpy.array(rows, numpy.float))
            

    def __storeDataDict(self, dates, dataArray, columnIndex):
        """
        Populte self.data TimeSeries object via the 2d dataArray read
        from file. Make sure to store data in GSM coordinates (the
        default used by pyLTR.SolarWind).
        """
        (vCartesian, bCartesian) = self.__getCartesian(dataArray, columnIndex)
        (vGsm, bGsm) = self.__heeq2gsm(dates, vCartesian, bCartesian)

        self.data.append('time_min', 'Time (Minutes since start)', 'min', dataArray[:,columnIndex['time']])

        self.data.append('bx', 'Bx (gsm)', r'$\mathrm{nT}$', bGsm[0,:])
        self.data.append('by', 'By (gsm)', r'$\mathrm{nT}$', bGsm[1,:])
        self.data.append('bz', 'Bz (gsm)', r'$\mathrm{nT}$', bGsm[2,:])
        
        self.data.append('vx', 'Vx (gsm)', r'$\mathrm{km/s}$', vGsm[0,:])
        self.data.append('vy', 'Vy (gsm)', r'$\mathrm{km/s}$', vGsm[1,:])
        self.data.append('vz', 'Vz (gsm)', r'$\mathrm{km/s}$', vGsm[2,:])

        self.data.append('n', 'Density', r'$\mathrm{1/cm^3}$', dataArray[:,columnIndex['N']])

        self.data.append('t', 'Temperature', r'$\mathrm{kK}$', dataArray[:,columnIndex['T']]*1e-3)
        

    def __getCartesian(self, dataArray, columnIndex):
        """ convert (r,theta,phi) velocity and magnetic field into cartesian """
        try:
            theta = numpy.deg2rad(90.0-dataArray[:,columnIndex['Lat']])
            phi = numpy.deg2rad(dataArray[:,columnIndex['Lon']])
        except AttributeError:
            theta = numpy.radians(90.0-dataArray[:,columnIndex['Lat']])
            phi = numpy.radians(dataArray[:,columnIndex['Lon']])

        vr = dataArray[:,columnIndex['Vrad']]
        vtheta = dataArray[:,columnIndex['Vthe']]
        vphi = dataArray[:,columnIndex['Vphi']]

        (vx,vy,vz) = self.__spherical2cartesian(vr, vtheta, vphi,
                                                theta, phi)
        br = dataArray[:,columnIndex['Brad']]
        btheta = dataArray[:,columnIndex['Bthe']]
        bphi = dataArray[:,columnIndex['Bphi']]

        (bx,by,bz) = self.__spherical2cartesian(br, btheta, bphi,
                                                theta, phi)

        return ( (vx,vy,vz), (bx,by,bz) )
                                                
        
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
        
    def __heeq2gsm(self, dates, vHeeq, bHeeq):
        """
        Transform magnetic field B and velocity V from HEEQ to GSM
        coordinates.  Returns a two 3xN vectors: vGSM and bGSM.
        """
        vGsm = []
        bGsm = []
        for i,d in enumerate(dates):            
            vGsm.append(pyLTR.transform.HEEQtoGSM(vHeeq[0][i], vHeeq[1][i], vHeeq[2][i], d))
            
            bGsm.append(pyLTR.transform.HEEQtoGSM(bHeeq[0][i], bHeeq[1][i], bHeeq[2][i], d))

        vGsm = numpy.array(vGsm, numpy.float)
        bGsm = numpy.array(bGsm, numpy.float)

        return ( numpy.array(vGsm.T), numpy.array(bGsm.T) )

    def __appendMetaData(self, filename):
        """
        Add standard metadata to the data dictionary.
        """
        metadata = {'Model': 'ENLIL',
                    'Source': filename,
                    'Date processed': datetime.datetime.now(),
                    'Start date': self.startDate
                    }
        
        self.data.append(key='meta',
                         name='Metadata for ENLIL Solar Wind file',
                         units='n/a',
                         data=metadata)
                                 

if __name__ == '__main__':
    import doctest
    doctest.testmod()
