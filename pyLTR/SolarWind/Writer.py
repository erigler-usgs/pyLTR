"""
Module that defines functions used to write solar wind data to file.
"""

# Custom
from pyLTR.Release import version as pyLTRversion
from pyLTR.TimeSeries import TimeSeries
# 3rd-party
import numpy

#Standard
import pickle
import datetime
import getpass

class LFM(object):
    """
    Write solar wind data in the LFM format to filename.

    Parameters:
      - swObj: pyLTR.SolarWind object containing data we wish to write.
      - filename: File to write
    """

    def __init__(self, swObj, filename):
        # Interpolate to one minute:
        time_1minute = list(range(int(swObj.data.getData('time_min').min()),
                             int(swObj.data.getData('time_min').max())))
        n    = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('n'))
        vx   = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('vx'))
        vy   = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('vy'))
        vz   = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('vz'))
        cs   = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('cs'))
        bx   = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('bx'))
        by   = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('by'))
        bz   = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('bz'))
        b    = numpy.interp(time_1minute, swObj.data.getData('time_min'), swObj.data.getData('b'))
    
        formatStr = self.__getWriteLFMFormatString(time_1minute,n,vx,vy,vz,cs,bx,by,bz,b)
    
        # Write the data to file
        lfmSwFilename = filename + '_SW-SM-DAT'
        print(('Saving "%s"' % lfmSwFilename))
        fh = open(lfmSwFilename, 'w')
        
        date = swObj.data.getData('meta')['Start date']
        fh.write(' %d  %d  %d  %d\n' % (date.year, date.timetuple().tm_yday, date.hour, date.minute))
        fh.write('%d     11\n' % (len(time_1minute)))
        fh.write(' DATA:\n')
        
        for i,time in enumerate(time_1minute):
            # Convert relevant quantities to SM Coordinates
            v_sm = swObj._gsm2sm(date+datetime.timedelta(minutes=time), vx[i],vy[i],vz[i])
            b_sm = swObj._gsm2sm(date+datetime.timedelta(minutes=time), bx[i],by[i],bz[i])
            tilt = swObj._getTiltAngle(date+datetime.timedelta(minutes=time))
    
            # Write data to file
            fh.write(formatStr % (time, n[i], v_sm[0],v_sm[1],v_sm[2],
                                  cs[i],
                                  b_sm[0],b_sm[1],b_sm[2],b[i],
                                  tilt))

    def __getWriteLFMFormatString(self,time_1minute,n,vx,vy,vz,cs,bx,by,bz,b,tilt=[6.283]):
        """
        Returns a Python C-style format string of how data should be
        written to the LFM SW-SM-DAT file.

        Returns a format string of the form:
          '%7.2f %4.1f %7.2f %7.2f %7.2f %7.2f %5.2f %6.2f %6.2f %6.2f %5.2f'
        
        ######################## WARNING! ########################
        This code contains a lot of cruft to format data as
        whitespace-delimeted columns centered on the decimal place!
        It should probably replace it with python-fortranformat:
        http://code.google.com/p/python-fortranformat/
        
        This duplicates the following FORMAT string in an old IDL
        Solar Wind script:
        '(10(f13.2,2x),f6.1,2x,3f10.4,f8.2,5f9.4)'        
        """
        # Implementation details:
        # Note:  '%%' escapes the '%' character in a string.
        #        So '%%%d.2f'%5  yields '%5.2f'
        #               time  density    vx      vy       vz     cs 
        formatStr  = '%%%d.2f %%%d.1f %%%d.2f %%%d.2f %%%d.2f %%%d.2f '
        #                bx       by      bz      b     tilt
        formatStr += '%%%d.2f %%%d.2f %%%d.2f %%%d.2f %%%d.3f\n'

        # We need the maximum number of digits before the decimal place for each variable:
        formatTuple = ( 3+len( str(int( max(abs(numpy.max(time_1minute)),abs(numpy.min(time_1minute))))) ),
                        4+len( str(int( max(abs(numpy.max(n)),abs(numpy.min(n))))) ),
                        4+len( str(int( max(abs(numpy.max(vx)),abs(numpy.min(vx))))) ),
                        4+len( str(int( max(abs(numpy.max(vy)),abs(numpy.min(vy))))) ),
                        4+len( str(int( max(abs(numpy.max(vz)),abs(numpy.min(vz))))) ),
                        4+len( str(int( max(abs(numpy.max(cs)),abs(numpy.min(cs))))) ),
                        4+len( str(int( max(abs(numpy.max(bx)),abs(numpy.min(bx))))) ),
                        4+len( str(int( max(abs(numpy.max(by)),abs(numpy.min(by))))) ),
                        4+len( str(int( max(abs(numpy.max(bz)),abs(numpy.min(bz))))) ),
                        4+len( str(int( max(abs(numpy.max(b)),abs(numpy.min(b))))) ),
                        5+len( str(int( max(abs(numpy.max(tilt)),abs(numpy.min(tilt))))) ) )

        # String interpolate:  Generate the I/O format string using the number of digits calcualted above.
        return (formatStr % formatTuple)


class TIEGCM(object):
    """
    Write solar wind data in TIEGCM IMF NetCDF format to filename.
    File is 5-minute data, smoothed by averaging 15 minutes of data
    from time-20 to time-5 (ie. lagged by 5 minutes).

    Parameters:
      - swObj: SolarWind object containing data we wish to write.
      - filename: File to write    

    Note: Input Solar Wind object should have a data quality array for
    each of the required variables:  
       ['is'+s+'Interped' for s in ['Bx','By','Bz','Vx','Vy','Vz','N','T']]
    """        

    def __init__(self, swObj, filename):

        # Size of interpolated data:
        d = swObj.data.getData('time_min')
        nElements = len(d) - self.__getFirstTimeIndex(swObj.data.getData('time_min'))

        # Allocate variables to do the boxcar average
        self.time_min = numpy.zeros( nElements )
        self.BxGSM = numpy.zeros( nElements )
        self.ByGSM = numpy.zeros( nElements )
        self.BzGSM = numpy.zeros( nElements )
        self.Velocity = numpy.zeros( nElements )
        self.rho = numpy.zeros( nElements )        

        # 15 minute average trailed by 5 minutes
        self.__boxcarAverage(swObj)

        # Sample data to 5 minutes
        self.__sample5min()
        
        self.isRealData = self.__getRealDataMask(swObj)
        
        # Get dates & days formatted properly for TIEGCM
        (self.dates, self.days) = self.__getTime(swObj.data.getData('meta')['Start date'],
                                                 self.time_min)

        # Do a sanity checks:
        if ( (len(self.dates) != len(self.BxGSM) ) |
             (len(self.dates) != len(self.ByGSM) ) |
             (len(self.dates) != len(self.BzGSM) ) |
             (len(self.dates) != len(self.Velocity) ) |
             (len(self.dates) != len(self.rho) ) |
             (len(self.dates) != self.isRealData.shape[1]) ):
            raise Exception('Error:  Data arrays for TIEGCM writer have size mismatch!')

        # Write TIEGCM IMF NetCDF file
        tiegcmSwFilename = filename + '.nc'
        print(('Saving "%s"' % tiegcmSwFilename))
        self.__writeNC(tiegcmSwFilename)

        # Write serialized TimeSeries pickle
        pklFilename = filename + '_TIEGCM.pkl'
        print(('Saving "%s"' % pklFilename))
        self.__writePKL(pklFilename)
        

    def __getTime(self, start, timeMinutes):
        """
        Returns list of TIEGCM-formatted dates & days from start
        incremented by timeMinutes.

        parameters:
          start: datetime.datetime object corresponding to start of file
          timeMinutes: List of minutes elapsed since start time.
        Returns:
          dates: List of YYYYDDD.FRAC (YYYY=year; DDD=day of year; FRAC=percentage of day)
          days: List of YYYYDDD (YYYY=year, DDD=day of year)
        """
        dates = [ ]
        days  = [ ]

        for t in timeMinutes:
            dt = start + datetime.timedelta(minutes=t)

            tt = dt.timetuple()
            dayFraction = (tt.tm_hour+tt.tm_min/60.+tt.tm_sec/(60.*60.))/24.

            yearDayOfYear = dt.strftime('%Y%j')
            dates.append( float(yearDayOfYear) + dayFraction )

            if yearDayOfYear not in days:
                days.append(yearDayOfYear)

        return (dates, days)

    def __getFirstTimeIndex(self, timeMinutes):
        """
        We need to average 15 minutes of data lagged by 5 minutes.
        Therefore, the first valid data point is 20 minutes after the
        start.  Find the index that corresponds to this.

        Parameters:
          timeMinutes:  time in minutes since start.

        # DocTest with data spaced by [0, 15, 30, 60] minutes:
        """

        start = timeMinutes[0]
        for i,date in enumerate(timeMinutes):
            if date >= (start + 20):
                return i

        # There's a problem if we made it this far.
        raise Exception('Not enough solar wind input to write a TIEGCM NetCDF file.  Greater than 20 minutes of data is required.  We do a 15 minute average lagged by 20 minutes.')
            
    def __boxcarAverage(self, swObj):
        """
        Compute 15 minute centered boxcar average lagged by 5 minutes
        of solar wind input required by TIEGCM.  Assumes data points
        are 5 minutes apart.

        LaTeX: $data(t_0) = \sum_{t-20}^{t-5} f(t)/15
        """
        dx = int( swObj.data.getData('time_min')[1] - swObj.data.getData('time_min')[0] )
        if (15 % dx) != 0:
            raise Exception('Time cadence in solar wind object incompatible with 15-minute trailing boxcar average lagged by 5 minutes.  Make sure time cadence is divisible by 15!')

        for i in range( len(self.time_min) ):
            # Compute 15-minute trailing boxcar average lagged by 5 minutes.

            # index i on LHS should be offset by 20 minutes RHS (raw data)
            self.time_min[i] = swObj.data.getData('time_min')[i]+20

            intervals = list(range(i,i+(15/dx)+1))
            nIntervals = len(intervals)
            
            # Compute 15-minute boxcar average
            for j in intervals:
                self.BxGSM[i]    += swObj.data.getData('bx')[j]
                self.ByGSM[i]    += swObj.data.getData('by')[j]
                self.BzGSM[i]    += swObj.data.getData('bz')[j]
                self.Velocity[i] += numpy.sqrt(pow(swObj.data.getData('vx')[j],2) + 
                                               pow(swObj.data.getData('vy')[j],2) + 
                                               pow(swObj.data.getData('vz')[j],2))
                self.rho[i]      += swObj.data.getData('n')[j]

            self.BxGSM[i]    /= nIntervals
            self.ByGSM[i]    /= nIntervals
            self.BzGSM[i]    /= nIntervals
            self.Velocity[i] /= nIntervals
            self.rho[i]      /= nIntervals
            

    def __sample5min(self):
        """
        Sample member data arrays to 5-minute time intervals.
        """
        
        timeInterp = numpy.arange(self.time_min[0],
                                  self.time_min[len(self.time_min)-1]+1,
                                  5.0)
        
        self.BxGSM = numpy.interp(timeInterp, self.time_min, self.BxGSM)
        self.ByGSM = numpy.interp(timeInterp, self.time_min, self.ByGSM)
        self.BzGSM = numpy.interp(timeInterp, self.time_min, self.BzGSM)
        self.Velocity = numpy.interp(timeInterp, self.time_min, self.Velocity)
        self.rho = numpy.interp(timeInterp, self.time_min, self.rho)
        self.time_min = timeInterp

    def __getRealDataMask(self, swObj):
        """
        Returns boolean array that determines where real unmodified
        data was used (True) versus bad data values that had to be
        interpolated (False).
        
        Dimensions of array are:
        [bx[:], by[:], bz[:], V[:], density[:]]
        where "[:]" denotes the size of the source array.
        """

        # Allocate output:
        isRealData = numpy.empty([8, len(self.time_min)])
        isRealData.fill(True)

        # Pre-compute indices to search through for speed:
        searchIndices = []
        for timeIdx, timeVal in enumerate(self.time_min):
            searchIndices.append([numpy.searchsorted(swObj.data.getData('time_min'), timeVal-20),
                                 numpy.searchsorted(swObj.data.getData('time_min'), timeVal-5)])        

        dataPairs = [['bx', 'isBxInterped'],
                     ['by', 'isByInterped'],
                     ['bz', 'isBzInterped'],
                     ['vx', 'isVxInterped'],
                     ['vy', 'isVyInterped'],
                     ['vz', 'isVzInterped'],
                     ['n',  'isNInterped']]           

        for varIdx, pair in enumerate(dataPairs):
            if pair[1] not in list(swObj.data.keys()):
                print('No data quality flag set in Solar Wind object.  Assuming valid data everywhere.')
            else:
                # Search the whole time series for bad/interpolated data values:
                for timeIdx, timeVal in enumerate(self.time_min):
                    # Step through data used 15-minute boxcar average
                    # trialed by 5 minutes for bad indices
                    for searchIdx in range(searchIndices[timeIdx][0], searchIndices[timeIdx][1]):
                        if swObj.data.getData(pair[1])[searchIdx]:
                            isRealData[varIdx][timeIdx] = False
                            break        

        # Return isRealData mask array for:
        # [bx, by, bz,   Velocity,    density]
        return numpy.array( [isRealData[0], isRealData[1], isRealData[2],
                             isRealData[3] * isRealData[4] * isRealData[5],
                             isRealData[6]] )
                            

    def __writeNC(self, filename):
        """
        Save Solar Wind data to TIEGCM IMF NetCDF file
        """
        import scipy.io
        fh = scipy.io.netcdf_file(filename, 'w')
        
        # Global Attributes
        fh.url_reference = "http://cdaweb.gsfc.nasa.gov/sp_phys"
        fh.Source = "Hourly OMNI combined 1AU IP Data"
        fh.Description = "15-minute average of OMNI data trailed by 5 minutes. Sampled to 5-minute output"
        fh.CreationTime = str( datetime.datetime.now() )
        fh.CreatedBy = getpass.getuser()
        fh.Version = pyLTRversion
    
        #
        # Main data arrays
        #

        # Dimensions for main data arrays
        fh.createDimension('ndata', len(self.dates))

        ##### date #############################################################
        dateVar = fh.createVariable('date', 'd', ('ndata',))
        dateVar.long_name = "year-day plus fractional day: yyyyddd.frac"
        dateVar[:] = self.dates
        
        ##### bx ###############################################################
        bxVar = fh.createVariable('bx', 'd', ('ndata',))
        bxVar.long_name = "1AU IP Bx, GSM"
        bxVar.units = 'nT'
        bxVar[:] = self.BxGSM
        
        bxMask = fh.createVariable('bxMask', 'b', ('ndata',))
        bxMask.long_name = "Quality flag: 0=data derived from linear interpolation.  1=data derived directly"
        bxMask.units = 'boolean'
        bxMask[:] = self.isRealData[0]
    
        ##### by ###############################################################
        byVar = fh.createVariable('by', 'd', ('ndata',))
        byVar.long_name = "1AU IP By, GSM"
        byVar.units = 'nT'
        byVar[:] = self.ByGSM
    
        byMask = fh.createVariable('byMask', 'b', ('ndata',))
        byMask.long_name = "Quality flag: 0=data derived from linear interpolation.  1=data derived directly"
        byMask.units = 'boolean'
        byMask[:] = self.isRealData[1]

        ##### bz ###############################################################
        bzVar = fh.createVariable('bz', 'd', ('ndata',))
        bzVar.long_name = "1AU IP Bz, GSM"
        bzVar.units = 'nT'
        bzVar[:] = self.BzGSM
    
        bzMask = fh.createVariable('bzMask', 'b', ('ndata',))
        bzMask.long_name = "Quality flag: 0=data derived from linear interpolation.  1=data derived directly"
        bzMask.units = 'boolean'
        bzMask[:] = self.isRealData[2]

        ##### swvel ############################################################
        swvelVar = fh.createVariable('swvel', 'd', ('ndata',))
        swvelVar.long_name = "1AU IP Plasma Speed"
        swvelVar.units = 'Km/s'
        swvelVar[:] = self.Velocity

        velocityMask = fh.createVariable('velMask', 'b', ('ndata',))
        velocityMask.long_name = "Quality flag: 0=data derived from linear interpolation.  1=data derived directly"
        velocityMask.units = 'boolean'
        velocityMask[:] = self.isRealData[3]
    
        ##### swden ############################################################
        swdenVar = fh.createVariable('swden', 'd', ('ndata',))
        swdenVar.long_name = "1AU IP N (ion)"
        swdenVar.units = 'per cc'
        swdenVar[:] = self.rho

        densityMask = fh.createVariable('denMask', 'b', ('ndata',))
        densityMask.long_name = "Quality flag: 0=data derived from linear interpolation.  1=data derived directly"
        densityMask.units = 'boolean'
        densityMask[:] = self.isRealData[4]

        #        
        # Unused data arrays
        #

        # Dimensions
        #fh.createDimension('ndays', len(self.days))
    
        ###### kp ###############################################################
        #kpVar = fh.createVariable('kp', 'd', ('ndata',))
        #kpVar.long_name = "3-hourly Kp index"
        #kpVar[:] = range(2)
        
        ##### days #############################################################
        # This was used by the F10.7 arrays, which we're not storing.
        #daysVar = fh.createVariable('days', 'i', ('ndays',))
        #daysVar.long_name = "year-day: yyyyddd for each day"
        #daysVar[:] = self.days

        ###### f107d ############################################################
        #f107dVar = fh.createVariable('f107d', 'd', ('ndays',))
        #f107dVar.long_name = ":F10.7 cm daily solar flux"
        #f107dVar.url_reference = "ftp://ftp.ngdc.noaa.gov/STP/GEOMAGNETIC_DATA/INDICES/KP_AP"
        #f107dVar[:] = range(2)
        
        ###### f107a ############################################################
        #f107aVar = fh.createVariable('f107a', 'd', ('ndays',))
        #f107aVar.long_name = "F10.7 cm 81-day average solar Bflux"
        #f107aVar.url_reference = "ftp://ftp.ngdc.noaa.gov/STP/GEOMAGNETIC_DATA/INDICES/KP_AP"
        #f107aVar[:] = range(2)
    
        ##### missing ##########################################################
        missingVar = fh.createVariable('missing', 'd', ())
        missingVar.long_name = "missing data value"
        missingVar.assignValue(1.0e36)
    
        # Write the data to file
        fh.close()

    def __writePKL(self, filename):
        """
        Serialize & write data as a pyLTR.TimeSeries object using
        Python's Pickle package.
        """
        dataTimeSeries = TimeSeries()
        dataTimeSeries.append('time_dates', 'Time (YYYYDOY.percent))', 'fraction of day', self.dates)
        dataTimeSeries.append('time_min', 'Time (Minutes since start)', 'min', self.time_min)

        dataTimeSeries.append('bx', 'Bx (gsm)', r'$\mathrm{nT}$', self.BxGSM)
        dataTimeSeries.append('bxMask', 'Was Bx derived from valid data?', r'bool', self.isRealData[0])

        dataTimeSeries.append('by', 'By (gsm)', r'$\mathrm{nT}$', self.ByGSM)
        dataTimeSeries.append('byMask', 'Was By derived from valid data?', r'bool', self.isRealData[1])

        dataTimeSeries.append('bz', 'Bz (gsm)', r'$\mathrm{nT}$', self.BzGSM)
        dataTimeSeries.append('bzMask', 'Was Bz derived from valid data?', r'bool', self.isRealData[2])

        dataTimeSeries.append('v', '1AU IP Plasma Speed', r'$\mathrm{Km/s}$', self.Velocity)
        dataTimeSeries.append('vMask', 'Was V derived from valid data?', r'bool', self.isRealData[3])

        dataTimeSeries.append('rho', '1AU IP N (ion)', r'$\mathrm{n/cc}$', self.rho)
        dataTimeSeries.append('rhoMask', 'Was Rho derived from valid data?', r'bool', self.isRealData[4])

        fh = open(filename,'wb')
        pickle.dump(dataTimeSeries,fh,protocol=2)
        fh.close()

if __name__ == '__main__':
    import doctest
    doctest.testmod()
