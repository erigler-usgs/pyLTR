"""
This module holds functions for dealing with time series data from geomagnetic 
observatories. The basic data unit is dictionary (may change to a class in the
future). This geomagDict dictionary must hold a minimum set of keys and values:

1) iagaID - a 3-character string uniquely identifying the observaotry
   (preferably an official IAGA designation, like BOU for Boulder)
2) dateTime - a vector of datetimes
3) doy - a corresponding vector of days-of-year
4) magCoord - a 4-character string defining 4 observatory components
   (e.g., HDZF for horizontal, declination, down, and scalar field)
5) four 1-character key names corresponding to magCoord characters 

Ideally, additional elements may be in the dictionary, serving as metadata
that may be expected in standard IAGA2002 or IMF files:

6) geoLatitude - (ideally geodetic) latitude of observatory
7) geoLongitude - (ideally geodetic) longitude of observatory
8) elevation - (ideally geodetic) elevation of observatory
9) sensorOrient - 4-character string defining orientation of sensors
   (may differ from magCoord if measurements transformed, e.g., XYZF)
10) dataSource - string identifying source of data (e.g., USGS Geomag Program)
11) stationName - string identifying station (longer than iagaID)
12) digitalSampling - string describing raw data sampling (e.g., .01 second)
13) dataInterval - string describing the sampling of reported data
    (e.g., filtered 1-minute (::15-01:45))
14) dataType - string describing data quality type (e.g., definitive)
15) decBase - declination baseline in minutes East (if None or not provided, 
    assume magCoord is wrt standard geographic coordinate axes, else instrument 
    axes...this is how USGS reports variational data)

The following lists existing/expected functionality:

1) read/write IAGA2002 files                    (done)
2) read/write IMF1.23 files                     (tbd)
3) transform between typical mag coordinates    (some done)


"""


# import NumPy
import numpy as np

# import pyplot
import matplotlib.pyplot as plt

# import datetime
import datetime as dt

# import glob
import glob

#import os
import os



def readIAGA2002(filename):
   """
   Read an IAGA2002 file into a specially constructed Python dictionary
   
   """
   
      # open filename for ascii reading
   fh = open(filename,'r')

   
   # read content line by line and process
   hdr = []
   iagaID = None
   magCoord = None
   decBase = None
   ut = []
   doy = []
   v1 = []
   v2 = []
   v3 = []
   v4 = []
   
   for line in fh:
      
      #print line,
      
      if line.strip()[-1] == "|":
         # anything ending in "|" is a header line; extract certain values for
         # use in this function, but store the entire header
         
         # don't bother to read the "format" header line, since it can only be
         # "IAGA-2002"
         dataFormat = "IAGA-2002"
         
         if line.strip().lower().startswith("source of data"):
            # store data source string (excluding trailing '|')
            dataSource = " ".join(line.split()[3:-1])
                  
         elif line.strip().lower().startswith("station name"):
            # store full station name string (excluding trailing '|')
            stationName = " ".join(line.split()[2:-1])
         
         elif (line.strip().lower().startswith("iaga code") or
               line.strip().lower().startswith("station code")):
            # store (typically) 3-character IAGA code
            iagaID = " ".join(line.split()[2:-1])
            
         elif line.strip().lower().startswith("geodetic latitude"):
            # store latitude as numpy.float or None
            try:
              geoLatitude = np.float(line.split()[2])
            except:
              geoLatitude = None
            
         elif line.strip().lower().startswith("geodetic longitude"):
            # store longitude as numpy.float or None
            try:
              geoLongitude = np.float(line.split()[2])
            except:
              geoLongitude = None
         
         elif line.strip().lower().startswith("elevation"):
            # store elevation as numpy.float or None
            try:
              elevation = np.float(line.split()[1])
            except:
              elevation = None
                  
         elif line.strip().lower().startswith("reported"):
            # determine what coordinates are stored in file
            try:
              magCoord = line.split()[1]
            except:
              magCoord = None
            if not 'sensorOrient' in locals():
              sensorOrient = magCoord
         
         elif line.strip().lower().startswith("sensor orientation"):
            # store orientation of the sensor
            try:
              sensorOrient = line.split()[2]
            except:
              sensorOrient = None
            if not 'magCoord' in locals():
              magCoord = sensorOrient
                  
         elif line.strip().lower().startswith("digital sampling"):
            # store digital sampling string (excluding trailing '|')
            digitalSampling = " ".join(line.split()[2:-1])
                  
         elif line.strip().lower().startswith("data interval type"):
            # store data interval type string (excluding trailing '|')
            dataInterval = " ".join(line.split()[3:-1])
                  
         elif line.strip().lower().startswith("data type"):
            # store data type string (excluding trailing '|')
            dataType = " ".join(line.split()[2:-1])
                  
         # comment lines are generally retained as part of the header, but 
         # sometimes they hold necessary data
         
         elif line.strip().lower().startswith("# decbas"):
            # store declination baseline in minutes of arc
            decBase = float(line.split()[2]) / 10
         
         # append this header line to a header list
         hdr.append(line.rstrip())
         
      elif line.strip():
         # if not a comment, and not empty/blank, should be data
                  
         # list of 'words' for each data line, assumes there are always 7
         ll = line.split()
         
         # convert date/time string into datetime objects
         ut.append(dt.datetime.strptime(ll[0]+ll[1], "%Y-%m-%d%H:%M:%S.%f"))
         
         # read in day of year
         doy.append(int(ll[2]))
         
         # read in 4 components expected in IAGA2002-formatted file
         v1.append(float(ll[3]))
         v2.append(float(ll[4]))
         v3.append(float(ll[5]))
         v4.append(float(ll[6]))
      
      else:
         print('ignoring blank line in file')
         
   fh.close()
   
   
   # determine actual coordinates from reported/magCoord and decBas
   # NOTE: this addresses the fact that reported is always all-caps in
   #       IAGA2002 files, but that the presence of decBas may hint at
   #       something non-standard, like the USGS' hdZF (where h and d
   #       are aligned with the sensor, not standard fixed coordinates.
   # FIXME: this is probably a bad idea, try to come up with a solution
   #        that is less ad-hoc -EJR 12/2014
   if decBase==None:
      if magCoord.lower() == 'hdzf':
         magCoord = 'HDZF'
      elif magCoord.lower() == 'hezf':
         magCoord = 'HEZF'
      if magCoord.lower() == 'hdzg':
         magCoord = 'HDZG'
      elif magCoord.lower() == 'hezg':
         magCoord = 'HEZG'
   else:
      if magCoord.lower() == 'hdzf':
         magCoord = 'hdZF'
      elif magCoord.lower() == 'hezf':
         magCoord = 'heZF'
      if magCoord.lower() == 'hdzg':
         magCoord = 'hdZG'
      elif magCoord.lower() == 'hezg':
         magCoord = 'heZG'
      
   
   # convert lists into numpy arrays
   ut = np.array(ut)
   doy = np.array(doy)      
   v1 = np.array(v1)
   v2 = np.array(v2)
   v3 = np.array(v3)
   v4 = np.array(v4)
   
   # convert 99999 flags to nans
   v1[v1>=99999] = np.nan
   v2[v2>=99999] = np.nan
   v3[v3>=99999] = np.nan
   v4[v4>=99999] = np.nan
   
   
   # initialize outDict, and add common keys
   outDict = {}
   outDict['dateTime'] = ut
   outDict['doy'] = doy
   outDict['header'] = hdr
   outDict['dataSource'] = dataSource
   outDict['stationName'] = stationName
   outDict['iagaID'] = iagaID
   outDict['geoLatitude'] = geoLatitude
   outDict['geoLongitude'] = geoLongitude
   outDict['elevation'] = elevation
   outDict['magCoord'] = magCoord
   outDict['sensorOrient'] = sensorOrient
   outDict['digitalSampling'] = digitalSampling
   outDict['dataInterval'] = dataInterval
   outDict['dataType'] = dataType
   outDict['decBase'] = decBase
   
   
   # add coordinate-specific keys
   outDict[magCoord[0]] = v1
   outDict[magCoord[1]] = v2
   outDict[magCoord[2]] = v3
   outDict[magCoord[3]] = v4
   
   
   return outDict

   

def writeIAGA2002(geomagDict, filename=None, merge=False):
   """
   Write a data dictionary containing certain geomagnetism-related keys out to
   an IAGA-2002 formatted data file.
   """
   
   # extract required metadata from geomagDict
   iagaID = geomagDict['iagaID']
   ut = geomagDict['dateTime']
   doy = geomagDict['doy']
   magCoord = geomagDict['magCoord']
   
   # extract optional metadata from geomagDict, substituting defaults if none
   if 'header' in geomagDict and geomagDict['header']:
      hdr = geomagDict['header']
   else:
      hdr = ""
   if 'dataSource' in geomagDict and geomagDict['dataSource']:
      dataSource = geomagDict['dataSource']
   else:
      dataSource = "Unknown"
   if 'stationName' in geomagDict and geomagDict['stationName']:
      stationName = geomagDict['stationName']
   else:
      stationName = iagaID
   if 'geoLatitude' in geomagDict and geomagDict['geoLatitude']:
      geoLatitude = geomagDict['geoLatitude']
   else:
      geoLatitude = np.nan
   if 'geoLongitude' in geomagDict and geomagDict['geoLongitude']:
      geoLongitude = geomagDict['geoLongitude']
   else:
      geoLongitude = np.nan
   if 'elevation' in geomagDict and geomagDict['elevation']:
      elevation = geomagDict['elevation']
   else:
      elevation = 0
   if 'sensorOrient' in geomagDict and geomagDict['sensorOrient']:
      sensorOrient = geomagDict['sensorOrient']
   else:
      sensorOrient = magCoord
   if 'digitalSampling' in geomagDict and geomagDict['digitalSampling']:
      digitalSampling = geomagDict['digitalSampling']
   else:
      digitalSampling = "Unknown"
   if 'dataInterval' in geomagDict and geomagDict['dataInterval']:
      dataInterval = geomagDict['dataInterval']
   else:
      dataInterval = "Unknown"
   if 'dataType' in geomagDict and geomagDict['dataType']:
      dataType = geomagDict['dataType']
   else:
      dataType = "Unknown"
   if 'decBase' in geomagDict and geomagDict['decBase']:
      decBase = geomagDict['decBase']
   else:
      decBase = None
   
   
   # construct standards-compliant filename from geomagDict elements
   # FIXME: need to deal with digitalSampling, dataType, dataInterval, etc.
   if filename==None:
      filename = (iagaID.lower() +
                  ut[0].strftime("%Y%m%d") +
                  magCoord.lower() +
                  ".txt")
   
   
   # merge with existing file if present
   # NOTE: if geomagDict has fill values, these will be replaced with whatever
   #       exists in filename before re-writing filename; if this is not what
   #       you want, set merge=False, and figure out how to write over filename.
   if merge:
      
      try:
         mergeDict = readIAGA2002(filename)
      except:
         print("Filename: "+filename+" does not exist; creating new file")
         mergeDict = {}
      
      if mergeDict:         
                  
         for v in magCoord:
            
            mIdx = np.nonzero(np.isnan(geomagDict[v]))[0]
            mDT = geomagDict['dateTime'][mIdx]            
            
            for i,d in zip(mIdx, mDT):
               
               # find corresponding datetime in mergeDict
               geomagDict[v][i] = mergeDict[v][mergeDict['dateTime'] == d]
            
               
      
   # makin' copies
   v1 = geomagDict[magCoord[0]].copy()
   v2 = geomagDict[magCoord[1]].copy()
   v3 = geomagDict[magCoord[2]].copy()
   v4 = geomagDict[magCoord[3]].copy()
   
   # replace NaNs with IAGA-standard flags
   # NOTE: I think USGS variation data files do this wrong, using 99999.99
   v1[np.isnan(v1)] = 99999.00
   v2[np.isnan(v2)] = 99999.00
   v3[np.isnan(v3)] = 99999.00
   v4[np.isnan(v4)] = 99999.00
   
   
   # IAGA standard is to use only upper-case letters, even when there ambiguity
   # about which vector component is being used
   reported = magCoord.upper()
   
         
   # open filename for ascii writing
   fh = open(filename,'w')
   
   
   # print the header
   fh.write(" Format                 " +
            "IAGA-2002" + 
            "                                    |\n")
   
   fh.write(" Source of Data         " + 
            ("%-45s"%dataSource)[:45] +         # ensure string is 45 chars long
            "|\n")
   
   fh.write(" Station Name           " + 
            ("%-45s"%stationName)[:45] +        # ensure string is 45 chars long
            "|\n")
   
   fh.write(" IAGA CODE              " + 
            ("%-45s"%iagaID)[:45] +             # ensure string is 45 chars long
            "|\n")
   
   fh.write(" Geodetic Latitude      " + 
            ("%07.3f"%geoLatitude)[:7] +             # ensure string is 7 chars long
            "                                      |\n")
   
   fh.write(" Geodetic Longitude     " + 
            ("%07.3f"%geoLongitude)[:7] +             # ensure string is 7 chars long
            "                                      |\n")
   
   fh.write(" Elevation              " + 
            ("%06i"%elevation)[:6] +             # ensure string is 6 chars long
            "                                       |\n")
   
   fh.write(" Reported               " + 
            ("%-4s"%reported)[:4] +         # ensure string is 4 chars long
            "                                         |\n")
   
   fh.write(" Sensor Orientation     " + 
            ("%-4s"%sensorOrient)[:4] +             # ensure string is 4 chars long
            "                                         |\n")

   fh.write(" Digital Sampling       " + 
            ("%-45s"%digitalSampling)[:45] +             # ensure string is 45 chars long
            "|\n")

   fh.write(" Data Interval Type     " + 
            ("%-45s"%dataInterval)[:45] +             # ensure string is 45 chars long
            "|\n")

   fh.write(" Data Type              " + 
            ("%-45s"%dataType)[:45] +             # ensure string is 45 chars long
            "|\n")

   
   if decBase != None:
      fh.write(" # DECBAS             " + 
               ("%6i"%(decBase*10))[:6] +         # ensure string is 6 chars long
               "    (Baseline declination value in       |\n" +
               " #                      tenths of minutes East (0-216,000))          |\n")
   
   
   
   # print column headings
   fh.write("DATE       TIME         DOY"+
            "     "+iagaID[:3].upper()+reported[0].upper()+" "+
            "     "+iagaID[:3].upper()+reported[1].upper()+" "+
            "     "+iagaID[:3].upper()+reported[2].upper()+" "+
            "     "+iagaID[:3].upper()+reported[3].upper()+"   |\n")
   
   # print time stamps and data
   for i in range(ut.size):
      fh.write(ut[i].strftime("%Y-%m-%d %H:%M:%S.%f")[:23] + # truncates to milliseconds
               " " + "%3i"%doy[i] + "   " +
               " " + "%9.2f"%v1[i] + 
               " " + "%9.2f"%v2[i] + 
               " " + "%9.2f"%v3[i] + 
               " " + "%9.2f"%v4[i] +
               "\n")


   return filename




def fakeGeomag(iagaID, magCoord, dtBegin, dtEnd, deltat=60):
   """
   Generate a fake geomagDict, with data filled by NaNs, and some basic header
   information
   """
   
   # generate array of datetimes and days-of-year
   tdelta = dt.timedelta(seconds=deltat)
   ut = []
   doy = []
   dtNext = dtBegin
   while dtNext <= dtEnd:
      ut.append(dtNext)
      doy.append(dtNext.timetuple().tm_yday)
      dtNext = dtNext + tdelta
   
   ut = np.array(ut)
   doy = np.array(doy)
   
   
   # initialize outDict
   outDict = {}
   outDict['iagaID'] = iagaID
   outDict['magCoord'] = magCoord
   outDict['dateTime'] = ut
   outDict['doy'] = doy
   outDict['fake'] = True # flag that this is a fake data set
   
   # generate common array of missing/bad data to be used regardless of magCoord
   #nines = np.zeros(ut.size) + 99999.
   nans = np.zeros(ut.size) + np.nan
   
   # create vector comonents based on magCoord
   outDict[magCoord[0]] = nans
   outDict[magCoord[1]] = nans
   outDict[magCoord[2]] = nans
   outDict[magCoord[3]] = nans
   
   
   return outDict




def getIAGADaily(iagaID, dtBegin, dtEnd, magCoord='XYZF', path='./', suffix='', extension='.min'):
   """
   Wrapper for readIAGA2002 and xformGeomagDict to read in all files with
   IAGA202 standard names and concatenate the observations into a single
   geomagDict output.
   """
   
   # generate a list of date strings that include both dtBegin and dtEnd
   dateString = []
   dtNext = dtBegin.date()
   while dtNext <= dtEnd.date():
      dateString.append(dtNext.strftime("%Y%m%d"))
      dtNext = dtNext + dt.timedelta(1)
         
   
   # intialize outDict
   outDict = {}
   
   # read corresponding IAGA2002 files
   for ymd in dateString:
      
      # assumes files are IAGA2002-compliant, with no capital letters
      filename = glob.glob(path+iagaID.lower()+ymd+"*"+suffix+extension)
            
      if len(filename) == 0:
         
         print("generating fake IAGA2002 output for date "+ymd)
         
         dt1 = dt.datetime.strptime(ymd, "%Y%m%d")
         dt2 = dt1 + dt.timedelta(minutes=1439)
         tmpDict = fakeGeomag(iagaID, magCoord, dt1, dt2)
         
      elif len(filename) == 1:
         
         # If exception is not recognized and deemed acceptable, raise
         # exception, otherwise generate fake data for the entire day
         try:
            tmpDict = readIAGA2002(filename[0])
            tmpDict = xformGeomag(tmpDict, magCoord=magCoord)
            
         except AttributeError:
            # 
            print(filename[0]+" exists, but required metadata cannot be read...")
            print("generating fake IAGA2002 output for date "+ymd)
         
            dt1 = dt.datetime.strptime(ymd, "%Y%m%d")
            dt2 = dt1 + dt.timedelta(minutes=1439)
            tmpDict = fakeGeomag(iagaID, magCoord, dt1, dt2)
         
         except ValueError:
            #
            print(filename[0]+" exists, cannot convert data to NumPY floats...")
            print("generating fake IAGA2002 output for date "+ymd)
         
            dt1 = dt.datetime.strptime(ymd, "%Y%m%d")
            dt2 = dt1 + dt.timedelta(minutes=1439)
            tmpDict = fakeGeomag(iagaID, magCoord, dt1, dt2)
         
         
      else:
         print("more than 1 "+iagaID+" file for date "+ymd)
         raise Exception
      
      # concatenate dictionaries
      outDict = catGeomagDict(outDict, tmpDict)
            
      
   # filter out unwanted datetimes
   outDict = pruneGeomagDict(outDict, dtMin=dtBegin, dtMax=dtEnd)
      
      
   return outDict




def putIAGADaily(geomagDict, path='./', suffix='', extension='.min', merge=False):
   """
   Wrapper for writeIAGA2002 that breaks geomagDict into daily files
   """
   
   # exit early if geomagDict data array(s) empty
   if geomagDict['dateTime'].size == 0:
      return []
   
   
   iagaID = geomagDict['iagaID'].lower()
   
   # extract dtBegin and dtEnd, and convert to midN[ight]Begin and midN[ight]End
   dtBegin = geomagDict['dateTime'][0]
   dtEnd = geomagDict['dateTime'][-1]
   
   # convert dates to datetimes
   midNBegin = dt.datetime(*dtBegin.timetuple()[:3])
   midNEnd = dt.datetime(*dtEnd.timetuple()[:3]) + dt.timedelta(days=1)
   
   
   # loop over and prune datetimes in geomagDict
   filenames = []
   for i in range((midNEnd - midNBegin).days): # this should return a timedelta with units of days
      
      dtMin = midNBegin + i*dt.timedelta(days=1)
      dtMax = midNBegin + ( (i+1)*dt.timedelta(days=1) - dt.timedelta.resolution)
      tmpDict = pruneGeomagDict(geomagDict, dtMin=dtMin, dtMax = dtMax) 
      
      # generate filename
      ymd = dtMin.strftime("%Y%m%d")
      filename = path + '/' + iagaID + ymd + suffix + extension
      # call writeIAGA2002, using default filename
      filenames.append(writeIAGA2002(tmpDict, filename=filename, merge=merge))
      
         
   return filenames




def plotGeomagDict(geomagDict, filename=None):
   """
   Plot data held in the geomagDict input. 
   """
   
   import matplotlib.dates as mdates
   
   
   # get iagaID
   iagaID = geomagDict['iagaID']
   
   # get magCoord 
   magCoord = geomagDict['magCoord']
   
   # exit early if geomagDict data array(s) empty
   if geomagDict['dateTime'].size == 0:
      return []
   else:
      ut = geomagDict['dateTime']
   
   
   # default filename, and file type (i.e., ".png")
   if filename==None:
      filename = (iagaID.lower() +
                  ut[0].strftime("%Y%m%d") +
                  magCoord.lower() + ".png")
   
#   plt.clf()
   plt.figure(figsize=(16,12))
   
   print(filename)
   
   ax=plt.subplot(411)
   ax.get_yaxis().get_major_formatter().set_useOffset(False) # remove annoying formatting
   if np.isnan(geomagDict[magCoord[0]]).all():
      ax.set_ylim((-1,1))
      ax.plot(ut, np.tile(None,ut.size))
   else:
      ax.plot(ut, geomagDict[magCoord[0]])
   
   ax.set_xlim((ut[0], ut[-1]))
   ax.yaxis.tick_left()
   ax.yaxis.set_label_position('left')
   ax.locator_params(axis='y', nbins = 6)
   ax.set_ylabel('X (nT)')
   plt.setp(ax.get_xticklabels(), visible=False)
   plt.grid(True) 

   ax0 = ax # for sharing axes on subsequent subplots
   
   
   ax=plt.subplot(412, sharex=ax0)
   ax.get_yaxis().get_major_formatter().set_useOffset(False) # remove annoying formatting
   if np.isnan(geomagDict[magCoord[1]]).all():
      ax.set_ylim((-1,1))
      ax.plot(ut, np.tile(None,ut.size))
   else:
      ax.plot(ut, geomagDict[magCoord[1]])
   
   ax.set_xlim((ut[0], ut[-1]))
   ax.yaxis.tick_right()
   ax.yaxis.set_label_position('right')
   ax.locator_params(axis='y', nbins = 6)
   ax.set_ylabel('Y (nT)')
   plt.setp(ax.get_xticklabels(), visible=False)
   plt.grid(True)   
   
   
   ax=plt.subplot(413, sharex=ax0)
   ax.get_yaxis().get_major_formatter().set_useOffset(False) # remove annoying formatting
   if np.isnan(geomagDict[magCoord[2]]).all():
      ax.set_ylim((-1,1))
      ax.plot(ut, np.tile(None,ut.size))
   else:
      ax.plot(ut, geomagDict[magCoord[2]])
   
   ax.set_xlim((ut[0], ut[-1]))
   ax.yaxis.tick_left()
   ax.yaxis.set_label_position('left')
   ax.locator_params(axis='y', nbins = 6)
   ax.set_ylabel('Z (nT)')
   plt.setp(ax.get_xticklabels(), visible=False)
   plt.grid(True)   
   
   
   ax=plt.subplot(414, sharex=ax0)
   ax.get_yaxis().get_major_formatter().set_useOffset(False) # remove annoying formatting
   if np.isnan(geomagDict[magCoord[3]]).all():
      ax.set_ylim((-1,1))
      ax.plot(ut, np.tile(None,ut.size))
   else:
      ax.plot(ut, geomagDict[magCoord[3]])
   
   ax.set_xlim((ut[0], ut[-1]))
   ax.yaxis.tick_right()
   ax.yaxis.set_label_position('right')
   ax.locator_params(axis='y', nbins = 6)
   ax.set_ylabel('F (nT)')
   plt.grid(True)   
   
   dfmt = mdates.DateFormatter('%H:%M\n%d%b\n%Y')
   plt.gca().xaxis.set_major_formatter(dfmt)
   
   
   plt.subplots_adjust(hspace=0)
   plt.subplots_adjust(bottom=.16)
   
   # matplotlib opens an existing file, then replaces its contents; this does
   # not update a directory's timestamp (in Linux), which is not a big deal,
   # but can make file system-level diagnostics a little problematic...just
   # delete the file before calling savefig
   try:
      os.remove(filename)
   except:
      # sometimes Python conventions are just silly...like using try/except
      # instead of actually testing for existance of a file
      pass
   plt.savefig(filename, dpi=150)
   
   
   
   return filename





def pruneGeomagDict(geomagDict, dtMin=None, dtMax=None):
   """
   Prune unwanted elements from Geomag data dictionary, and/or add fake data if
   date range is outside of datetimes held in geomagDict
   (for now, just filter on min/max datetimes, but use keywords as input so
    that we might add other filter items in the future with little effort)
   """
   
      
   outDict = geomagDict.copy()
   
   
   iagaID = geomagDict['iagaID']
   magCoord = geomagDict['magCoord']
      
   
   # first generate fake data dictionaries to pre-/post-pad the actual data
   fakeDictPre = fakeGeomag(iagaID, magCoord, 
                            dtMin, 
                            geomagDict['dateTime'][0] - dt.timedelta(seconds=60))
   fakeDictPost = fakeGeomag(iagaID, magCoord, 
                             geomagDict['dateTime'][-1] + dt.timedelta(seconds=60), 
                             dtMax)
                              
      
   # next, prune good data
   isGood = np.equal(geomagDict['dateTime'] >= dtMin,
                     geomagDict['dateTime'] <= dtMax)
   
   outDict['dateTime'] = outDict['dateTime'][isGood]
   outDict['doy'] = outDict['doy'][isGood]
   
   outDict[magCoord[0]] = outDict[magCoord[0]][isGood]
   outDict[magCoord[1]] = outDict[magCoord[1]][isGood]
   outDict[magCoord[2]] = outDict[magCoord[2]][isGood]
   outDict[magCoord[3]] = outDict[magCoord[3]][isGood]
   
   
   # finally, concatenate fakeDicts with outDict
   outDict = catGeomagDict(fakeDictPre, outDict)
   outDict = catGeomagDict(outDict, fakeDictPost)
   
   return outDict





def catGeomagDict(geomagDict1, geomagDict2):
   """
   Concatenate Geomag dictionaries with basic checks for compatibilty
   """
  
   if (not geomagDict2 and 
       geomagDict1 and 'magCoord' in geomagDict1 and 'iagaID' in geomagDict1):
      # empty geomagDict2, valid geomagDict1
      outDict = geomagDict1.copy()
   
   elif (not geomagDict1 and 
         geomagDict2 and 'magCoord' in geomagDict2 and 'iagaID' in geomagDict2):
      # empty geomagDict1, valid geomagDict2
      outDict = geomagDict2.copy()
   
   elif ('magCoord' in geomagDict1 and 'magCoord' in geomagDict2 and
         'iagaID' in geomagDict1 and 'iagaID' in geomagDict2 and
         geomagDict1['magCoord'] == geomagDict2['magCoord'] and
         geomagDict1['iagaID'] == geomagDict2['iagaID']):
            
      
      # normally get metadata from geomagDict 2, UNLESS it is a fake dictionary
      # (which probably doesn't have most of the metadat required...this is an
      #  ugly kludge, but I'm not sure what else to do -EJR 12/2014)
      if 'fake' in geomagDict2 and geomagDict2['fake']:
         outDict = geomagDict1.copy()
      else:
         outDict = geomagDict2.copy()
      
      
      
      outDict['dateTime'] = np.concatenate((geomagDict1['dateTime'], 
                                            geomagDict2['dateTime']))
      outDict['doy'] = np.concatenate((geomagDict1['doy'], 
                                       geomagDict2['doy']))
      
      magCoord = outDict['magCoord']
      
      
      outDict[magCoord[0]] = np.concatenate((geomagDict1[magCoord[0]],
                                             geomagDict2[magCoord[0]]))
      outDict[magCoord[1]] = np.concatenate((geomagDict1[magCoord[1]],
                                             geomagDict2[magCoord[1]]))
      outDict[magCoord[2]] = np.concatenate((geomagDict1[magCoord[2]],
                                             geomagDict2[magCoord[2]]))
      outDict[magCoord[3]] = np.concatenate((geomagDict1[magCoord[3]],
                                             geomagDict2[magCoord[3]]))
      
            
   else:
      
      print(geomagDict1['iagaID'], geomagDict1['magCoord'])
      print(geomagDict2['iagaID'], geomagDict1['magCoord'])
      
      print("Geomag data dictionaries are incompatible")
      raise Exception
   
   
   return outDict




def xformGeomag(geomagDict, magCoord=None, iagaID=None, decBase=None, xform=True):
   """
   Transform a geomagDict dictionary into a different coordinate system. This can
   be a true transformation of the observations, or just the metadata held in the
   dictionary can be modified via keyword arguments with xform=False.
   
   FIXME: change how decBase is handled:
          * if decBase is in geomagDict, assume coordinates in geomagDict are
            wrt the instrument;
          * if decBase is specified as keyword argument, assume desired output
            is wrt an istrument oriented by decBase relative to geographic
            north;
          * outDict['decBase'] will always equal the keyword argument.
          
   
   INPUTS:
   geomagDict   : a dictionary holding metadata and data for geomagnetic
                  observatory data; must contain the minimumset of keys:
                  'iagaID'    : 3-character observatory code
                                  (None if no "iaga code" line in header)
                  'magCoord'    : 4-character coordinate system specifier
                                  (None if no "reported" line in header)
                  'decBase'      : declination baseline in minutes of arc;
                                  (None if no "# decbas" line in header)

                  
                  ...with identical-length lists of time stamps:
                  'datetime'    : datetime objects
                  'doy'         : days-of-year
                  
                  ...and four identical-length arrays assigned to keys that are
                  specified by the 4 magCoord characters (e.g., for heZF
                  'h'           : horizontal magnetic vector component along 
                                  axis rotated by angle decBas from geographic north
                  'e'           : horizontal vector component rotated 90 degrees from H
                  'Z'           : downward vector component
                  'F'           : field amplitude measured by scalar instrument)
   
   KEYWORDS:
   magCoord     : the coordinate system (HDZF, hdZF, HEZF, heZF, or XYZ) of
                  results; if xform=False, key value for magCoord will change,
                  and the vector component key names will be changed, but nothing 
                  is done to the actual data.
                  
   iagaID     : the (typically) 3-letter IAGA code for the observatory;
                  this only changes the iagaID dictionary key, and never
                  affects the transformation of observations
   
   decBase       : the declination baseline of the observatory; non-None values
                  change the interpretation of magCoord when transforming obs-
                  ervations, possibly overriding the decBase value in geomagDict
   
   xform:       : transform observations; if False, only modify metadata and 
                  dictionary key names
   
   OUTPUTS:
   outDict              : dictionary holding transformed elements of geomagDict
   
   """
   
   # set default values for keword arguments
   if magCoord==None:
      magCoord = geomagDict['magCoord']
      
   if iagaID==None:
      iagaID = geomagDict['iagaID']
      
   if decBase==None:
      # decBase may not always be in geomagDict
      if 'decBase' in geomagDict:
         decBase = geomagDict['decBase']
   
   
   # extract reported coordinates from dictionary
   reported = geomagDict['magCoord']
   
      
   
   # initialize outDict
   outDict = geomagDict.copy()
   # delete old keys from outDict, will be replaced later
   for elem in outDict['magCoord']:
      del outDict[elem]
   
   
   # transform observations as requested
   if magCoord[:3] == 'XYZF'[:3]:
      
      if reported[:3] == 'HDZF'[:3]:
         
         if xform:
            X, Y, Z = HDZtoXYZ(geomagDict['H'], geomagDict['D'], geomagDict['Z']) 
         else:
            X, Y, Z = geomagDict['H'], geomagDict['D'], geomagDict['Z']
            
      
      elif reported[:3] == 'hdZF'[:3]:
         
         if xform:
            X, Y, Z = hdZtoXYZ(geomagDict['h'], geomagDict['d'], geomagDict['Z'], 
                               decBase=decBase) # decBase *should* be non-None
         else:
            X, Y, Z = geomagDict['h'], geomagDict['d'], geomagDict['Z']
         
         decBase = None
         
         
      elif reported[:3] == 'HEZF'[:3]:
         
         if xform:
            X, Y, Z = HEZtoXYZ(geomagDict['H'], geomagDict['E'], geomagDict['Z'])
         else:
            X, Y, Z = geomagDict['H'], geomagDict['E'], geomagDict['Z']
            
            
      elif reported[:3] == 'heZF'[:3]:
         
         if xform:
            X, Y, Z = heZtoXYZ(geomagDict['h'], geomagDict['e'], geomagDict['Z'], 
                               decBase=decBase) # decBase *should* be non-None
         else:
            X, Y, Z = geomagDict['h'], geomagDict['e'], geomagDict['Z']
         
         decBase = None
         
      
      elif reported[:3] == 'XYZF'[:3]:
         # data is already in magCoord coordinates
         X, Y, Z = geomagDict['X'], geomagDict['Y'], geomagDict['Z']
         
         
      else:
         print("cannot convert "+reported[:3]+" coordinates into "+magCoord[:3])
         raise Exception

      # copy to outDict
      outDict['X'] = X
      outDict['Y'] = Y
      outDict['Z'] = Z
      outDict[reported[3]] = geomagDict[reported[3]]
      
      
   
   
   elif magCoord[:3] == 'HDZF'[:3]:
      
      if reported[:3] == 'XYZF'[:3]:
         
         if xform:
            H, D, Z = XYZtoHDZ(geomagDict['X'], geomagDict['Y'], geomagDict['Z'])
         else:
            H, D, Z = geomagDict['X'], geomagDict['Y'], geomagDict['Z']
            
      
      elif reported[:3] == 'HEZF'[:3]:
         
         if xform:
            H, D, Z = HEZtoHDZ(geomagDict['H'], geomagDict['E'], geomagDict['Z'])
         else:
            H, D, Z = geomagDict['H'], geomagDict['E'], geomagDict['Z']
            
            
      elif reported[:3] == 'heZF'[:3]:
         if xform:
            H, D, Z = heZtoHDZ(geomagDict['h'], geomagDict['e'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None
         else:
            H, D, Z = geomagDict['h'], geomagDict['e'], geomagDict['Z']
         
         decBase = None
         
      
      elif reported[:3] == 'hdZF'[:3]:
         if xform:
            H, D, Z = hdZtoHDZ(geomagDict['h'], geomagDict['d'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None
         else:
            H, D, Z = geomagDict['h'], geomagDict['d'], geomagDict['Z']
         
         decBase = None
         
      
      elif reported[:3] == 'HDZF'[:3]:
         # data is already in magCoord coordinates
         H, D, Z = geomagDict['H'], geomagDict['D'], geomagDict['Z']
         
         
      else:
         print("cannot convert "+reported+" coordinates into "+magCoord)
         raise Exception
      
      # copy to outDict
      outDict['H'] = H
      outDict['D'] = D
      outDict['Z'] = Z
      outDict[reported[3]] = geomagDict[reported[3]]
      
      
   
   
   elif magCoord[:3] == "HEZF"[:3]:
      
      if reported[:3] == 'XYZF'[:3]:
         
         if xform:
            H, E, Z = XYZtoHEZ(geomagDict['X'], geomagDict['Y'], geomagDict['Z'])
         else:
            H, E, Z = geomagDict['X'], geomagDict['Y'], geomagDict['Z']
         
         
      elif reported[:3] == 'HDZF'[:3]:
         
         if xform:
            H, E, Z = HDZtoHEZ(geomagDict['H'], geomagDict['D'], geomagDict['Z'])
         else:
            H, E, Z = geomagDict['H'], geomagDict['D'], geomagDict['Z']
            
         
      elif reported[:3] == 'hdZF'[:3]:
         
         if xform:
            H, E, Z = hdZtoHEZ(geomagDict['h'], geomagDict['d'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None
         else:
            H, E, Z = geomagDict['h'], geomagDict['d'], geomagDict['Z']
         
         decBase = None
         
      
      elif reported[:3] == 'heZF'[:3]:
         
         if xform:
            H, E, Z = heZtoHEZ(geomagDict['h'], geomagDict['e'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None
         else:
            H, E, Z = geomagDict['h'], geomagDict['e'], geomagDict['Z']
         
         decBase = None
         
      
      elif reported[:3] == 'HEZF'[:3]:
         # data is already in magCoord coordinates
         H, E, Z = geomagDict['H'], geomagDict['E'], geomagDict['Z']
         
      
      else:
         print("cannot convert "+reported+" coordinates into "+magCoord)
         raise Exception
      
      # copy to outDict
      outDict['H'] = H
      outDict['E'] = E
      outDict['Z'] = Z
      outDict[reported[3]] = geomagDict[reported[3]]
      
   
   
   
   elif magCoord[:3] == 'hdZF'[:3]:
      
      if reported[:3] == 'XYZF'[:3]:
         
         if xform:
            h, d, Z = XYZtohdZ(geomagDict['X'], geomagDict['Y'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None
         else:
            h, d, Z = geomagDict['X'], geomagDict['Y'], geomagDict['Z']
            
      
      elif reported[:3] == 'HEZF'[:3]:
         
         if xform:
            h, d, Z = HEZtohdZ(geomagDict['H'], geomagDict['E'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None
         else:
            h, d, Z = geomagDict['H'], geomagDict['E'], geomagDict['Z']
            
            
      elif reported[:3] == 'heZF'[:3]:
         if xform:
            h, d, Z = heZtohdZ(geomagDict['h'], geomagDict['e'], geomagDict['Z'])
         else:
            h, d, Z = geomagDict['h'], geomagDict['e'], geomagDict['Z']
            
      
      elif reported[:3] == 'HDZF'[:3]:
         if xform:
            h, d, Z = HDZtohdZ(geomagDict['H'], geomagDict['D'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None
         else:
            h, d, Z = geomagDict['H'], geomagDict['D'], geomagDict['Z']
      
      
      elif reported[:3] == 'hdZF'[:3]:
         # data is already in magCoord coordinates
         h, d, Z = geomagDict['h'], geomagDict['d'], geomagDict['Z']
         
         
      else:
         print("cannot convert "+reported+" coordinates into "+magCoord)
         raise Exception
      
      # copy to outDict
      outDict['h'] = h
      outDict['d'] = d
      outDict['Z'] = Z
      outDict[reported[3]] = geomagDict[reported[3]]
      
      
   
      
   elif magCoord[:3] == "heZF"[:3]:
      
      if reported [:3]== 'XYZF'[:3]:
         
         if xform:
            h, e, Z = XYZtoheZ(geomagDict['X'], geomagDict['Y'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None)
         else:
            h, e, Z = geomagDict['X'], geomagDict['Y'], geomagDict['Z']
         
         
      elif reported[:3] == 'HDZF'[:3]:
         
         if xform:
            h, e, Z = HDZtoheZ(geomagDict['H'], geomagDict['D'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None)
         else:
            h, e, Z = geomagDict['H'], geomagDict['D'], geomagDict['Z']
            
         
      elif reported[:3] == 'hdZF'[:3]:
         
         if xform:
            h, e, Z = hdZtoheZ(geomagDict['h'], geomagDict['d'], geomagDict['Z'])
         else:
            h, e, Z = geomagDict['h'], geomagDict['d'], geomagDict['Z']
      
      
      elif reported[:3] == 'HEZF'[:3]:
         
         if xform:
            h, e, Z = HEZtoheZ(geomagDict['H'], geomagDict['E'], geomagDict['Z'],
                               decBase=decBase) # decBase should be non-None
         else:
            h, e, Z = geomagDict['H'], geomagDict['E'], geomagDict['Z']
      
      
      elif reported[:3] == 'heZF'[:3]:
         # data is already in magCoord coordinates
         h, e, Z = geomagDict['h'], geomagDict['e'], geomagDict['Z']
         
         
      else:
         print("cannot convert "+reported+" coordinates into "+magCoord)
         raise Exception
      
      # copy to outDict
      outDict['h'] = h
      outDict['e'] = e
      outDict['Z'] = Z
      outDict[reported[3]] = geomagDict[reported[3]]
   
   

   
   else:
      print("unrecognized magnetic coordinate system")
      raise Exception
   
   
   
   
   
   # finally, put in remaining metadata required to determine if two
   # IAGA-2002 time series dictionaries can be properly 'concatenated'.
   outDict['iagaID'] = iagaID
   outDict['magCoord'] = magCoord[:3]+reported[3]
   outDict['decBase'] = decBase
   
   return outDict
   








def XYZtoHDZ(X, Y, Z):
   """
   Convert XYZ geomagnetic coordinates to HDZ.
   
   INPUTS:
   X                    : geographic northward component (nT)
   Y                    : geographic eastward comonent (nT)
   Z                    : downward component (nT)
   
   KEYWORDS:
   
   OUTPUTS:
   H                    : horizontal component (nT)
   D                    : declination (min. of arc)
   Z                    : downward component (nT) - just copied through
   """
   
   H = np.sqrt(X**2 + Y**2)
   D = np.arctan2(Y,X) * 180./np.pi * 60.
      
   return H,D,Z



def XYZtohdZ(X, Y, Z, decBase=0):
   """
   Convert XYZ geomagnetic coordinates to hdZ.
   
   INPUTS:
   X                    : geographic northward component (nT)
   Y                    : geographic eastward comonent (nT)
   Z                    : downward component (nT)
   
   KEYWORDS:
   decBase               : declination baseline (min. of arc)
   
   OUTPUTS:
   h                    : component aligned with magnetometer primary axis (nT)
   d                    : angle clockwise from h (min. of arc)
   Z                    : downward component (nT) - just copied through
   """
   
   H = np.sqrt(X**2 + Y**2)
   D = np.arctan2(Y,X) * 180./np.pi * 60.
   
   d = D - decBase
   h = H * np.cos(d * 1./60 * np.pi/180)
   
   return h,d,Z



def HDZtoXYZ(H, D, Z):
   """
   Convert HDZ geomagnetic coordinates to XYZ.
   
   INPUTS:
   H                    : horizontal component (nT)
   D                    : declination (min. of arc)
   Z                    : downward component (nT) - just copied through
   
   KEYWORDS:
   
   OUTPUTS:
   X                    : geographic northward component (nT)
   Y                    : geographic eastward comonent (nT)
   Z                    : downward component (nT)
   """
      
   X = H * np.cos(D * 1./60 * np.pi/180)
   Y = H * np.sin(D * 1./60 * np.pi/180)
   
   return X,Y,Z



def XYZtoHEZ(X, Y, Z):
   """
   Convert XYZ geomagnetic coordinates to HEZ.
   
   INPUTS:
   X                    : geographic northward component (nT)
   Y                    : geographic eastward comonent (nT)
   Z                    : downward component (nT)
   
   KEYWORDS:
   
   OUTPUTS:
   H                    : horizontal comonent (nT)
   E                    : eastward comopnent (nT)
   Z                    : downward component (nT) - just copied through   
   """
   
   H = np.sqrt(X**2 + Y**2)
   E = Y
   
   return H,E,Z



def XYZtoheZ(X, Y, Z, decBase=0):
   """
   Convert XYZ geomagnetic coordinates to heZ.
   
   INPUTS:
   X                    : geographic northward component (nT)
   Y                    : geographic eastward comonent (nT)
   Z                    : downward component (nT)
   
   KEYWORDS:
   decBase               : declination baseline (min. of arc)
   
   OUTPUTS:
   h                    : component aligned with magnetometer primary axis (nT)
   e                    : component 90 degrees clockwise from h (nT)
   Z                    : downward component (nT) - just copied through   
   """
   
   H = np.sqrt(X**2 + Y**2)
   D = np.arctan2(Y, X)
   d = D - decBase * 1./60 * np.pi/180
   
   h = H * np.cos(d)
   e = H * np.sin(d)
   
   return h,e,Z



def HEZtoXYZ(H, E, Z):
   """
   Convert HEZ geomagnetic coordinates to XYZ.
   
   INPUTS:
   H                    : horizontal comonent (nT)
   E                    : eastward comopnent (nT)
   Z                    : downward component (nT) - just copied through   
   
   KEYWORDS:
   
   OUTPUTS:
   X                    : geographic northward component (nT)
   Y                    : geographic eastward comonent (nT)
   Z                    : downward component (nT)
   """
   
   D = np.arcsin(E/H)
   
   X = H * np.cos(D)
   Y = H * np.sin(D)
   
   return X,Y,Z



def heZtoXYZ(h, e, Z, decBase=0):
   """
   Convert heZ geomagnetic coordinates to XYZ.
   
   INPUTS:
   h                    : component aligned with magnetometer primary axis (nT)
   e                    : component 90 degrees clockwise from h (nT)
   Z                    : downward component (nT) - just copied through   
   
   KEYWORDS:
   decBase               : declination baseline (min. of arc)
   
   OUTPUTS:
   X                    : geographic northward component (nT)
   Y                    : geographic eastward comonent (nT)
   Z                    : downward component (nT)
   """
   
   H = np.sqrt(h**2 + e**2)
   
   d = np.arctan2(e,h)
   D = d + decBase * np.pi/180. * 1./60.
   
   X = H * np.cos(D)
   Y = H * np.sin(D)
   
   return X,Y,Z



def hdZtoXYZ(h, d, Z, decBase=0):
   """
   Convert hdZ geomagnetic coordinates to XYZ.
   
   INPUTS:
   h                    : component aligned with magnetometer primary axis (nT)
   d                    : angle clockwise from h (min. of arc)
   Z                    : downward component (nT) - just copied through   
   
   KEYWORDS:
   decBase               : declination baseline (min. of arc)
   
   OUTPUTS:
   X                    : geographic northward component (nT)
   Y                    : geographic eastward comonent (nT)
   Z                    : downward component (nT)
   """
   
   H = h / np.cos(d * 1./60 * np.pi/180)
   D = (d + decBase) * 1./60 * np.pi/180
   
   X = H * np.cos(D)
   Y = H * np.sin(D)
   
   return X,Y,Z



def HDZtoHEZ(H, D, Z):
   """
   Convert HDZ geomagnetic coordinates to HEZ.
   
   INPUTS:
   H                    : horizontal component (nT)
   D                    : declination (min. of arc)
   Z                    : downward component (nT) - just copied through
   
   KEYWORDS:
   
   OUTPUTS:
   H                    : horizontal comonent (nT)
   E                    : eastward comopnent (nT)
   Z                    : downward component (nT) - just copied through   
   """
   
   
   E = H * np.sin(D * 1./60 * np.pi/180)
   
   return H,E,Z



def HEZtoHDZ(H, E, Z):
   """
   Convert HEZ geomagnetic coordinates to HDZ.
   
   INPUTS:
   H                    : horizontal comonent (nT)
   E                    : eastward comopnent (nT)
   Z                    : downward component (nT) - just copied through   
   
   KEYWORDS:
   
   OUTPUTS:
   H                    : horizontal component (nT)
   D                    : declination (min. of arc)
   Z                    : downward component (nT) - just copied through
   """
   
   D = np.arcsin(E/H) * 60. * 180/np.pi
      
   return H,D,Z



def HDZtoheZ(H, D, Z, decBase=0):
   """
   Convert HDZ geomagnetic coordinates to heZ.
   
   INPUTS:
   H                    : horizontal comonent (nT)
   D                    : declination (min. of arc)
   Z                    : downward component (nT) - just copied through   
   
   KEYWORDS:
   decBase               : declination baseline (min. of arc)
   
   OUTPUTS:
   h                    : component aligned with magnetometer primary axis (nT)
   e                    : component 90 degrees clockwise from h (nT)
   Z                    : downward component (nT) - just copied through   
   """
   
   d = (D - decBase) * 1./60 * np.pi/180
   h = H * np.cos(d)
   e = H * np.sin(d)
   
   return h,e,Z



def HEZtohdZ(H, E, Z, decBase=0):
   """
   Convert HEZ geomagnetic coordinates to HDZ.
   
   INPUTS:
   H                    : horizontal comonent (nT)
   E                    : eastward comopnent (nT)
   Z                    : downward component (nT) - just copied through   
   
   KEYWORDS:
   decBase               : declination baseline (min. of arc)
   
   OUTPUTS:
   h                    : component aligned with magnetometer primary axis (nT)
   d                    : angle clockwise from h (min. of arc)
   Z                    : downward component (nT) - just copied through
   """
   
   D = np.arcsin(E/H) * 60. * 180/np.pi
   d = D - decBase
   h = H * np.cos(d * 1./60. * np.pi/180)
   
   return h,d,Z



def hdZtoHEZ(h, d, Z, decBase=0):
   """
   Convert hdZ geomagnetic coordinates to HEZ.
   
   INPUTS:
   h                    : component aligned with magnetometer primary axis (nT)
   d                    : angle clockwise from h (min. of arc)
   Z                    : downward component (nT) - just copied through
   
   KEYWORDS:
   decBase               : declination baseline (min. of arc)
   
   OUTPUTS:
   H                    : horizontal comonent (nT)
   E                    : eastward comopnent (nT)
   Z                    : downward component (nT) - just copied through   
   """
   
   H = h / np.cos(d * 1./60 * np.pi/180)
   E = H * np.sin((d + decBase) * 1./60 * np.pi/180)
  
   return H,E,Z



def heZtoHDZ(h, e, Z, decBase=0):
   """
   Convert HEZ geomagnetic coordinates to HDZ.
   
   INPUTS:
   h                    : component aligned with magnetometer primary axis (nT)
   e                    : component 90 degrees clockwise from h (nT)
   Z                    : downward component (nT) - just copied through   
   
   KEYWORDS:
   decBase               : declination baseline (min. of arc)
   
   OUTPUTS:
   H                    : horizontal component (nT)
   D                    : declination (min. of arc)
   Z                    : downward component (nT) - just copied through
   """
   
   d = np.arctan2(e,h)
   H = h / np.cos(d)
   D = decbas + d * 60. * 180./np.pi

   return H,D,Z



def HDZtohdZ(H, D, Z, decBase=0):
   """
   Convert geographic HDZ into magnetometer-aligned hdZ
   """

   d = D - decBase
   h = H * np.cos(d * 1./60 * np.pi/180)
   
   return h,d,Z
   
   
def hdZtoHDZ(h, d, Z, decBase=0):
   """
   Convert magnetometer-aligned hdZ to geographic HDZ
   """
   
   D = d + decBase
   H = h / np.cos(d * 1./60 * np.pi/180)
   
   return H,D,Z


def HEZtoheZ(H, E, Z, decBase=0):
   """
   Convert geographic HEZ into magnetometer-aligned heZ
   """
   D = np.arctan2(E,H)
   d = D - decBase * 1./60 * np.pi/180
   h = H * np.cos(d)
   e = h * np.tan(d)
   
   return h,e,Z
   
   
def heZtoHEZ(h, e, Z, decBase=0):
   """
   Convert magnetometer-aligned heZ to geographic HEZ
   """
   
   d = np.arctan2(e,h)
   H = h / np.cos(d)
   E = H * np.sin(d + decBase * 1./60 * np.pi/180)

   return H,E,Z



def heZtohdZ(h, e, Z):
   """
   Convert magnetometer-aligned heZ to magnetometer-aligned hdZ
   """
   
   d = np.arctan2(e,h)

   return h,d,Z



def hdZtoheZ(h, d, Z):
   """
   Convert magnetometer-aligned hdZ to magnetometer-aligned heZ
   """
   
   e = h * np.tan(d * 1./60 * np.pi/180)

   return h,e,Z

