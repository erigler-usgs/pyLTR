from pyLTR.Models import Model
from numpy import arange, vstack, reshape
import scipy.io

import datetime
import glob
import os
import re

class AMPERE(Model):
    """
    Implementation of AMPERE NetCDF Import
    
    Standard Variables include
        mlt - Magnetic Local Time
        colat - Colatitude
        Jr - Radial current
        
    """
    
    def __init__(self,fileName):
        """
        Parameters - Name of file to open
        """
        
        self.__io =  scipy.io.netcdf_file(fileName)
        self.__timeRange = []
        self.__keys = []
        self.getTimeRange()
        
    def getTimeRange(self):
        """
        Returns a list of datetime objects of the start times for each set in the file
        """
        start_yr = self.__io.variables['start_yr']
        start_mo = self.__io.variables['start_mo']
        start_dy = self.__io.variables['start_dy']
        start_hr = self.__io.variables['start_hr']
        start_mt = self.__io.variables['start_mt']
        start_sc = self.__io.variables['start_sc']
        
        if not self.__timeRange:
            for i in arange(len(start_hr.data)):
                self.__timeRange.append(datetime.datetime(
                                        start_yr[i],start_mo[i],start_dy[i],
                                        start_hr[i],start_mt[i],start_sc[i]))
                                    
        return self.__timeRange
        
    def getVarNames(self):
        """
        Returns a list of variable names
        """
        
        self.__keys = list(self.__io.variables.keys())
        return self.__keys
        
    def getAttributeNames(self):
        """
        Returns a list of Attribute names 
        """
        
        return list(self.__io._attributes.keys())
        
    def readAttribute(self,attrName):
        """
        Returns the specified Attribute
        """
        
        assert(attrName in self.__io.attributes)
        return self.__io.attributes[attrName]
        
    def read(self,varName,time):
        """
        Returns the specifed Variable from the specfied time in the file.  Use an
        item in the lists returned by getVarNames() and getTimeRange()
        to obtain valid data.
        
        If time is not found last timestep in array is returned.
        
        """
        
        # Need to determine block to read of netCDF file so get index of time
        
        if not self.__timeRange:
            self.getTimeRange()
        else:
            for i,date in enumerate(self.__timeRange):
                if (date - time).total_seconds() == 0:
                    index = i
        
        nlat = self.__io.variables['nlat'][index]
        nlon = self.__io.variables['nlon'][index]
        
        data = reshape(self.__io.variables[varName][index],(nlon,nlat))
        data = vstack([data,data[0,:]])
        
        return data
        
        
        
        
        
        
        
        
    
        
        
        
        
    