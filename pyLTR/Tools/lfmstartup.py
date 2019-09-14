import numpy as n
from pyhdf.SD import SD, SDC
import sys, os

class lfmstartup():
   """
   Class for creating initial input files for LFM code
   Eventually this will include LFM, MFLFM
   Right now only LFM is supported
   """
   def __init__(self,fileName,dims,nspecies=1):
        """
        Create the HDF file
         Inputs:
         fileName - Name of file to create
         dims - (NI,NJ,NK) tuple of grid size
         nspecies - number of ion speices (default 1)
        """
        (self.ni,self.nj,self.nk)=dims
        self.fileName = fileName
        self.varNames = ['X_grid','Y_grid','Z_grid',
                         'rho_','vx_','vy_','vz_','c_','bx_','by_','bz_',
                         'bi_','bj_','bk_','ei_','ej_','ek_','ai_','aj_','ak_']
        if (nspecies > 1):
            for i in range(1,nspecies+1):
                for var in ['rho_.','vx_.','vy_.','vz_.','c_.']:
                    self.varNames.append(var+str(i))
                    
        self.varUnits = ['cm','cm','cm',
                        'g/cm^3','cm/s','cm/s','cm/s','cm/s',
                        'gauss','gauss','gauss',
                        'gauss*cm^2','gauss*cm^2','gauss*cm^2',
                        'cgs*cm','cgs*cm','cgs*cm','dummy','dummy','dummy']
        
        if (nspecies > 1):
            for i in range(1,nspecies+1):
                for var in ['g/cm^3','cm/s','cm/s','cm/s','cm/s']:
                    self.varUnits.append(var)
                    
   def open(self,mjd=0.0,tzero=3000.0):
       """
       Open the HDF file and set the global attributes
       Inputs:
           MJD - Modified Julian Date - default 0.0
           tzero - Solar wind initialization time - default 3000.0
           
       """
       self.f = SD(self.fileName,mode=SDC.WRITE|SDC.CREATE)
       self.setGlobalAttr(mjd,tzero)
       self.initVar()
       
       return
       
   def setGlobalAttr(self,mjd,tzero):
        self.f.attr('mjd').set(SDC.FLOAT64,mjd)
        self.f.attr('time_step').set(SDC.INT32,0)
        self.f.attr('time_8byte').set(SDC.FLOAT64,0.)
        self.f.attr('time').set(SDC.FLOAT32,0.)
        self.f.attr('tilt_angle').set(SDC.FLOAT32,0.)
        self.f.attr('tzero').set(SDC.FLOAT32,tzero)
        self.f.attr('file_contents').set(SDC.CHAR,'a')
        self.f.attr('dipole_moment').set(SDC.CHAR,'b')
        self.f.attr('written_by').set(SDC.CHAR,'Python initialzer')
        
        return
   
   def initVar(self):
        
        vars = {}
        for varName,varUnit in zip(self.varNames,self.varUnits):
             vars[varName] = self.f.create(varName,SDC.FLOAT32,
                                           (self.nk+1,self.nj+1,self.ni+1))
             vars[varName].attr('ni').set(SDC.INT32,self.ni+1)
             vars[varName].attr('nj').set(SDC.INT32,self.nj+1)
             vars[varName].attr('nk').set(SDC.INT32,self.nk+1)
             vars[varName].attr('units').set(SDC.CHAR,varUnit)
            
             vars[varName][:]  = n.zeros((self.nk+1,self.nj+1,self.ni+1),
                                       dtype='float32')
   
   def writeVar(self,varName,arr):
        """
        Writes Array to HDF File
        Inputs
          varName - Name of variable to add
          arr - 3d array to add to file
        """
        iend = arr.shape[2]
        jend = arr.shape[1]
        kend = arr.shape[0]
        self.f.select(varName)[:kend,:jend,:iend]=arr.astype('float32')
        
        return
        
   def close(self):
        self.f.end()
        
        return
          
        
 
                                
       
    