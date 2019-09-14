from pyLTR.Models import Model
from pyLTR.Models.Hdf4Io import Hdf4Io
from pyLTR.Models.Hdf5Io import Hdf5Io

import numpy

import datetime
import glob
import os
import re

class IPWM(Model):
    """
    Implementation class for Model I/O.
    Also has methods for basic conversions of IPWM data
    from the native units to physical units.
    """

    def __init__(self, runPath, runName,ext='.hdf'):
        """
        Parameters:
          runPath: path to directory containing IPWM data files.
          runName: Optional parameter.  When specified, it searches
          runPath/runName* for files.  This is useful for single
          directories containing multiple runs.
        """
        Model.__init__(self, runPath, runName)

        filePrefix = os.path.join(self.runPath, self.runName)
        self.__fileList = glob.glob(filePrefix + '*_ipwm_*'+ext)
        self.__fileList.sort()

        #
        # regex is used to determine filename convention (eg. UT
        # '2012-07-13T02-15-00' vs Timestep '0008000') and to obtain
        # runName if it is undefined.
        # 
        regex = (r'^([\S\-]+)' + # \S is "digits, letters and underscore".
                 r'_ipwm_' + 
                 r'('
                 r'\d{4}\-\d{2}\-\d{2}T\d{2}\-\d{2}\-\d{2}Z' +  # UT-based filename
                 r'|' + 
                 r'\d{7}'  # timestep-based filename
                 r')' +
                 ext +                     
                 r'$')
        r = re.match(regex, os.path.basename(self.__fileList[0]))
        if not r:
            raise Exception('Having trouble identifying IPWM files.  Are you looking at the correct directory?')
        if (len(r.groups()) != 2):
            raise Exception('Having trouble identifying IPWM files.  Are you looking at the correct directory?')
        
        if not self.runName:
            self.runName = r.groups()[0]
        
        if len(r.groups()[1]) == 7:
            # Timestep filenames (eg. 'RunName_mhd_0008000.hdf')
            self.__timestepFilenames = True
        else:
            # UT-based filenames (eg. 'RunName_mhd_2012-07-13T02-47-00Z.hdf')
            self.__timestepFilenames = False
        
        # Pick the HDF4 or HDF5 IO Object based upon file extension
        if (ext == '.hdf'):
            self.__io = Hdf4Io( self.__fileList[0] )
        else:
            self.__io = Hdf5Io( self.__fileList[0] )
        self.runExt = ext

        # Set list of datetime objects corresponding to self.__fileList
        self.__timeRange = []
        self.getTimeRange()


    def getTimeRange(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available.
        
        IPWM has files named like
        '[runName]_ipwm_1995-03-21T16-00-00Z.hdf', one time step per
        file.

        Assumes self.__timestepFilenames boolean is set.
        """
        if not self.__timeRange:
            if self.__timestepFilenames:
                self.__timeRange = self.__getTimeRange_timestep()
            else:
                self.__timeRange = self.__getTimeRange_ut()

        return self.__timeRange


    def __getTimeRange_ut(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available.
        
        IPWM has files named like
        '[runName]_ipwm_1995-03-21T16-00-00Z.hdf', one time step per
        file.
        """        
        for f in self.__fileList:
            regex = ( r'^' + self.runName + '_ipwm_' +
                      r'(\d{4})\-(\d{2})\-(\d{2})' +
                      r'T' +
                      r'(\d{2})\-(\d{2})-(\d{2})' +
                      r'Z'+
                      self.runExt + r'$' )
            
            r = re.match(regex, os.path.basename(f))
            
            assert( len(r.groups()) == 6 )
            t = [ int(match) for match in r.groups() ]
            self.__timeRange.append( datetime.datetime(year=t[0],
                                                       month=t[1],
                                                       day=t[2],
                                                       hour=t[3],
                                                       minute=t[4],
                                                       second=t[5]) )
            
        return self.__timeRange

    #Should not be used for IPWM??
    def __getTimeRange_timestep(self):
        """
        Returns a list of datetime objects corresponding to all the
        time-varrying data available.
        
        LFM has files named like 
        '[runName]_mhd_069000.hdf',when in step dump mode.
        """

        # Ask user to input starting time...
        format_regex = "^([\s]*\d{4})[\s]+(0?[1-9]|1[012])[\s]+(0?[1-9]|[12]\d|3[01])[\s]+(0?\d|1\d|2[0-4])[\s]+([0-5]?\d?|60)[\s]+([0-5]?\d?|60)[\s]*$"
        while True:
            line = eval(input('Enter start time of run (YYYY MM DD HH MM SS):'))
            r = re.match(format_regex,line)
            if r:
                vals = [int(s) for s in line.split()]
                runStart=datetime.datetime(year=vals[0],month=vals[1],day=vals[2],
                                           hour=vals[3],minute=vals[4],second=vals[5])
                break
            else:
                print('Invaild entry')
        
        # Calculate the time range...
        for f in self.__fileList:

            self.__io.setFilename(f)
            #hdf = self.__getHdfObj()
            #assert(hdf.attributes().has_key('time'))
            #seconds = hdf.attributes()['time']

            #try:
            elapsedSeconds = self.__io.readAttribute('time_8byte', None)
            #except hdferror:
            #    elapsedSeconds = self.__io.readAttribute('time', None)
            date_time = runStart+datetime.timedelta(seconds=elapsedSeconds-3000.0)
            self.__timeRange.append(date_time)
        
        return self.__timeRange
        

    def __setTimeValue(self, time):
        """
        Select a particular time slice.  Use getTimeRange() to see all
        available data.
        """
        # Check the input:
        assert( isinstance(time, datetime.datetime) )
        
        # This will raise an exception if the time slice is missing:
        i = (self.__timeRange).index(time)
       
        filename=self.__fileList[i]
        
        # Set this filename for IO.
        self.__io.setFilename(filename)

    def getAttributeNames(self):
        """
        Returns a list of attributes available in the dataset.    
        """
        return( self.__io.getAttributeNames() )

    def readAttribute(self, attrName, time):
        """
        Returns an attribute at a particular time.  Use an item in the
        lists returned by getAttributeNames() and getTimeRange() to
        obtain valid data.
        """
        self.__setTimeValue(time)
        return self.__io.readAttribute(attrName, None)
    def getVarNames(self):
        """
        Returns a list of variables available in the dataset.    
        """
        return( self.__io.getVarNames() )

    def read(self, varName, time, start=None, count=None, stride=None):
        """
        Returns a variable (varName) at a particular time.  Use an
        item in the lists returned by getVarNames() and getTimeRange()
        to obtain valid data.

        WARNING:  start,count,stride hasn't been tested extensively!
        
        FIXME: IPWM HDF4 files are written in Fortran ordering.
        Accessing a single element via start index is unpredictable.
        Need to read *all* the data and numpy.reshape it before
        accessing individual elements!
        """

        self.__setTimeValue(time)

        data = self.__io.read(varName, None, start, count, stride)

        #Put everything back into Fortran/MATLAB order
        s = data.shape
        data = numpy.reshape(data.ravel(), s[::-1], order='F')

        return data

    #Utility to extract dimensioned state variables from q
    def exdimstate(self,varName, q, bms):
        #params from params.f90
        ni=4
        nih=6
        nit=10
        nnt=8
        con=0
        conh=4
        mom=10
        enmom=14
        ien=14
        enien=18
        een=19
        ihf=19
        nvar=22
        nneut=8
        
        pthp=1
        pthep=2
        ptop=3
        ptenop=4
        ptnp=5
        pto2p=6
        ptn2p=7
        ptnop=8
        ptop2d=9
        ptop2p=10
        pth=1
        pthe=5
        pto=2
        ptn=7
        pto2=4
        ptn2=6
        ptno=3
        ptn2d=8

        #parameters for nondimensionalization
        #n0=1e4 cm^-3
        #t0=1000 K
        #v0=sqrt(kB T0 / amu) in cm/s
        #p0ev=n0 kB T0 in eV cm^-3
        n0=1e4
        t0=1e3
        v0=2.8834809e5
        p0ev=8.61733e2
        h0=3.9810792e-04
        
        ami = numpy.zeros(nit)
        amn = numpy.zeros(nnt)

        ami[pthp-1]  = 1.0
        ami[pthep-1] = 4.0
        ami[ptop-1]  = 16.0
        ami[ptop2d-1]= 16.0
        ami[ptop2p-1]= 16.0
        ami[ptnp-1]  = 14.0
        ami[pto2p-1] = 32.0
        ami[ptn2p-1] = 28.0
        ami[ptnop-1] = 30.0
        
        amn[pth-1] = 1.0
        amn[pthe-1]= 4.0
        amn[pto-1] = 16.0
        amn[ptn-1] = 14.0
        amn[pto2-1]= 32.0
        amn[ptn2-1]= 28.0
        amn[ptno-1]= 30.0
        amn[ptn2d-1]=14.0

        shq=q.shape
        nvar=shq[0]
        nnz=shq[1]
        nreal=shq[2]

        if (varName is 'Ni'):
            deni=numpy.zeros((nit,nnz,nreal))
            for j in range(nit):
                deni[j,:,:]=q[j,:,:]*bms[:,:]*n0
            return deni #cm^-3
        elif (varName is 'Ne'): 
            #quasineutrality 
            #FIXME: neglects suprathermal electrons at the moment
            ne=q[0,:,:]*bms[:,:]*n0
            for j in range(1,nit):
                ne[:,:]=ne[:,:]+q[j,:,:]*bms[:,:]*n0
            return ne #cm^-3
        elif (varName is 'vi'):
            vi=numpy.zeros((ni,nnz,nreal))
            for j in range(ni):
                vi[j,:,:]=q[mom+j,:,:]/q[j,:,:]*v0
            return vi #cm/s
        elif (varName is 've'):
            #current continuity 
            #FIXME: neglects suprathermal electrons and FAC at the moment
            ne[:,:]=q[0,:,:]
            for j in range(1,nit):
                ne[:,:]=ne[:,:]+q[j,:,:]
            ve[:,:]=q[mom,:,:]
            for j in range(1,ni):
                ve[:,:]=ve[:,:]+q[mom+j,:,:]
            ve=ve/ne*v0
            return ve #cm/s
        elif (varName is 'Te'):
            ne=q[0,:,:]
            for j in range(1,nit):
                ne[:,:]=ne[:,:]+q[j,:,:]
            Te=(2.0/3.0)*q[een-1,:,:]/ne[:,:]*t0
            return Te #K
        elif (varName is 'Ti'):
            Ti=numpy.zeros((ni-1,nnz,nreal))
            for j in range(ni-1):
                Ti[j,:,:]=(2.0/3.0)*(q[ien+j,:,:]-0.5*q[mom+j,:,:]**2./q[j,:,:])/q[j,:,:]*ami[j]*t0
            return Ti #K
        elif (varName is 'Tif'):
            Ti=numpy.zeros((nit,nnz,nreal))
            for j in range(ni-1):
                Ti[j,:,:]=(2.0/3.0)*(q[ien+j,:,:]-0.5*q[mom+j,:,:]**2./q[j,:,:])/q[j,:,:]*ami[j]*t0
            for j in range(ni,nit):
                Ti[j,:,:]=Ti[ptop-1,:,:]
            return Ti #K   
        elif (varName is 'enp'):
            enp=0.5*16*1.67e-27*1e-4/1.6e-19*v0**2*q[enien-1,:,:]/(0.5*q[ptenop-1,:,:])-(q[mom+ptenop-1,:,:]/q[ptenop-1,:,:])**2
            return enp #eV
        elif (varName is 'hi'):
            hi=numpy.zeros((ni-1,nnz,nreal))
            for j in range(ni-1):
                hi[j,:,:]=q[ihf+j,:,:]*bms[:,:]*ami(j)*h0
            return hi #erg cm^-2 s^-1
    
    #Utility to extract the dimensioned ambipolar electric field
    def exdimeeambi(self,eeambi):
        #parameters for nondimensionalization
        #n0=1e4 cm^-3
        #t0=1000 K
        #v0=sqrt(kB T0 / amu) in cm/s
        #p0ev=n0 kB T0 in eV cm^-3
        n0=1e4
        t0=1e3
        v0=2.8834809e5
        p0ev=8.61733e2
        h0=3.9810792e-04
    
        eeambiev = eeambi*p0ev/(n0*v0)
        return eeambiev #eV/m

    #Utility to compute field-aligned ambipolar potential drop
    def exambipotential(self,eeambiev,dels):
        #parameters for nondimensionalization
        #n0=1e4 cm^-3
        #t0=1000 K
        #v0=sqrt(kB T0 / amu) in cm/s
        #p0ev=n0 kB T0 in eV cm^-3
        n0=1e4
        t0=1e3
        v0=2.8834809e5
        p0ev=8.61733e2
        h0=3.9810792e-04
        V=0.0*eeambiev
        sh=eeambiev.shape()
        nnz=sh[0]
        npts=sh[1]
        for i in range(nnz):
            V[i,:]=V[i-1,:]+eeambiev[i,:]*dels[:,i]*v0 
            #dels is assumed to be dimensionless
        for k in range(npts):
            minV=min(V[:,k])
            V[:,k]=V[:,k]-minV

        return V #Volts

    #Utility to compute collision frequencies
    def exnuin(self,denn,ti,tn):
        #params from params.f90
        ni=4
        nih=6
        nit=10
        nnt=8
        con=0
        conh=4
        mom=10
        enmom=14
        ien=14
        enien=18
        een=19
        ihf=19
        nvar=22
        nneut=8
        
        pthp=1
        pthep=2
        ptop=3
        ptenop=4
        ptnp=5
        pto2p=6
        ptn2p=7
        ptnop=8
        ptop2d=9
        ptop2p=10
        pth=1
        pthe=5
        pto=2
        ptn=7
        pto2=4
        ptn2=6
        ptno=3
        ptn2d=8

        #parameters for nondimensionalization
        #n0=1e4 cm^-3
        #t0=1000 K
        #v0=sqrt(kB T0 / amu) in cm/s
        #p0ev=n0 kB T0 in eV cm^-3
        n0=1e4
        t0=1e3
        v0=2.8834809e5
        p0ev=8.61733e2
        h0=3.9810792e-04
        
        ami = numpy.zeros(nit)
        amn = numpy.zeros(nnt)

        ami[pthp-1]  = 1.0
        ami[pthep-1] = 4.0
        ami[ptop-1]  = 16.0
        ami[ptop2d-1]= 16.0
        ami[ptop2p-1]= 16.0
        ami[ptnp-1]  = 14.0
        ami[pto2p-1] = 32.0
        ami[ptn2p-1] = 28.0
        ami[ptnop-1] = 30.0
        
        amn[pth-1] = 1.0
        amn[pthe-1]= 4.0
        amn[pto-1] = 16.0
        amn[ptn-1] = 14.0
        amn[pto2-1]= 32.0
        amn[ptn2-1]= 28.0
        amn[ptno-1]= 30.0
        amn[ptn2d-1]=14.0
           
        #neutral polarizabilities
        alpha0=numpy.zeros(nneut)
        alpha0[pth-1]  = 0.67
        alpha0[pthe-1] = 0.21
        alpha0[ptn-1]  = 1.10
        alpha0[pto-1]  = 0.79
        alpha0[ptn2-1] = 1.76
        alpha0[ptno-1] = 1.74
        alpha0[pto2-1] = 1.59
 
           
        shq=denn.shape
        nnz=shq[0]
        nreal=shq[-1]
        
        #initialize to zero
        nuin=numpy.zeros((nnz,nit,nneut,nreal))
    
        #collision frequencies/factors
        
        #hydrogen (H)
        
        j = pthp-1
        for nn in range(7):
            if ( nn == pto-1 ):
               teff    = ti[j,:,:] 
               fac     = ( 1.00 - .047 * numpy.log10(teff) ) ** 2
               tfactor = numpy.sqrt(teff) * fac
               nuin[:,j,nn,:]  = 6.61e-11 * denn[:,nn,:] * tfactor
            elif (nn==pth-1):
               teff=0.5*(ti[j,:,:]+tn[:,:])
               fac = (1-0.083*numpy.log10(teff))**2.
               tfactor=numpy.sqrt(teff)*fac
               nuin[:,j,nn,:] = 2.65e-10 * denn[:,nn,:] * tfactor
            else:
               amuf    = ami[j] * amn[nn] / ( ami[j] + amn[nn] )
               amimn   = amn[nn] / ( ami[j] + amn[nn] )
               nufacin = 2.59e-9 / numpy.sqrt(amuf) * amimn * numpy.sqrt(alpha0[nn])
               nuin[:,j,nn,:] = nufacin * denn[:,nn,:]

        
        #helium (He)
        
        j = pthep-1
        for nn in range(7):    
            if ( nn==pthe-1):
               teff=0.5*(ti[j,:,:]+tn[:,:])
               fac = (1-0.093*numpy.log10(teff))**2.
               tfactor=numpy.sqrt(teff)*fac
               nuin[:,j,nn,:] = 8.73e-11 * denn[:,nn,:] * tfactor
            else:
               amuf    = ami[j] * amn[nn] / ( ami[j] + amn[nn] )
               amimn   = amn[nn] / ( ami[j] + amn[nn] )
               nufacin = 2.59e-9 / numpy.sqrt(amuf) * amimn * numpy.sqrt(alpha0[nn])
               nuin[:,j,nn,:] = nufacin * denn[:,nn,:]

        #nitrogen (N)
        j = ptnp-1
        for nn in range(7):
            if (nn==ptn-1):
               teff=0.5*(ti[j,:,:]+tn[:,:])
               fac = (1-0.063*numpy.log10(teff))**2.
               tfactor=numpy.sqrt(teff)*fac
               nuin[:,j,nn,:] = 3.83e-11*denn[:,nn,:]*tfactor
            else:
               amuf    = ami[j] * amn[nn] / ( ami[j] + amn[nn] )
               amimn   = amn[nn] / ( ami[j] + amn[nn] )
               nufacin = 2.59e-9 / numpy.sqrt(amuf) * amimn * numpy.sqrt(alpha0[nn])
               nuin[:,j,nn,:] = nufacin * denn[:,nn,:]

        #oxygen (O)
        
        j = ptop-1
        for nn in range(7):    
            if ( nn==pto-1 ):
               teff    = 0.5 * ( ti[j,:,:] + tn[:,:] )
               fac     = ( 1.0 - .064 * numpy.log10(teff) ) ** 2
               tfactor = numpy.sqrt(teff) * fac
               nuin[:,j,nn,:]  = 3.67e-11 * denn[:,nn,:] * tfactor
            elif (nn==pth-1):
               nuin[:,j,nn,:]=4.63e-12*denn[:,nn,:]*numpy.sqrt(tn[:,:]+0.0625*ti[j,:,:])
            else:
               amuf    = ami[j] * amn[nn] / ( ami[j] + amn[nn] )
               amimn   = amn[nn] / ( ami[j] + amn[nn] )
               nufacin = 2.59e-9 / numpy.sqrt(amuf) * amimn * numpy.sqrt(alpha0[nn])
               nuin[:,j,nn,:] = nufacin * denn[:,nn,:]
            #meta stable oxygen ions
            nuin[:,ptop2d-1,nn,:]=nuin[:,ptop-1,nn,:]
            nuin[:,ptop2p-1,nn,:]=nuin[:,ptop-1,nn,:]
    
        #nitrogen 2(N2)
        
        j = ptn2p-1
        for nn in range(7):    
            if ( nn == ptn2-1 ):
               teff    = 0.5 * ( ti[j,:,:] + tn[:,:] )
               fac     = ( 1.00 - .069 * numpy.log10(teff) ) ** 2
               tfactor = numpy.sqrt(teff) * fac
               nuin[:,j,nn,:] = 5.14e-11 * denn[:,nn,:] * tfactor
            else:
               amuf    = ami[j] * amn[nn] / ( ami[j] + amn[nn] )
               amimn   = amn[nn] / ( ami[j] + amn[nn] )
               nufacin = 2.59e-9 / numpy.sqrt(amuf) * amimn * numpy.sqrt(alpha0[nn])
               nuin[:,j,nn,:] = nufacin * denn[:,nn,:]

        
        #nitric oxide (N0)
        j = ptnop-1
        for nn in range(7):
            amuf    = ami[j] * amn[nn] / ( ami[j] + amn[nn] )
            amimn   = amn[nn] / ( ami[j] + amn[nn] )
            nufacin = 2.59e-9 / numpy.sqrt(amuf) * amimn * numpy.sqrt(alpha0[nn])
            nuin[:,j,nn,:] = nufacin * denn[:,nn,:]
    
        #oxygen 2(O2)
        j = pto2p-1
        for nn in range(7):    
            if ( nn == pto2-1 ):
               teff    = 0.5 * ( ti[j,:,:] + tn[:,:] )
               fac     = ( 1.00 - .073 * numpy.log10(teff) ) ** 2
               tfactor = numpy.sqrt(teff) * fac
               nuin[:,j,nn,:] = 2.59e-11 * denn[:,nn,:] * tfactor
            else:
               amuf    = ami[j] * amn[nn] / ( ami[j] + amn[nn] )
               amimn   = amn[nn] / ( ami[j] + amn[nn] )
               nufacin = 2.59e-9 / numpy.sqrt(amuf) * amimn * numpy.sqrt(alpha0[nn])
               nuin[:,j,nn,:] = nufacin * denn[:,nn,:]

    
        #energized oxygen
        j=ptenop-1
        for nn in range(7):
            nuin[:,j,nn,:]=0.0
 
        return nuin #s^-1

    def unroll(self,xin):
        sh=xin.shape
        if (sh[-1]==129):
            rf=1
        elif (sh[-1]==513):
            rf=2
        elif (sh[-1]==2049):
            rf=4
        else:
            print("I don't understand this resolution")
            return None
        
        nphi=32*rf
        nr=7*rf+1
        xout=numpy.zeros((sh[0],nr,nphi+1))
        
        #pole
        for i in range(nphi+1):
            xout[:,0,i]=xin[:,0]

        #first region
        for i in range(2*rf):
            xout[:,i+1,:-4:4]=xin[:,i*nphi//4+1:(i+1)*nphi//4+1]
            xout[:,i+1,1:-3:4]=xin[:,i*nphi//4+1:(i+1)*nphi//4+1]
            xout[:,i+1,2:-2:4]=xin[:,i*nphi//4+1:(i+1)*nphi//4+1]
            xout[:,i+1,3:-1:4]=xin[:,i*nphi//4+1:(i+1)*nphi//4+1]
            
            xout[:,i+1,-1]=xout[:,i+1,0]

        #second region
        offset=2*rf*nphi//4+1
        for i in range(3*rf):
            xout[:,i+1+2*rf,:-2:2]=xin[:,i*nphi//2+offset:(i+1)*nphi//2+offset]
            xout[:,i+1+2*rf,1:-1:2]=xin[:,i*nphi//2+offset:(i+1)*nphi//2+offset]

            xout[:,i+1+2*rf,-1]=xout[:,i+1+2*rf,0]

        #third region
        offset=1+2*rf*nphi//4+3*rf*nphi//2
        for i in range(2*rf):
            xout[:,i+1+5*rf,:-1]=xin[:,i*nphi+offset:(i+1)*nphi+offset]

            xout[:,i+1+5*rf,-1]=xout[:,i+1+5*rf,0]

        return xout
        

    def unroll_phi(self,xin):
        sh=xin.shape
        if (sh[-1]==129):
            rf=1
        elif (sh[-1]==513):
            rf=2
        elif (sh[-1]==2049):
            rf=4
        else:
            print("I don't understand this resolution")
            return None
        
        nphi=32*rf
        dphi=2*numpy.pi/nphi
        nr=7*rf+1
        xout=numpy.zeros((sh[0],nr,nphi+1))
        
        #pole
        for i in range(nphi+1):
            xout[:,0,i]=(i+0.5)*dphi

        #first region
        for i in range(2*rf):
            xout[:,i+1,:-3:4]=xin[:,i*nphi//4+1:(i+1)*nphi//4+1]-1.5*dphi
            xout[:,i+1,1:-2:4]=xin[:,i*nphi//4+1:(i+1)*nphi//4+1]-0.5*dphi
            xout[:,i+1,2:-1:4]=xin[:,i*nphi//4+1:(i+1)*nphi//4+1]+0.5*dphi
            xout[:,i+1,3::4]=xin[:,i*nphi//4+1:(i+1)*nphi//4+1]+1.5*dphi

            xout[:,i+1,-1]=xout[:,i+1,0]+2*numpy.pi

        #second region
        offset=2*rf*nphi//4+1
        for i in range(3*rf):
            xout[:,i+1+2*rf,:-1:2]=xin[:,i*nphi//2+offset:(i+1)*nphi//2+offset]-0.5*dphi
            xout[:,i+1+2*rf,1::2]=xin[:,i*nphi//2+offset:(i+1)*nphi//2+offset]+0.5*dphi

            xout[:,i+1+2*rf,-1]=xout[:,i+1+2*rf,0]+2*numpy.pi

        #third region
        offset=1+2*rf*nphi//4+3*rf*nphi//2
        for i in range(2*rf):
            xout[:,i+1+5*rf,:-1]=xin[:,i*nphi+offset:(i+1)*nphi+offset]

            xout[:,i+1+5*rf,-1]=xout[:,i+1+5*rf,0]+2*numpy.pi

        return xout  
        


