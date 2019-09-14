"""
Computes Vector Potential for x,y,z location 
"""
import numpy as n

class VectorPotential(object):
                      
    def __init__(self,sinner=0.96,souter=1.05,x_max=22.0,x_min=-185.0,
               r_max=80.0,bx_zero=0.0,rearth=6.38e8,geoqmu=0.3):
        
        self.__sinner = sinner
        self.__souter = souter
        self.__x_max = x_max*rearth
        self.__x_min = x_min*rearth
        self.__r_max = r_max*rearth
        self.__xq_off = (self.__x_max+self.__x_min)/2.0
        self.__aq = 2.0/(self.__x_max-self.__x_min)
        self.__bq = 1.0/self.__r_max
        self.__bxq = bx_zero
        self.__cos_tilt = 1.0
        self.__sin_tilt = 0.01
        self.__geoqmu = geoqmu*rearth**3
        
        return
    
                                    
    def ax(self,x,y,z):
        r = n.sqrt(x*x+y*y+z*z)
        s = n.sqrt(((x-self.__xq_off)*self.__aq)**2 + (y*y+z*z)*self.__bq*self.__bq)
        if ( s < self.__sinner):
            return (self.__geoqmu/r**2)*y/r
        elif (s < 1.0):
            return (self.__geoqmu/r**2)*(y/r*(1.0-s)/(1.0-self.__sinner))
        else:
            return 0.0

            
    def ay(self,x,y,z):
        r = n.sqrt(x*x+y*y+z*z)
        s = n.sqrt(((x-self.__xq_off)*self.__aq)**2 + (y*y+z*z)*self.__bq*self.__bq)
        if ( s < self.__sinner):
            return -1.0*(self.__geoqmu/r**2)*x/r
        elif (s < 1.0):
            return -1.0*(self.__geoqmu/r**2)*(x/r*(1.0-s)/(1.0-self.__sinner))
        elif (s < self.__souter):
            return (-1.0*self.__bxq*(z*self.__cos_tilt-x*self.__sin_tilt)*
                    (s-1.0)/(self.__souter-1.0))
        else:
            return -1.0*self.__bxq*(z*self.__cos_tilt-x*self.__sin_tilt)
 
            
    def az(self,x,y,z):
        return 0.0
 
    def axArray(self,x,y,z):
        r = n.sqrt(x*x+y*y+z*z)
        s = n.sqrt(((x-self.__xq_off)*self.__aq)**2 + (y*y+z*z)*self.__bq*self.__bq)
        ax = x*0.0
        locs = s < 1.0
        ax[locs] = (self.__geoqmu/r[locs]**2)*(y[locs]/r[locs]*(1.0-s[locs])/
                   (1.0-self.__sinner))
        locs = s < self.__sinner
        ax[locs] = (self.__geoqmu/r[locs]**2)*y[locs]/r[locs]
        return ax
            
    def ayArray(self,x,y,z):
        r = n.sqrt(x*x+y*y+z*z)
        s = n.sqrt(((x-self.__xq_off)*self.__aq)**2 + (y*y+z*z)*self.__bq*self.__bq)
        ay = -1.0*self.__bxq*(z*self.__cos_tilt-x*self.__sin_tilt)
        locs = s < self.__souter
        ay[locs] = (-1.0*self.__bxq*(z[locs]*self.__cos_tilt-x[locs]*self.__sin_tilt)*
                    (s[locs]-1.0)/(self.__souter-1.0))
        locs = s < 1.0
        ay[locs] = -1.0*(self.__geoqmu/r[locs]**2)*(x[locs]/r[locs]*(1.0-s[locs])
                  /(1.0-self.__sinner))
        locs = s < self.__sinner
        ay[locs] = -1.0*(self.__geoqmu/r[locs]**2)*x[locs]/r[locs]
        return ay
            
    def azArray(self,x,y,z):
        return x*0.0
        
    def integrateVectorPotential(self,Ai,Aj,Ak,x,y,z):
        """
        Computes the Face Values of the Vector Potential
        """
        (nkp1,njp1,nip1)=x.shape
        ni = nip1 - 1
        nj = njp1 - 1
        nk = nkp1 - 1
        for k in range(nkp1):
            for j in range(njp1):
                for i in range(ni):
                    xijk = x[k,j,i]
                    xip1 = x[k,j,i+1]
                    yijk = y[k,j,i]
                    yip1 = y[k,j,i+1]
                    zijk = z[k,j,i]
                    zip1 = z[k,j,i+1]
                    Ai[k,j,i] = ((xip1-xijk)*self.gaussLineInt(self.ax,xijk,yijk,zijk,
                                                        xip1,yip1,zip1) +
                                (yip1-yijk)*self.gaussLineInt(self.ay,xijk,yijk,zijk,
                                                    xip1,yip1,zip1) +                                                 
                                (zip1-zijk)*self.gaussLineInt(self.az,xijk,yijk,zijk,
                                                    xip1,yip1,zip1))
    
                                                    
        for k in range(nkp1):
            for j in range(nj):
                for i in range(nip1):
                    xijk = x[k,j,i]
                    xjp1 = x[k,j+1,i]
                    yijk = y[k,j,i]
                    yjp1 = y[k,j+1,i]
                    zijk = z[k,j,i]
                    zjp1 = z[k,j+1,i]
                    Aj[k,j,i] = ((xjp1-xijk)*self.gaussLineInt(self.ax,xijk,yijk,zijk,
                                                    xjp1,yjp1,zjp1) +
                                (yjp1-yijk)*self.gaussLineInt(self.ay,xijk,yijk,zijk,
                                                    xjp1,yjp1,zjp1) +                                                 
                                (zjp1-zijk)*self.gaussLineInt(self.az,xijk,yijk,zijk,
                                                    xjp1,yjp1,zjp1))
        for k in range(nk):
            for j in range(njp1):
                for i in range(nip1):
                    xijk = x[k,j,i]
                    xkp1 = x[k+1,j,i]
                    yijk = y[k,j,i]
                    ykp1 = y[k+1,j,i]
                    zijk = z[k,j,i]
                    zkp1 = z[k+1,j,i]
                    Ak[k,j,i] = ((xkp1-xijk)*self.gaussLineInt(self.ax,xijk,yijk,zijk,
                                                    xkp1,ykp1,zkp1) +
                                (ykp1-yijk)*self.gaussLineInt(self.ay,xijk,yijk,zijk,
                                                    xkp1,ykp1,zkp1) +                                                 
                                (zkp1-zijk)*self.gaussLineInt(self.az,xijk,yijk,zijk,
                                                    xkp1,ykp1,zkp1))
        return
    
    def integrateVectorPotentialArray(self,Ai,Aj,Ak,x,y,z):
        """
        Computes the Face Values of the Vector Potential
        Uses Python Vector Operations for speed
        """
        (nkp1,njp1,nip1)=x.shape
        ni = nip1 - 1
        nj = njp1 - 1
        nk = nkp1 - 1
        for i in range(ni):
            xijk = x[:,:,i]
            xip1 = x[:,:,i+1]
            yijk = y[:,:,i]
            yip1 = y[:,:,i+1]
            zijk = z[:,:,i]
            zip1 = z[:,:,i+1]
            Ai[:,:,i] = ((xip1-xijk)*self.gaussLineInt(self.axArray,xijk,yijk,zijk,
                                                    xip1,yip1,zip1) +
                            (yip1-yijk)*self.gaussLineInt(self.ayArray,xijk,yijk,zijk,
                                                xip1,yip1,zip1) +                                                 
                            (zip1-zijk)*self.gaussLineInt(self.azArray,xijk,yijk,zijk,
                                                xip1,yip1,zip1))
    
        for j in range(nj):
            xijk = x[:,j,:]
            xjp1 = x[:,j+1,:]
            yijk = y[:,j,:]
            yjp1 = y[:,j+1,:]
            zijk = z[:,j,:]
            zjp1 = z[:,j+1,:]
            
            Aj[:,j,:] = ((xjp1-xijk)*self.gaussLineInt(self.axArray,xijk,yijk,zijk,
                                                xjp1,yjp1,zjp1) +
                            (yjp1-yijk)*self.gaussLineInt(self.ayArray,xijk,yijk,zijk,
                                                xjp1,yjp1,zjp1) +                                                 
                            (zjp1-zijk)*self.gaussLineInt(self.azArray,xijk,yijk,zijk,
                                                xjp1,yjp1,zjp1))
        
        for k in range(nk):
            xijk = x[k,:,:]
            xkp1 = x[k+1,:,:]
            yijk = y[k,:,:]
            ykp1 = y[k+1,:,:]
            zijk = z[k,:,:]
            zkp1 = z[k+1,:,:]
            Ak[k,:,:] = ((xkp1-xijk)*self.gaussLineInt(self.axArray,xijk,yijk,zijk,
                                                xkp1,ykp1,zkp1) +
                            (ykp1-yijk)*self.gaussLineInt(self.ayArray,xijk,yijk,zijk,
                                                xkp1,ykp1,zkp1) +                                                 
                            (zkp1-zijk)*self.gaussLineInt(self.azArray,xijk,yijk,zijk,
                                                xkp1,ykp1,zkp1))
        return
        
    def gaussLineInt(self,func,xa,ya,za,xb,yb,zb):
        """
        Integrate FX(x,y,z) over the line (xa,ya,za) to (xb,yb,zb).  This
        subroutine does Gaussian integration with the first twelve Legendre
        polynomials as the basis fuctions.  Abromowitz and Stegun page 916.
        """
        
        # Positive zeros of 12th order Legendre polynomial
        a = [0.1252334085,  0.3678314989,  0.5873179542,
            0.7699026741,  0.9041172563,  0.9815606342]
        # Gaussian Integration coefficients for a 12th order polynomial
        wt = [0.2491470458,  0.2334925365,  0.2031674267,
            0.1600783285,  0.1069393259,  0.0471753363]
            
        dx = (xb-xa)/2.0
        dy = (yb-ya)/2.0
        dz = (zb-za)/2.0
        xbar = (xb+xa)/2.0
        ybar = (yb+ya)/2.0
        zbar = (zb+za)/2.0
        
        sum = 0.0
        for i in range(len(a)):
            sum = sum + wt[i] * (
                                func(xbar+a[i]*dx,ybar+a[i]*dy,zbar+a[i]*dz)+
                                func(xbar-a[i]*dx,ybar-a[i]*dy,zbar-a[i]*dz)
                                )
    
        return 0.5*sum    
    
    def setVectorPotentialBoundary(self,Ai,Aj,Ak):
        """
        Sets the Vector Potential Values along the Boundary
        """
        (nkp1,njp1,nip1)=Ai.shape
        ni = nip1 - 1
        nj = njp1 - 1
        nk = nkp1 - 1
        ksum1 = n.sum(Ai[:-1,0,:-1],axis=0)/float(nk)
        ksum2 = n.sum(Ai[:-1,nj,:-1],axis=0)/float(nk)
        Ai[:-1,0,:-1] = ksum1
        Ai[:-1,0,:-1] = ksum2
        Ak[:-1,0,:] = 0.0
        Ak[:-1,0,:] = 0.0
    
        return
    
    def computeMagneticFlux(self,Ai,Aj,Ak,Bi,Bj,Bk):
            """
            Compute Magnetic Flux through faces
            """
            
            Bi[:-1,:-1,:] = Aj[:-1,:-1,:] - Aj[1:,:-1,:] + Ak[:-1,1:,:] - Ak[:-1,:-1,:]
            Bj[:-1,:,:-1] = -1.0*(Ai[:-1,:,:-1] - Ai[1:,:,:-1] + 
                                Ak[:-1,:,1:] - Ak[:-1,:,:-1])
            Bk[:,:-1,:-1] = Ai[:,:-1,:-1] - Ai[:,1:,:-1] + Aj[:,:-1,1:] - Aj[:,:-1,:-1]
                        
            return
    