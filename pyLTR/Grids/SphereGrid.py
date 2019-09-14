import numpy as n
import sys
from pyLTR.Grids import HexahedralGrid as HG

class SphereGrid(object):
    def __init__(self,dims):
        """
        Compute a Spherical Grid
        Inputs
           dims - 3d array of dimension sizes
        """   
        (self.ni,self.nj,self.nk)=dims
        
        
        #Phi,Theta,R of cell corners
        self.__isPTRCornerComputed = False
        self.__p = None
        self.__t = None
        self.__r = None
        
        #Phi, Theta, R of cell centers
        self.__isPTRCenterComputed = False
        self.__pc = None
        self.__tc = None
        self.__rc = None
              
        #Use the HexGrid for x,y,z stuff along with other parameters
        self.__isHexGridComputed = False
        self.__x = None
        self.__y = None
        self.__z = None
        self.hgSphere=None
    
    def __set_r_theta_phi(self,rmin,rmax,thetamin):
        """
        Compute p,t,r arrays for cell corners given domain
        Inputs
          rmin - minium radius
          rmax - max radius
          thetamin - minium theata
        """
        phi   = n.linspace(0,2.*n.pi,self.nk+1)
        theta = n.linspace(0+thetamin,n.pi-thetamin,self.nj+1)
        r     = n.linspace(rmin,rmax,self.ni+1)
        
        return(r,theta,phi)    
    
    def ptrCorner(self,rmin = None, rmax = None, thetamin = None,
                r = None, theta = None, phi = None):
        """
        Computes the Phi,Theta,R values of cell corners
        Inputs
          rmin - Min radius defaults to None
          rmax - Max Radius defaults to None
          thetamin - Min Theta defaults to None
          r - array of radii - defaults to None
          theta - array of theta defaults to None
          phi - array of phi defaults to None
        
        NB - Must specify either min/max or r,theta,phi for code to work
        """
        if rmin is None and rmax is None and r is None:
            sys.exit('Must specfiy either r min/max or r')
        if thetamin is None and theta is None:
            sys.exit('Must specify either thetamin or theta')
            
        if self.__isPTRCornerComputed:
            return (self.__p,self.__t,self.__r)
            
        # if (rmin,rmax,thetamin) specified, rewrite (r,theta,phi) even if defined
        if rmin is not None and rmax is not None and thetamin is not None:
            r,theta,phi = self.__set_r_theta_phi(rmin,rmax,thetamin)
        self.__p,self.__t,self.__r = n.meshgrid(phi,theta,r,indexing='ij')    
        self.__isPTRCornerComputed = True
        
        return (self.__p,self.__t,self.__r)
        
    def ptrCenter(self):
        if not self.__isPTRCornerComputed:
            sys.exit('Must call ptrCorner before ptrCenter')
            
        if self.__isPTRCenterComputed:
            return (self.__pc,self.__tc,self.__rc)
            
        self.__pc = 0.125*(self.__p[:-1,:-1,:-1]+self.__p[1:,:-1,:-1]+
                           self.__p[:-1,1:,:-1]+self.__p[1:,1:,:-1]+
                           self.__p[:-1,:-1,1:]+self.__p[1:,:-1,1:]+
                           self.__p[:-1,1:,1:]+self.__p[1:,1:,1:])
        self.__tc = 0.125*(self.__t[:-1,:-1,:-1]+self.__t[1:,:-1,:-1]+
                           self.__t[:-1,1:,:-1]+self.__t[1:,1:,:-1]+
                           self.__t[:-1,:-1,1:]+self.__t[1:,:-1,1:]+
                           self.__t[:-1,1:,1:]+self.__t[1:,1:,1:])
        self.__rc = 0.125*(self.__r[:-1,:-1,:-1]+self.__r[1:,:-1,:-1]+
                           self.__r[:-1,1:,:-1]+self.__r[1:,1:,:-1]+
                           self.__r[:-1,:-1,1:]+self.__r[1:,:-1,1:]+
                           self.__r[:-1,1:,1:]+self.__r[1:,1:,1:])
                           
        return (self.__pc,self.__tc,self.__rc)
                           
    def xyzCorner(self):
        """
        Returns x,y,z values of cell corners
        """
        
        if not self.__isPTRCornerComputed:
            sys.exit('Must call ptrCorner before ptrCenter')
            
        if self.__isHexGridComputed:
            return (self.__x,self.__y,self.__z)
            
        self.__x = self.__r*n.sin(self.__t)*n.cos(self.__p)
        self.__y = self.__r*n.sin(self.__t)*n.sin(self.__p)
        self.__z = self.__r*n.cos(self.__t)
        
        #Go ahead a compute the HexGrid
        self.hgSphere=HG(self.__x,self.__y,self.__z)
        
        return (self.__x,self.__y,self.__z)
        
    def xyzCenter(self):
        """
        Returns the x,y,z of cell centers
        """
        
        if self.__isHexGridComputed:
            return self.hgSphere.cellCenters()
            
        self.xyzCorner()
        return self.hgSphere.cellCenters()
                
if __name__ == '__main__':
    sg = SphereGrid((53,48,64))
    (p,t,r) = sg.ptrCorner(rmin=21.5,rmax=215,thetamin=1.0e-6)
    (pc,tc,rc) = sg.ptrCenter()
    (x,y,z) = sg.xyzCorner()
    (xc,yc,zc) = sg.xyzCenter()

    
    
        
