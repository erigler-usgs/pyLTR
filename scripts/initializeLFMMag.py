import pyLTR
import numpy as n
import configparser
import os
from pyhdf.SD import SD, SDC


def integrateVectorPotential(Ai,Aj,Ak,vp,x,y,z):
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
                Ai[k,j,i] = ((xip1-xijk)*gaussLineInt(vp.ax,xijk,yijk,zijk,
                                                      xip1,yip1,zip1) +
                             (yip1-yijk)*gaussLineInt(vp.ay,xijk,yijk,zijk,
                                                 xip1,yip1,zip1) +                                                 
                             (zip1-zijk)*gaussLineInt(vp.az,xijk,yijk,zijk,
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
                Aj[k,j,i] = ((xjp1-xijk)*gaussLineInt(vp.ax,xijk,yijk,zijk,
                                                 xjp1,yjp1,zjp1) +
                             (yjp1-yijk)*gaussLineInt(vp.ay,xijk,yijk,zijk,
                                                 xjp1,yjp1,zjp1) +                                                 
                             (zjp1-zijk)*gaussLineInt(vp.az,xijk,yijk,zijk,
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
                Ak[k,j,i] = ((xkp1-xijk)*gaussLineInt(vp.ax,xijk,yijk,zijk,
                                                 xkp1,ykp1,zkp1) +
                             (ykp1-yijk)*gaussLineInt(vp.ay,xijk,yijk,zijk,
                                                 xkp1,ykp1,zkp1) +                                                 
                             (zkp1-zijk)*gaussLineInt(vp.az,xijk,yijk,zijk,
                                                 xkp1,ykp1,zkp1))
    return
    
def gaussLineInt(func,xa,ya,za,xb,yb,zb):
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

def setVectorPotentialBoundary(Ai,Aj,Ak):
    """
    Sets the Vector Potential Values along the Boundary
    """
    (nkp1,njp1,nip1)=Ai.shape
    ni = nip1 - 1
    nj = njp1 - 1
    nk = nkp1 - 1
    ksum1 = sum(Ai[:-1,0,:-1],axis=0)/float(nk)
    ksum2 = sum(Ai[:-1,nj,:-1],axis=0)/float(nk)
    Ai[:-1,0,:-1] = ksum1
    Ai[:-1,0,:-1] = ksum2
    Ak[:-1,0,:] = 0.0
    Ak[:-1,0,:] = 0.0

    return

def computeMagneticFlux(Ai,Aj,Ak,Bi,Bj,Bk):
        """
        Compute Magnetic Flux through faces
        """
        
        Bi[:-1,:-1,:] = Aj[:-1,:-1,:] - Aj[1:,:-1,:] + Ak[:-1,1:,:] - Ak[:-1,:-1,:]
        Bj[:-1,:,:-1] = -1.0*(Ai[:-1,:,:-1] - Ai[1:,:,:-1] + 
                              Ak[:-1,:,1:] - Ak[:-1,:,:-1])
        Bk[:,:-1,:-1] = Ai[:,:-1,:-1] - Ai[:,1:,:-1] + Aj[:,:-1,1:] - Aj[:,:-1,:-1]
                    
        return
        
                
#import grid data from GRID HDF Files
gridpath = '/Users/wiltbemj/src/LTR-para/LFM-para/src/startup/grids'
gridfile = 'GRIDOUT-53x48x64.hdf'
gridhdf = SD(os.path.join(gridpath,gridfile),mode=SDC.READ)
sds=gridhdf.select('X_grid')
x=sds.get()
sds=gridhdf.select('Y_grid')
y=sds.get()
sds=gridhdf.select('Z_grid')
z=sds.get()
(nkp1,njp1,nip1)=x.shape
ni = nip1 - 1
nj = njp1 - 1
nk = nkp1 - 1

path = '/Users/wiltbemj/mhd_data'
fileName = 'LFMTest.hdf'
lfmh = pyLTR.Tools.lfmstartup.lfmstartup(os.path.join(path,fileName),(ni,nj,nk))
lfmh.open(tzero=0.0)
Re = 6.38e8
xcm = x*Re
ycm = y*Re
zcm = z*Re
lfmh.writeVar('X_grid',xcm)
lfmh.writeVar('Y_grid',ycm)
lfmh.writeVar('Z_grid',zcm)
lfmh.writeVar('rho_',2.0e-25*n.ones((nkp1,njp1,nip1)))
lfmh.writeVar('c_',  1.0e5*n.ones((nkp1,njp1,nip1)))
lfmh.writeVar('vx_',n.zeros((nkp1,njp1,nip1)))
lfmh.writeVar('vy_',n.zeros((nkp1,njp1,nip1)))
lfmh.writeVar('vz_',n.zeros((nkp1,njp1,nip1)))
(bx,by,bz)=pyLTR.Physics.Dipole.Dipole(x,y,z)
lfmh.writeVar('bx_',bx)
lfmh.writeVar('by_',by)
lfmh.writeVar('bz_',bz)
Ai = n.zeros((nkp1,njp1,nip1),dtype='float32')
Aj = n.zeros((nkp1,njp1,nip1),dtype='float32')
Ak = n.zeros((nkp1,njp1,nip1),dtype='float32')
vp=pyLTR.Physics.VectorPotential.VectorPotential()
integrateVectorPotential(Ai,Aj,Ak,vp,xcm,ycm,zcm)
setVectorPotentialBoundary(Ai,Aj,Ak)
Bi = n.zeros((nkp1,njp1,nip1))
Bj = n.zeros((nkp1,njp1,nip1))
Bk = n.zeros((nkp1,njp1,nip1))
computeMagneticFlux(Ai,Aj,Ak,Bi,Bj,Bk)
lfmh.writeVar('bi_',Bi)
lfmh.writeVar('bj_',Bj)
lfmh.writeVar('bk_',Bk)

div_abs = abs(Bi[:-1,:-1,1:]-Bi[:-1,:-1,:-1]+Bj[:-1,1:,:-1]-Bj[:-1,:-1,:-1]+Bk[1:,:-1,:-1]-Bk[:-1,:-1,:-1])

lfmh.close()