import numpy as n

def LFMCurrent(grid,bx,by,bz,rion=6.5e8):
    """
    Computes current density from magnetic field
    grid - must be the Hexahedral grid object containing the cell centers of the 
           LFM grid in Rion units
    bx   - Bx magnetic field at cell centers in G
    by   - By magnetic field at cell centers in G
    bz   - Bz magnetic field at cell centers in G
    
    rion - ion radius defaults to 6.5e8 cm set equal to 1 if you pass raw grid
    
    Returns
    
    jx   - X current density in A/m^2
    jy   - Y current density in A/m^2
    jz   - Z current density in A/m^2
    """
    
    #Will need the location of the cell centers and volume for this calculation
    #See Merkin jpara.tex file in LTR repository for details of this calculation
    
    (xccI,yccI,zccI,xccJ,yccJ,zccJ,xccK,yccK,zccK)=grid.faceCenters()
    volume = grid.cellVolume()
    
    #calcualte the line integrals of b between cell centers
    binti=0.5*((bx[:-1,:,:]+bx[1:,:,:])*(grid.x[1:,:,:]-grid.x[:-1,:,:])+
           (by[:-1,:,:]+by[1:,:,:])*(grid.y[1:,:,:]-grid.y[:-1,:,:])+
           (bz[:-1,:,:]+bz[1:,:,:])*(grid.z[1:,:,:]-grid.z[:-1,:,:]))
           
    bintj=0.5*((bx[:,:-1,:]+bx[:,1:,:])*(grid.x[:,1:,:]-grid.x[:,:-1,:])+
           (by[:,:-1,:]+by[:,1:,:])*(grid.y[:,1:,:]-grid.y[:,:-1,:])+
           (bz[:,:-1,:]+bz[:,1:,:])*(grid.z[:,1:,:]-grid.z[:,:-1,:]))
           
    bintk=0.5*((bx[:,:,:-1]+bx[:,:,1:])*(grid.x[:,:,1:]-grid.x[:,:,:-1])+
           (by[:,:,:-1]+by[:,:,1:])*(grid.y[:,:,1:]-grid.y[:,:,:-1])+
           (bz[:,:,:-1]+bz[:,:,1:])*(grid.z[:,:,1:]-grid.z[:,:,:-1]))
           
    #compute the area integrals of current density
    jk = binti[:,:-1,:]+bintj[1:,:,:]-binti[:,1:,:]-bintj[:-1,:,:]
    ji = bintj[:,:,:-1]+bintk[:,1:,:]-bintj[:,:,1:]-bintk[:,:-1,:]
    jj = bintk[:-1,:,:]+binti[:,:,1:]-bintk[1:,:,:]-binti[:,:,:-1]

    #Do the loops in Eq 8 of Slava's jpara documentation
    jx=xccI[1:,:,:]*ji[1:,:,:]-xccI[:-1,:,:]*ji[:-1,:,:]
    jy=yccI[1:,:,:]*ji[1:,:,:]-yccI[:-1,:,:]*ji[:-1,:,:]
    jz=zccI[1:,:,:]*ji[1:,:,:]-zccI[:-1,:,:]*ji[:-1,:,:]

    jx=xccJ[:,1:,:]*jj[:,1:,:]-xccJ[:,:-1,:]*jj[:,:-1,:]+jx
    jy=yccJ[:,1:,:]*jj[:,1:,:]-yccJ[:,:-1,:]*jj[:,:-1,:]+jy
    jz=zccJ[:,1:,:]*jj[:,1:,:]-zccJ[:,:-1,:]*jj[:,:-1,:]+jz

    jx=xccK[:,:,1:]*jk[:,:,1:]-xccK[:,:,:-1]*jk[:,:,:-1]+jx
    jy=yccK[:,:,1:]*jk[:,:,1:]-yccK[:,:,:-1]*jk[:,:,:-1]+jy
    jz=zccK[:,:,1:]*jk[:,:,1:]-zccK[:,:,:-1]*jk[:,:,:-1]+jz
    
    #turn it into current density
    jx = jx/volume
    jy = jy/volume
    jz = jz/volume
  
    #   The current is in [Gs/ionosphere radius] inits.
    #   Convert to MKS:
    #      [Gs]       1.e-4 [T]
    #   ---------- = ----------------
    #   [6.5e8 cm]    6.5e8 1.e-2 [m]
    #   and divide by mu_0 = 4PI/1.e7 [H/m] to get the MKS current density:
    #  Conversion factor = 10^5/4pi/6.5e8        
    
    jx = jx*1.0e5/(4.0*n.pi)/rion
    jy = jy*1.0e5/(4.0*n.pi)/rion
    jz = jz*1.0e5/(4.0*n.pi)/rion
    
    return (jx,jy,jz)


