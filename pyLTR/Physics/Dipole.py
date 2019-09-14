"""
   Computes Dipole Magnetic field for points on X,Y,Z grid
"""

def Dipole(x,y,z,geoqmu=0.3):
  """
  Computes the Cartesian Compoents of the dipole magnetic field
  Requires
    X - Grid of X locations in [Re]
    Y - Grid of Y locations in [Re]
    Z - Grid of Z locations in [Re]
  Optional
    Geoqmu - default value of 0.3 is used
  """
  from numpy import sqrt
    
  #xc=0.125*( x[:-1,:-1,:-1]+x[1:,:-1,:-1]+x[:-1,1:,:-1]+x[:-1,:-1,1:]+
  #                x[1:,1:,:-1]+x[1:,:-1,1:]+x[:-1,1:,1:]+x[1:,1:,1:] )
  #yc=0.125*( y[:-1,:-1,:-1]+y[1:,:-1,:-1]+y[:-1,1:,:-1]+y[:-1,:-1,1:]+
  #                y[1:,1:,:-1]+y[1:,:-1,1:]+y[:-1,1:,1:]+y[1:,1:,1:] )
  #zc=0.125*( z[:-1,:-1,:-1]+z[1:,:-1,:-1]+z[:-1,1:,:-1]+z[:-1,:-1,1:]+
  #                z[1:,1:,:-1]+z[1:,:-1,1:]+z[:-1,1:,1:]+z[1:,1:,1:] )
  xc=x
  yc=y
  zc=z
  Rsq=(xc[:,:,:]**2+yc[:,:,:]**2+zc[:,:,:]**2)
  R = sqrt(Rsq)
  
  bdip_x = -1.0*(geoqmu/Rsq)*(3*xc[:,:,:]*zc[:,:,:]/Rsq)/R
  bdip_y = -1.0*(geoqmu/Rsq)*(3*yc[:,:,:]*zc[:,:,:]/Rsq)/R
  bdip_z = -1.0*(geoqmu/Rsq)*(2.0 - 3.0*(xc[:,:,:]**2+yc[:,:,:]**2)/Rsq)/R

  return (bdip_x,bdip_y,bdip_z)
