"""
   Computes dervied quantities from basic MIX variables
   
"""
import pylab as p
import pyLTR

def checkUnits(dataDict,units):
    """
    Checks the units on the variable
    >>> checkUnits({'name': 'test', 'units': 'furlong/fortnight'},units='meters') 
    Failed with incorrect units for test
    Has furlong/fortnight wants meters
    False
    
    >>> checkUnits({'name': 'test', 'units': 'meters'},units='meters')
    True
    """
    
    if 'name' not in dataDict:
      print('Dictionary lacks name')
      return False
    if 'units' in dataDict:
       if dataDict['units'] != units:
          print("Failed with incorrect units for "+dataDict['name']) 
          print('Has '+dataDict['units']+' wants '+units)
          return False
    else:
       print(dataDict['units'])
       print("Must supply units for "+dataDict['name'])
       return False

    return True


def efield(x,y,psi,ri=6500.0e3,oh=False):
    """
    Computes:
      (phi,theta)   - spherical position vector components [rad]
                      (these really should be computed in a separate function,
                       and passed as an input instead of x,y)
      (ephi,etheta) - spherical electric field vector components [V/m]
    
    Requires:
      x     - grid of X locations (multiples of ri)
      y     - grid of Y location (multiples of ri)
      psi   - Potential
    
    Optional
      ri - Ionospheric radius (default=6500e3 m)
      oh - flag indicating {x,y} grids are for the "opposite hemisphere" of
           the current POV (i.e., if looking down on the north pole, {x,y}
           correspond to psi in the southern hemisphere, and vice-versa);
           this assures theta vector components are determined correctly.
           (default=False)
    """
    """
    NOTES/FIXMES:
      A) Optional input parameters (e.g., ri, oh) were added that would not
         be necessary if the {x,y} grids had just been a 3D {phi,theta,rho}
         from the beginning. HOWEVER, efield() was one of the first functions
         in MIXCalcs.py, so the optional parameters are required to maintain
         backward compatibility. The ri parameter is simple enough, but to 
         mitigate possible confusion over oh, here are expected use-cases:
         
         oh==False:
         1) {x,y} are unaltered MIX file coordinate grids, psi is from the 
            northern hemisphere, and the desired output is in SM coordinates;
         2) {x,y} are unaltered MIX file coordinate grids, psi is from the
            southern hemisphere, and the desired output is in SM rotated 180
            degrees about X axis;
            
         oh==True:
         1) {x,y} are rotated MIX file coordinate grids, psi is from the
            southern hemisphere, and the desired output is in SM coordiantes;
         2) {x,y} are rotated MIX file coordinate grids, psi is from the
            northern hemisphere, but the desired output is in SM rotated 180
            degrees about X axis;
         
         Note that an earlier vesion of efield() was designed to rotate {x,y}
         itself, but this was later deemed a poor design choice. Now it is
         incumbent on the user to make certain {x,y} have been rotated to
         match the desired/expected POV.
         
         
      B) Of course everything addressed in A) implies that efield() cannot calculate
         electric fields on a grid that spans the equator, an arbitrary restriction.
         Consider biting the bulle, and changing the interface for this function,
         fixing any calls to efield() within the pyLTR code-base, and hoping there
         are not too many users out there who have made use of this function yet.
         -EJR 8/2014
    """

    # convert x,y grid to phi,theta
    phi, theta = _xy2pt(x/ri, y/ri, oh)
    
    
    # Efield is just -1*grad(psi)/ri
    (ephi,etheta) = _grad2d_sph(psi,(phi,theta))
    ephi = -1.*ephi/ri
    etheta = -1.*etheta/ri
    
    
    # The following is kind of a mess, and is of questionable value... 
    # consider just setting pole values to zero, or NaN
    
    # at the poles (theta==0|pi), psi is equal for all phi, so the gradient in phi 
    # is zero; however, the arc length in phi is also zero (e.g., sin(0)), which
    # leads to ephi=0/0==NaN if nothing else is done; however, it can be shown
    # that if theta==0, phi can have *any* value, and it will not affect the
    # transformation, so we just set dphi=0. 
    # On the other hand, there may be unique theta components to the efield
    # at the pole, since there are many unique psi for theta!=0 (this is not
    # physical, but a consequence of the MIX solver at the pole); so, these are
    # transformed into Cartesian coordinates, averaged, then transformed back
    # to local spherical to provide a consistent efield vector across the pole
    ephi[p.isnan(ephi)] = 0.0
    
    zeroThetaMask = p.logical_or(theta == 0.0, theta == p.pi)
    if zeroThetaMask.any():
       cartTmp = pyLTR.transform.SPHtoCAR(phi[zeroThetaMask], 0.0, ri,
                                          0.0, etheta[zeroThetaMask], 0.0)
       
       cartAvgVec = (p.mean(cartTmp[0]), p.mean(cartTmp[1]), p.mean(cartTmp[2]) )
       
       # pyLTR's (i.e., Geopack's) CARtoSPH assumes phi==0 for all theta==0,
       # which is reasonable, but we prefer a local spherical direction vector
       # for each phi[zeroThetaMask]. 
       etheta[zeroThetaMask] = (p.cos(phi[zeroThetaMask]) * cartAvgVec[0] +
                                p.sin(phi[zeroThetaMask]) * cartAvgVec[1])
       ephi[zeroThetaMask] = (p.cos(phi[zeroThetaMask]) * cartAvgVec[1] -
                              p.sin(phi[zeroThetaMask]) * cartAvgVec[0])
    
    # return position and efield vector components
    return (phi,theta),(ephi,etheta)
    
    """
    ##
    ## The following calculates efield at mid-points, NOT the same gridpoints
    ## as the inputs x,y,psi...this uses 1-sided differencing, which smoothes
    ## less, but is prone to noisy artifacts -EJR 11/2013
    ##
    
    # compute deltas in phi and theta
    dtheta = theta[0,1:] - theta[0,:-1]
    dphi = phi[1:,1] - phi[:-1,1]

    # allocate output grids
    nphi = x.shape[0] - 1
    ntheta = x.shape[1] - 1
        
    phi_mid = p.zeros((nphi,ntheta))
    theta_mid = p.zeros((nphi,ntheta))
    
    ephi = p.zeros((nphi, ntheta))
    etheta = p.zeros((nphi, ntheta))
    
    # compute output phi,theta grids
    theta_mid = _midGrid(theta)
    phi_mid = _midGrid(phi)
    
    # compute etheta
    for j in range(nphi): # loop over phi
        etheta[j,:] = -1 * ( (psi[j,1:]-psi[j,:-1]) + (psi[j+1,1:]-psi[j+1,:-1]))/2./dtheta/ri
        
    # compute ephi
    for i in range(ntheta): # loop over theta
        ephi[:,i] = -1 * 1./p.sin(theta_mid[0,i])*( (psi[1:,i]-psi[:-1,i])+(psi[1:,i+1]-psi[:-1,i+1]) )/2./dphi/ri
    
    # not sure why outputs are ordered phi,theta instead of theta,phi, but at 
    # least a couple pyLTR functions expect this, so leave as-is for now -EJR 9/2013
    return (phi_mid,theta_mid),(ephi,etheta)
    """

def efieldDict(x,y,psi,ri=6500.0e3,oh=False):

    if not checkUnits(x,'m'):
       raise Exception('Unit check failed')
    if not checkUnits(y,'m'):
       raise Exception('Unit check failed')
    if not checkUnits(psi,'kV'):
       raise Exception('Unit check failed')
    
    # rescale inputs to more natural physical units before calling efield()
    angles,efields = efield(x['data'], y['data'], psi['data']*1.e3, ri, oh)
        
    # rescale outputs to more typical plotting units before returning
    phi_dict = {'data':angles[0],'name':r'$\phi$','units':r'rad'}
    theta_dict = {'data':angles[1],'name':r'$\theta$','units':r'rad'}
    ephi_dict = {'data':efields[0]*1.e3,'name':r'$E_\phi$','units':r'mV/m'}
    etheta_dict = {'data':efields[1]*1.e3,'name':r'$E_\theta$','units':r'mV/m'}
    
    return (phi_dict,theta_dict),(ephi_dict,etheta_dict)


def joule(efield,sigmap):
   """
   Computes:
     Qj     - joule heating
   
   Requires:
     efield - efield tuple (ephi,etheta) as generated by efield() function
     sigmap - Pedersen conductivities at efield locations
   """
   Qj = sigmap*(efield[0]**2+efield[1]**2)
      
   return Qj

def jouleDict(efield,sigmap):
 
    if not checkUnits(efield[0],'mV/m'):
       raise Exception('Unit check failed')
    if not checkUnits(efield[1],'mV/m'):
       raise Exception('Unit check failed')
    if not checkUnits(sigmap,'S'):
       raise Exception('Unit check failed')

    Qj = joule((efield[0]['data']*1.e-3, efield[1]['data']*1.e-3), sigmap['data'])
    Qj_dict = {'data':Qj*1.e3, 'name':'Joule Heating', 'units':'mW/m^2'}

    return Qj_dict


def jped(efield,sigmap):
   """
   Computes:
     Jp     - magnitude of Pedersen current
   
   Requires
     efield - efield tuple (ephi,etheta) as generated by efield() function
     sigmap - Pedersen conductivities at same gridpoints as efield components
   """
   """
   NOTE: this function is no different from jhall(), and its output could easily
         be obtained from jphitheta()...look into overloading
   """
   
   Jp = sigmap*p.sqrt(efield[0]**2+efield[1]**2)
   
   return Jp 

def jpedDict(efield,sigmap):
    """
    The output units here (i.e., micro-Amps/meter) are probably not
    appropriate for ionospheric current sheets, which are typically
    on the order of 1 Amp/meter...presumably micro-Amps were chosen
    because this is an appropriate unit for FACs, which are *volume*
    currents, with density distributed along another length dimension.
    """
    if not checkUnits(efield[0],'mV/m'):
       raise Exception('Unit check failed')
    if not checkUnits(efield[1],'mV/m'):
       raise Exception('Unit check failed')
    if not checkUnits(sigmap,'S'):
       raise Exception('Unit check failed')

    jpedval=jped((efield[0]['data']*1.e-3,efield[1]['data']*1.e-3),sigmap['data'])
    jpeddict = {'data':jpedval*1.e6,'name':'Pedersen Current','units':r'$\mu A/m$'}

    return jpeddict


def jhall(efield,sigmah):
   """
   Computes:
     Jh     - magnitude of Hall current
   
   Requires
     efield - efield tuple (ephi,etheta) as generated by efield() function
     sigmah - Hall conductivities at same gridpoints as efield components
   """
   """
   NTOE: this function is no different from jped(), and its output could easily
         be obtained from jphitheta()...look into overloading
   """
   
   Jh = sigmah*p.sqrt(efield[0]**2+efield[1]**2)
   
   return Jh

def jhallDict(efield,sigmah):
    """
    The output units here (i.e., micro-Amps/meter) are probably not
    appropriate for ionospheric current sheets, which are typically
    on the order of 1 Amp/meter...presumably micro-Amps were chosen
    because this is an appropriate unit for FACs, which are *volume*
    currents, with density distributed along another length dimension.
    """
    if not checkUnits(efield[0],'mV/m'):
       raise Exception('Unit check failed')
    if not checkUnits(efield[1],'mV/m'):
       raise Exception('Unit check failed')
    if not checkUnits(sigmah,'S'):
       raise Exception('Unit check failed')

    jhallval=jhall((efield[0]['data']*1.e-3,efield[1]['data']*1.e-3),sigmah['data'])
    jhalldict = {'data':jhallval*1.e6,'name':'Hall Current','units':r'$\mu A/m$'}

    return jhalldict


def jphitheta(efield,sigmap,sigmah,colatitude=0.0):
   """
   Computes:
     (jphi,jtheta)          - ionospheric curent density vectors in spherical 
                              coordinates
     (jpedphi,jpedtheta)    - Pedersen component of ionospheric current density
     (jhallphi,jhalltheta)  - Hall component of ionospheric current density
   
   Requires
     efield - efield tuple (ephi,etheta) as generated by efield() function
     sigmap - Pedersen conductivities at efield position coordinates
     sigmah - Hall conductivities at efield position coordinates
   
   Optional
     colatitude - dipole magnetic colatitude (i.e., north magnetic polar angle),
                  used to scale conductivities according to magnetic dip angle.
                  (default=0)
                  NOTE: a non-standard definition of dipole magnetic "dip angle"
                        is used in the MIX ionospheric potential solver. To be
                        fully consistent with MIX, we convert colatitude into 
                        this value, then employ MIX's conductivity tensor to
                        apply Ohm's law. For reference, this "dip angle" is:
                        
                        cos(dipAngle) = -2 * cos(colatitude) / sqrt(1+3*cos(colatitude)**2)
     
   """
   
   # cosine of "dip angle"
   cosDip = -2 * p.cos(colatitude) / p.sqrt(1+3*p.cos(colatitude)**2)
   
   # Ohms law in Merkin & Lyon (2010) tensor form
   jpedphi = sigmap * efield[0]
   jpedtheta = sigmap * efield[1] / cosDip**2
   
   jhallphi = sigmah * efield[1] / cosDip
   jhalltheta = -sigmah * efield[0] / cosDip
   
   
   """
   ###
   ### This does not fully account for dip-angle, only the magnetic hemisphere
   ###
   
   # start with simpler Pedersen current density vectors
   jpedphi = sigmap * efield[0]
   jpedtheta = sigmap * efield[1]
   
   
   # rotate efield -90(+90) degrees for northern(southern) hemisphere for Hall currents
   isnorth=True
   if not(isnorth):
      jhallphi = sigmah * efield[1]
      jhalltheta = sigmah * -efield[0]
   else:
      jhallphi = sigmah * -efield[1]
      jhalltheta = sigmah * efield[0]
   """
   
   
   # combine Pedersen and Hall vector components
   jphi = jpedphi + jhallphi
   jtheta = jpedtheta + jhalltheta
   
   
   return (jphi,jtheta),(jpedphi,jpedtheta),(jhallphi,jhalltheta)

def jphithetaDict(efield,sigmap,sigmah,colatitude=0.0):
    """
    The output units here (i.e., micro-Amps/meter) are probably not
    appropriate for ionospheric current sheets, which are typically
    on the order of 1 Amp/meter...presumably micro-Amps were chosen
    because this is an appropriate unit for FACs, which are *volume*
    currents, with density distributed along another length dimension.
    """
    if not checkUnits(efield[0],'mV/m'):
       raise Exception('Unit check failed')
    if not checkUnits(efield[1],'mV/m'):
       raise Exception('Unit check failed')
    if not checkUnits(sigmap,'S'):
       raise Exception('Unit check failed')
    if not checkUnits(sigmah,'S'):
       raise Exception('Unit check failed')

    ((jphi,jtheta),
     (jpedphi,jpedtheta),
     (jhallphi,jhalltheta)) = jphitheta((efield[0]['data']*1.e-3, efield[1]['data']*1.e-3),
                                         sigmap['data'], sigmah['data'], colatitude=colatitude)
    jphidict = {'data':jphi*1.e6,'name':'Phi Current','units':r'$\mu A/m$'}
    jthetadict = {'data':jtheta*1.e6,'name':'Theta Current','units':r'$\mu A/m$'}
    jpedphidict = {'data':jpedphi*1.e6,'name':'Pedersen Phi Current','units':r'$\mu A/m$'}
    jpedthetadict = {'data':jpedtheta*1.e6,'name':'Pedersen Theta Current','units':r'$\mu A/m$'}
    jhallphidict = {'data':jhallphi*1.e6,'name':'Hall Phi Current','units':r'$\mu A/m$'}
    jhallthetadict = {'data':jhalltheta*1.e6,'name':'Hall Theta Current','units':r'$\mu A/m$'}
    
    return ((jphidict,jthetadict),
            (jpedphidict,jpedthetadict),
            (jhallphidict,jhallthetadict))


"""
The following "support" routines should eventually be cleaned up, made more
robust/flexible, and documented more thoroughly, before making them "public"
"""


def _xy2pt(x,y,oh=False):
   """
   Convert x,y coordinates to phi,theta, assuming a spherical surface with
     radius equal to 1. 
   
   x,y are assumed to have been rotated to be consistent with the desired/
    expected POV. However, there still exists the issue of whether x,y are
    for the nearer hemisphere, or the "opposite hemisphere". If oh==True,
    adjust theta accordingly.
   
   In actuality, floating point round-off error leads to a grid of phi,theta
     that is not regular (i.e., it is not a perfect meshgrid). This error does
     not hurt numerics much, but can cause issues with plotting algorithms, so
     we choose to enforce a regular meshgrid in phi,theta based on the first
     columns,rows of x and y. THIS IS ANOTHER CONSEQUENCE OF STRANGE DESIGN
     CHOICES FOR THE MIX FILE FORMAT.
   
   """
   
   phis = p.arctan2(y[:,-1],x[:,-1])
   phis[phis<0] = phis[phis<0] + 2*p.pi
   
   if oh:
     phis[0] = 2*p.pi
     phis[-1] = 0
   else:
     phis[0] = 0
     phis[-1] = 2*p.pi
      
   
   thetas = p.arcsin(x[0,:]) # assumes x[0,:] corresponds to y==0
   
   phi,theta = p.meshgrid(phis, thetas, indexing='ij')
   
   if oh:
      theta = p.pi - theta
   
        
   return (phi, theta)


def _grad_phi(psi, xxx_todo_changeme):
   """
   Compute the 1D gradient of a scalar field in the phi direction on a unit-
   radius spherical shell. Assumes psi is a 2D array with phi changing along
   axis 0. We use central differences for interior points, one-sided differences
   for exterior points, and address simple periodic boundaries.
   """
   (phi,theta) = xxx_todo_changeme
   dphi = p.diff(phi,axis=0)
   dtheta = p.diff(theta,axis=1)


   # pre-allocate output grid
   dpsidphi = p.zeros(phi.shape)

   # use weighted central differences to compute gradient on the interior
   dpsidphi[1:-1,:] = (((p.diff(psi[:-1,:],axis=0) / dphi[:-1,:]**2 +
                         p.diff(psi[1:,:],axis=0) / dphi[1:,:]**2) /
                        (1/dphi[:-1,:] + 1/dphi[1:,:]) ) *
                       1./p.sin(theta[1:-1,:]) )

   
   # compute phi gradients at exterior points
   if p.mod(phi[0,0],2*p.pi) == p.mod(phi[-1,0],2*p.pi):
      # use weighted central differences to compute gradient if periodic boundary
      dpsidphi[[0,-1],:] = p.tile(((p.diff(psi[:2,:],axis=0) / dphi[0,:]**2 +
                                    p.diff(psi[-2:,:],axis=0) / dphi[-1,:]**2) /
                                   (1/dphi[0,:] + 1/dphi[-1,:]) ) *
                                  1./p.sin(theta[0,:]), (2,1) )
   else:
      # use one-sided difference to compute gradient if not a periodic boundary
      dpsidphi[-1,:] = (p.diff(psi[-2:,:],axis=0) / dphi[-1,:] / p.sin(theta[-1,:]) )
      dpsidphi[0,:] = (p.diff(psi[:2,:],axis=0) / dphi[0,:] / p.sin(theta[0,:]) )
   
   
   return dpsidphi


def _grad_theta(psi, xxx_todo_changeme1):
   """
   Compute the 1D gradient of a scalar field in the theta direction on a unit-
   radius spherical shell.  Assumes psi is a 2D array with theta changing along
   axis 1. We use central differences for interior points, one-sided differences
   for exterior points, and address simple periodic boundaries.
   """
   (phi,theta) = xxx_todo_changeme1
   dphi = p.diff(phi,axis=0)
   dtheta = p.diff(theta,axis=1)


   # pre-allocate output grid
   dpsidtheta = p.zeros(theta.shape)
   
   
   # use weighted central differences to compute theta gradient on the interior
   dpsidtheta[:,1:-1] = (((p.diff(psi[:,:-1],axis=1) / dtheta[:,:-1]**2 +
                           p.diff(psi[:,1:],axis=1) / dtheta[:,1:]**2) /
                          (1/dtheta[:,:-1] + 1/dtheta[:,1:]) ) )

      
   # compute theta gradients at exterior points
   if p.mod(theta[0,0],2*p.pi) == p.mod(theta[0,-1],2*p.pi):
      # use weighted central differences to compute gradient if periodic boundary
      dpsidtheta[:,[0,-1]] = p.tile(((p.diff(psi[:,:2],axis=1) / dtheta[:,0]**2 +
                                      p.diff(psi[:,-2:],axis=1) / dtheta[:,-1]**2) /
                                     (1/dtheta[:,0] + 1/dtheta[:-1]) ), (1,2) )
   else:
      # use one-sided difference to compute gradient if not a periodic boundary
      dpsidtheta[:,-1] = (p.diff(psi[:,-2:],axis=1).T / dtheta[:,-1])
      dpsidtheta[:,0] = (p.diff(psi[:,:2],axis=1).T / dtheta[:,0])
   
   return dpsidtheta


def _grad2d_sph(psi, xxx_todo_changeme2):
   """
   Compute the gradient of a 2D scalar field on a unit-radius spherical shell.
   Axis==0 corresponds to the phi coordinate; axis==1 corresponds to the theta
   coordinate
   """
   (phi,theta) = xxx_todo_changeme2
   dpsidphi = _grad_phi(psi,(phi,theta))
   dpsidtheta = _grad_theta(psi,(phi,theta))
   
   return (dpsidphi,dpsidtheta)


def _div2d_sph(xxx_todo_changeme3, xxx_todo_changeme4):
   """
   Compute the divergence of a 2D vector field on a unit-radius spherical shell.
   """
   (vphi,vtheta) = xxx_todo_changeme3
   (phi,theta) = xxx_todo_changeme4
   sintheta = p.sin(theta)
   
   # compute 1D gradients of vphi and vtheta*sintheta
   dvdphi = _grad_phi(vphi,(phi,theta))
   dvdtheta = _grad_theta(vtheta*sintheta,(phi,theta)) / sintheta
   
   # sum up the dv's
   div2d = dvdphi + dvdtheta
      
   return div2d


def _midGrid(inGrid):
   """
   Compute average value of inGrid's 2, 4, or 8 gridpoints for 1D, 2D, or 3D
   arrays. (is there a general n-dimensional way to do this? -EJR 7/2013)
   """
   
   from pylab import array,ones,zeros
   
   outGrid = zeros(array(inGrid.shape) - ones(inGrid.ndim))
   
   if (inGrid.ndim > 3):
      raise Exception('Too many dimensions in array')
   elif (inGrid.ndim > 2):
      outGrid = .125 * (inGrid[:-1,:-1,:-1] + inGrid[:-1,:-1, 1:] + 
                        inGrid[:-1, 1:,:-1] + inGrid[ 1:,:-1,:-1] +
                        inGrid[:-1, 1:, 1:] + inGrid[ 1:,:-1, 1:] +
                        inGrid[ 1:, 1:,:-1] + inGrid[ 1:, 1:, 1:])
   elif (inGrid.ndim > 1):
      outGrid = .25 * (inGrid[:-1,:-1] + inGrid[:-1,1:] + inGrid[1:,:-1] + inGrid[1:,1:])
   else:
      outGrid = .5 * (inGrid[:-1] + inGrid[1:])
        
   
   return outGrid

