import numpy as np

class dalecs(object):
  """
  Class for Dipole-Aligned Loop Equivalent Current System (DALECS). This is a
  thin wrapper for dalecs_sphere. See dalecs_sphere docstrings for details.
  
  Parameters
  ----------
  ion_phi_theta : tuple or list of longitude and colatitude meshgrids
                  (required)
  ion_rho       : scalar indicating a spical ionosphere radius
                  (default=6500e3)
  ndip          : integer number of discrete dipole segments
                  (default=10)
  isI           : True if currents are expected (not densities)
                  (default=False)
  iono          : True generates ionospheric segment
                  (default=True)
  fac           : True generates field-aligned current segment
                  (default=True)
  equator       : True generates equatorial segment
                  (default=True)
  """
  def __init__(self, ion_phi_theta, ion_rho=6500e3, ndip=10, isI=False,
               iono=True, fac=True, equator=True):
    
    # for now, DALECS must be defined in spherical coordinates, although once
    # defined, they can be converted to Cartesian and back again; the grid will
    # only be regular in dipole coordinates.
    self.coords = "spherical"

    # ion_phi and ion_theta are grids of longitude-like and colatitude-like
    # coordinates; they should be compatible with numpy's meshgrid()
    self.ion_phi = ion_phi_theta[0]
    self.ion_theta = ion_phi_theta[1]
    
    # ion_radius - ionosphere radius
    self.ion_rho = ion_rho
    
    # ndip - number of discrete dipole-aligned segments
    self.ndip = ndip
    
    # is_I - are we working with current (True) or current density (False)
    self.isI = isI
    
    # generate min/max longitude and colatitude values for cells
    rion_min, rion_max = _edgeGrid((self.ion_phi, self.ion_theta))
    
    # define radius min/max so that nothing gets trimmed
    rion_min.append(np.zeros(self.ion_phi.shape) + self.ion_rho)
    rion_max.append(np.zeros(self.ion_phi.shape) + np.Inf)

    # amps_type1 and amps_type2 are the same as _Jvecs_type1_iono[0] and
    # _Jvecs_type2_iono[1], respectively, but since it is possible
    # to trim _Jvecs*, we need some way to keep our scaling functions
    # accessible, regardless of any trimming
    self.amps_type1 = np.ones(self.ion_phi.shape)
    self.amps_type2 = np.ones(self.ion_theta.shape)

    # _bs_?_matrix_type1 and _bs_?_matrix_type2 may hold
    # scaling matrices to speed up Biot-Savart calculations
    self._bs_sphere_matrix_type1 = None
    self._bs_sphere_matrix_type2 = None
    self._bs_cart_matrix_type1 = None
    self._bs_cart_matrix_type2 = None

    # generate full type1 and type2 current loops
    print('Initializing ', self.ion_phi.size, ' type1 Bostrom loops')
    (self._rvecs_type1_iono, 
     self._Jvecs_type1_iono, 
     self._dvecs_type1_iono) = dalecs_sphere(
       rion_min, rion_max,
       (self.amps_type1, np.zeros(self.amps_type2.shape)),
       n=self.ndip, isI=self.isI, type2=False,
       iono=(True & iono), fac=False, equator=False
    )
    (self._rvecs_type1_fac, 
     self._Jvecs_type1_fac, 
     self._dvecs_type1_fac) = dalecs_sphere(
       rion_min, rion_max,
       (self.amps_type1, np.zeros(self.amps_type2.shape)),
       n=self.ndip, isI=self.isI, type2=False,
       iono=False, fac=(True & fac), equator=False
    )
    (self._rvecs_type1_equator, 
     self._Jvecs_type1_equator, 
     self._dvecs_type1_equator) = dalecs_sphere(
       rion_min, rion_max,
       (self.amps_type1, np.zeros(self.amps_type2.shape)),
       n=self.ndip, isI=self.isI, type2=False,
       iono=False, fac=False, equator=(True & equator)
    )

    print('Initializing ', self.ion_phi.size, ' type2 Bostrom loops')
    (self._rvecs_type2_iono, 
     self._Jvecs_type2_iono, 
     self._dvecs_type2_iono) = dalecs_sphere(
       rion_min, rion_max,
       (np.zeros(self.amps_type1.shape), self.amps_type2),
       n=self.ndip, isI=self.isI, type1=False,
       iono=(True & iono), fac=False, equator=False
    )    
    (self._rvecs_type2_fac, 
     self._Jvecs_type2_fac, 
     self._dvecs_type2_fac) = dalecs_sphere(
       rion_min, rion_max,
       (np.zeros(self.amps_type1.shape), self.amps_type2),
       n=self.ndip, isI=self.isI, type1=False,
       iono=False, fac=(True & fac), equator=False
    )    
    (self._rvecs_type2_equator, 
     self._Jvecs_type2_equator, 
     self._dvecs_type2_equator) = dalecs_sphere(
       rion_min, rion_max,
       (np.zeros(self.amps_type1.shape), self.amps_type2),
      n=self.ndip, isI=self.isI, type1=False,
       iono=False, fac=False, equator=(True & equator)
    )    
   
  
  def copy(self):
    """
    Return a deep copy of this object
    """
    import copy
    return copy.deepcopy(self)
    
  
  def trim(self,
           phi_min=None,
           phi_max=None,
           theta_min=None,
           theta_max=None,
           rho_min=None,
           rho_max=None):
    """
    Return a new dalecs object with unwanted phi, theta, or rho ranges removed
    """
    out = self.copy()

    if out.coords == "cartesian":
      # convert to spherical since trim assumes spherical coordinates
      out.spherical()
      change_back = True
    else:
      change_back = False
    
    # _trim_phi_theta_rho() the copies of rvecs*, Jvecs*, and dvecs*
    (out._rvecs_type1_iono,
     out._Jvecs_type1_iono,
     out._dvecs_type1_iono) = _trim_phi_theta_rho(
      out._rvecs_type1_iono, out._Jvecs_type1_iono, out._dvecs_type1_iono,
      phi_min=phi_min, phi_max=phi_max,
      theta_min=theta_min, theta_max=theta_max,
      rho_min=rho_min, rho_max=rho_max
    )
    (out._rvecs_type1_fac,
     out._Jvecs_type1_fac,
     out._dvecs_type1_fac) = _trim_phi_theta_rho(
      out._rvecs_type1_fac, out._Jvecs_type1_fac, out._dvecs_type1_fac,
      phi_min=phi_min, phi_max=phi_max,
      theta_min=theta_min, theta_max=theta_max,
      rho_min=rho_min, rho_max=rho_max
    )
    (out._rvecs_type1_equator,
     out._Jvecs_type1_equator,
     out._dvecs_type1_equator) = _trim_phi_theta_rho(
      out._rvecs_type1_equator, out._Jvecs_type1_equator, out._dvecs_type1_equator,
      phi_min=phi_min, phi_max=phi_max,
      theta_min=theta_min, theta_max=theta_max,
      rho_min=rho_min, rho_max=rho_max
    )

    (out._rvecs_type2_iono,
     out._Jvecs_type2_iono,
     out._dvecs_type2_iono) = _trim_phi_theta_rho(
      out._rvecs_type2_iono, out._Jvecs_type2_iono, out._dvecs_type2_iono,
      phi_min=phi_min, phi_max=phi_max,
      theta_min=theta_min, theta_max=theta_max,
      rho_min=rho_min, rho_max=rho_max
    )
    (out._rvecs_type2_fac,
     out._Jvecs_type2_fac,
     out._dvecs_type2_fac) = _trim_phi_theta_rho(
      out._rvecs_type2_fac, out._Jvecs_type2_fac, out._dvecs_type2_fac,
      phi_min=phi_min, phi_max=phi_max,
      theta_min=theta_min, theta_max=theta_max,
      rho_min=rho_min, rho_max=rho_max
    )
    (out._rvecs_type2_equator,
     out._Jvecs_type2_equator,
     out._dvecs_type2_equator) = _trim_phi_theta_rho(
      out._rvecs_type2_equator, out._Jvecs_type2_equator, out._dvecs_type2_equator,
      phi_min=phi_min, phi_max=phi_max,
      theta_min=theta_min, theta_max=theta_max,
      rho_min=rho_min, rho_max=rho_max
    )
    
    if change_back:
      # change back to Cartesian
      out.cartesian()

    return out
    
    
  def scale(self, Jion):
    """
    Rescale's DALECS by modifying dalecs_sphere's amps_type1 and amps_type2.
    No multiplication is performed unless/until user retrieves Jvecs,
    Jvecs_type1, of Jvecs_type2.
    """
    if (np.shape(Jion[0]) != np.shape(self._dvecs_type1_iono) or
        np.shape(Jion[1]) != np.shape(self._dvecs_type2_iono)):
      raise Exception("Input Jion dimensions do not match DALECS' dimensions")
    else:
      # replace scaling matrices amps_type1 and amps_type2
      self.amps_type1 = Jion[0]
      self.amps_type2 = Jion[1]
  
  
  @property
  def rvecs(self):
    """
    Return rvecs for Type1 and Type2 current systems as single object array
    """
    phis1, thetas1, rhos1 = self.rvecs_type1
    phis2, thetas2, rhos2 = self.rvecs_type2
    
    phis = np.empty(phis1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(phis1.flat, phis2.flat)):
      phis.flat[i] = np.hstack((one, two))
    
    thetas = np.empty(thetas1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(thetas1.flat, thetas2.flat)):
      thetas.flat[i] = np.hstack((one, two))
      
    rhos = np.empty(rhos1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(rhos1.flat, rhos2.flat)):
      rhos.flat[i] = np.hstack((one, two))
    
    return phis, thetas, rhos
  
  
  @property
  def rvecs_type1(self):
    """
    Return component rvecs for Type1 current systems as single object array
    """
    phis1, thetas1, rhos1 = self._rvecs_type1_iono
    phis2, thetas2, rhos2 = self._rvecs_type1_fac
    phis3, thetas3, rhos3 = self._rvecs_type1_equator
    
    phis = np.empty(phis1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(phis1.flat, phis2.flat, phis3.flat)):
      phis.flat[i] = np.hstack((one, two, three))
    
    thetas = np.empty(thetas1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(thetas1.flat, thetas2.flat, thetas3.flat)):
      thetas.flat[i] = np.hstack((one, two, three))
      
    rhos = np.empty(rhos1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(rhos1.flat, rhos2.flat, rhos3.flat)):
      rhos.flat[i] = np.hstack((one, two, three))
    
    return phis, thetas, rhos

  
  @property
  def rvecs_type2(self):
    """
    Return component rvecs for Type2 current systems as single object array
    """
    phis1, thetas1, rhos1 = self._rvecs_type2_iono
    phis2, thetas2, rhos2 = self._rvecs_type2_fac
    phis3, thetas3, rhos3 = self._rvecs_type2_equator
    
    phis = np.empty(phis1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(phis1.flat, phis2.flat, phis3.flat)):
      phis.flat[i] = np.hstack((one, two, three))
    
    thetas = np.empty(thetas1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(thetas1.flat, thetas2.flat, thetas3.flat)):
      thetas.flat[i] = np.hstack((one, two, three))
      
    rhos = np.empty(rhos1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(rhos1.flat, rhos2.flat, rhos3.flat)):
      rhos.flat[i] = np.hstack((one, two, three))
    
    return phis, thetas, rhos

  
  @property
  def Jvecs(self):
    """
    Return Jvecs for Type1 and Type2 current systems as single object array
    """
    Jphis1, Jthetas1, Jrhos1 = self.Jvecs_type1
    Jphis2, Jthetas2, Jrhos2 = self.Jvecs_type2
    
    Jphis = np.empty(Jphis1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(Jphis1.flat, Jphis2.flat)):
      Jphis.flat[i] = np.hstack((one, two))
    
    Jthetas = np.empty(Jthetas1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(Jthetas1.flat, Jthetas2.flat)):
      Jthetas.flat[i] = np.hstack((one, two))
      
    Jrhos = np.empty(Jrhos1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(Jrhos1.flat, Jrhos2.flat)):
      Jrhos.flat[i] = np.hstack((one, two))
    
    return Jphis, Jthetas, Jrhos
  
  
  @property
  def Jvecs_type1(self):
    """
    Return scaled component Jvecs for Type1 current systems as single array
    """
    Jphis1, Jthetas1, Jrhos1 = self._Jvecs_type1_iono
    Jphis2, Jthetas2, Jrhos2 = self._Jvecs_type1_fac
    Jphis3, Jthetas3, Jrhos3 = self._Jvecs_type1_equator
    
    Jphis = np.empty(Jphis1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(Jphis1.flat, Jphis2.flat, Jphis3.flat)):
      Jphis.flat[i] = np.hstack((one, two, three)) * self.amps_type1.flat[i]
    
    Jthetas = np.empty(Jthetas1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(Jthetas1.flat, Jthetas2.flat, Jthetas3.flat)):
      Jthetas.flat[i] = np.hstack((one, two, three)) * self.amps_type1.flat[i]
      
    Jrhos = np.empty(Jrhos1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(Jrhos1.flat, Jrhos2.flat, Jrhos3.flat)):
      Jrhos.flat[i] = np.hstack((one, two, three)) * self.amps_type1.flat[i]
    
    return Jphis, Jthetas, Jrhos

  
  @property
  def Jvecs_type2(self):
    """
    Return scaled component Jvecs for Type2 current systems as single array
    """
    Jphis1, Jthetas1, Jrhos1 = self._Jvecs_type2_iono
    Jphis2, Jthetas2, Jrhos2 = self._Jvecs_type2_fac
    Jphis3, Jthetas3, Jrhos3 = self._Jvecs_type2_equator
    
    Jphis = np.empty(Jphis1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(Jphis1.flat, Jphis2.flat, Jphis3.flat)):
      Jphis.flat[i] = np.hstack((one, two, three)) * self.amps_type2.flat[i]
    
    Jthetas = np.empty(Jthetas1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(Jthetas1.flat, Jthetas2.flat, Jthetas3.flat)):
      Jthetas.flat[i] = np.hstack((one, two, three)) * self.amps_type2.flat[i]
      
    Jrhos = np.empty(Jrhos1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(Jrhos1.flat, Jrhos2.flat, Jrhos3.flat)):
      Jrhos.flat[i] = np.hstack((one, two, three)) * self.amps_type2.flat[i]
    
    return Jphis, Jthetas, Jrhos

  
  @property
  def dvecs(self):
    """
    Return dvecs for Type1 and Type2 current systems as single object array
    """
    ds1 = self.dvecs_type1
    ds2 = self.dvecs_type2
    
    ds = np.empty(ds1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(ds1.flat, ds2.flat)):
      ds.flat[i] = np.hstack((one, two))
    
    return ds

  @property
  def dvecs_type1(self):
    """
    Return component dvecs for Type1 current systems as single object array
    """
    ds1 = self._dvecs_type1_iono
    ds2 = self._dvecs_type1_fac
    ds3 = self._dvecs_type1_equator

    ds = np.empty(ds1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(ds1.flat, ds2.flat, ds3.flat)):
      ds.flat[i] = np.hstack((one, two, three))
    
    return ds

  @property
  def dvecs_type2(self):
    """
    Return component dvecs for Type2 current systems as single object array
    """
    ds1 = self._dvecs_type2_iono
    ds2 = self._dvecs_type2_fac
    ds3 = self._dvecs_type2_equator

    ds = np.empty(ds1.shape, dtype='object')
    for i, (one, two, three) in enumerate(zip(ds1.flat, ds2.flat, ds3.flat)):
      ds.flat[i] = np.hstack((one, two, three))
    
    return ds

  def cartesian(self):
    """
    Convert rvecs and Jvecs for all current systems into Cartesian coordinates
    """

    if self.coords == "cartesian":
      print("DALECS already in Cartesian coordinates.")

    else:
      self.coords = "cartesian"

      rvecs1 = [self._rvecs_type1_iono,
                self._rvecs_type1_fac, 
                self._rvecs_type1_equator]
      rvecs2 = [self._rvecs_type2_iono,
                self._rvecs_type2_fac, 
                self._rvecs_type2_equator]
      Jvecs1 = [self._Jvecs_type1_iono,
                self._Jvecs_type1_fac, 
                self._Jvecs_type1_equator]
      Jvecs2 = [self._Jvecs_type2_iono,
                self._Jvecs_type2_fac, 
                self._Jvecs_type2_equator]

      for rvecs_type1, rvecs_type2, Jvecs_type1, Jvecs_type2 in zip(
        rvecs1, rvecs2, Jvecs1, Jvecs2):
        
        for i in range(len(rvecs_type1[0].flat)):
          # convert Jvecs first, since rvecs must still be spherical
          (Jvecs_type1[0].flat[i],
           Jvecs_type1[1].flat[i],
           Jvecs_type1[2].flat[i]) = _sp2cart_dir(
             (Jvecs_type1[0].flat[i],
              Jvecs_type1[1].flat[i],
              Jvecs_type1[2].flat[i]),          
             (rvecs_type1[0].flat[i],
              rvecs_type1[1].flat[i],
              rvecs_type1[2].flat[i])
          )
          (Jvecs_type2[0].flat[i],
           Jvecs_type2[1].flat[i],
           Jvecs_type2[2].flat[i]) = _sp2cart_dir(
             (Jvecs_type2[0].flat[i],
              Jvecs_type2[1].flat[i],
              Jvecs_type2[2].flat[i]),          
             (rvecs_type2[0].flat[i],
              rvecs_type2[1].flat[i],
              rvecs_type2[2].flat[i])
          )
          # now we can convert rvecs
          (rvecs_type1[0].flat[i],
           rvecs_type1[1].flat[i],
           rvecs_type1[2].flat[i]) = _sp2cart_pos(
             (rvecs_type1[0].flat[i],
              rvecs_type1[1].flat[i],
              rvecs_type1[2].flat[i])
          )
          (rvecs_type2[0].flat[i],
           rvecs_type2[1].flat[i],
           rvecs_type2[2].flat[i]) = _sp2cart_pos(
             (rvecs_type2[0].flat[i],
              rvecs_type2[1].flat[i],
              rvecs_type2[2].flat[i])
          )


  def spherical(self):
    """
    Convert rvecs for Type1 and Type2 current systems spherical coordinates
    """
    if self.coords == "spherical":
      print("DALECS already in spherical coordinates.")

    else:
      self.coords = "spherical"
      
      rvecs1 = [self._rvecs_type1_iono,
                self._rvecs_type1_fac, 
                self._rvecs_type1_equator]
      rvecs2 = [self._rvecs_type2_iono,
                self._rvecs_type2_fac, 
                self._rvecs_type2_equator]
      Jvecs1 = [self._Jvecs_type1_iono,
                self._Jvecs_type1_fac, 
                self._Jvecs_type1_equator]
      Jvecs2 = [self._Jvecs_type2_iono,
                self._Jvecs_type2_fac, 
                self._Jvecs_type2_equator]

      for rvecs_type1, rvecs_type2, Jvecs_type1, Jvecs_type2 in zip(
        rvecs1, rvecs2, Jvecs1, Jvecs2):
        
        for i in range(len(self.rvecs_type1[0].flat)):
          # convert Jvecs first, since rvecs must still be Cartesian
          (Jvecs_type1[0].flat[i],
           Jvecs_type1[1].flat[i],
           Jvecs_type1[2].flat[i]) = _cart2sp_dir(
             (Jvecs_type1[0].flat[i],
              Jvecs_type1[1].flat[i],
              Jvecs_type1[2].flat[i]),          
             (rvecs_type1[0].flat[i],
              rvecs_type1[1].flat[i],
              rvecs_type1[2].flat[i])
          )
          (Jvecs_type2[0].flat[i],
           Jvecs_type2[1].flat[i],
           Jvecs_type2[2].flat[i]) = _cart2sp_dir(
             (Jvecs_type2[0].flat[i],
              Jvecs_type2[1].flat[i],
              Jvecs_type2[2].flat[i]),          
             (rvecs_type2[0].flat[i],
              rvecs_type2[1].flat[i],
              rvecs_type2[2].flat[i])
          )
          # now we can convert rvecs
          (rvecs_type1[0].flat[i],
           rvecs_type1[1].flat[i],
           rvecs_type1[2].flat[i]) = _cart2sp_pos(
             (rvecs_type1[0].flat[i],
              rvecs_type1[1].flat[i],
              rvecs_type1[2].flat[i])
          )
          (rvecs_type2[0].flat[i],
           rvecs_type2[1].flat[i],
           rvecs_type2[2].flat[i]) = _cart2sp_pos(
             (rvecs_type2[0].flat[i],
              rvecs_type2[1].flat[i],
              rvecs_type2[2].flat[i])
          )

  
  def bs_sphere(self, obs_phi_theta_rho, matrix=False,
                type1=True, type2=True):
    """
    Integrate Biot-Savart in spherical coordinates.

    - full BS integration (default) can be slow; 
      if matrix=True, the first pass is even slower, but
      each subsequent call using the same DALECS object will
      be well over 1000x faster than full since it requires
      only a matrix multiplication.
    - type1 and type2 currents can be isolated
    """
    obs_phis = obs_phi_theta_rho[0]
    obs_thetas = obs_phi_theta_rho[1]
    obs_rhos = obs_phi_theta_rho[2]

    if matrix:
      if (self._bs_sphere_matrix_type1 is None or
          self._bs_sphere_matrix_type2 is None):
        
        print("Calculating BS matrices")
        
        # pre-allocate type1 BS matrix
        self._bs_sphere_matrix_type1 = np.empty(
          (3 * obs_phis.size,
           self.amps_type1.size)
        )
        # pre-allocate type2 BS matrix
        self._bs_sphere_matrix_type2 = np.empty(
          (3 * obs_phis.size,
           self.amps_type2.size)
        )
        
        # loop over each DALEC/Obs pair to fill out the scaling matrices
        for i in np.arange(obs_phis.size):
          for j in np.arange(self.amps_type1.size):

            # calculate deltaB at Obs[i] given type1 DALEC[j]
            (self._bs_sphere_matrix_type1[3*i + 0, j],
             self._bs_sphere_matrix_type1[3*i + 1, j],
             self._bs_sphere_matrix_type1[3*i + 2, j]) = np.hstack(
               bs_sphere(
                (np.hstack((self._rvecs_type1_iono[0].flat[j],
                            self._rvecs_type1_fac[0].flat[j],
                            self._rvecs_type1_equator[0].flat[j])),
                 np.hstack((self._rvecs_type1_iono[1].flat[j],
                            self._rvecs_type1_fac[1].flat[j],
                            self._rvecs_type1_equator[1].flat[j])),
                 np.hstack((self._rvecs_type1_iono[2].flat[j],
                            self._rvecs_type1_fac[2].flat[j],
                            self._rvecs_type1_equator[2].flat[j])) ),
                (np.hstack((self._Jvecs_type1_iono[0].flat[j],
                            self._Jvecs_type1_fac[0].flat[j],
                            self._Jvecs_type1_equator[0].flat[j])),
                 np.hstack((self._Jvecs_type1_iono[1].flat[j],
                            self._Jvecs_type1_fac[1].flat[j],
                            self._Jvecs_type1_equator[1].flat[j])),
                 np.hstack((self._Jvecs_type1_iono[2].flat[j],
                            self._Jvecs_type1_fac[2].flat[j],
                            self._Jvecs_type1_equator[2].flat[j])) ),
                np.hstack((self._dvecs_type1_iono.flat[j],
                           self._dvecs_type1_fac.flat[j],
                           self._dvecs_type1_equator.flat[j])),
                (obs_phis.flat[i], obs_thetas.flat[i], obs_rhos.flat[i])
               )
            ) # np.hstack required because bs_sphere() returns array of arrays

            # calculate deltaB at Obs[i] given type2 DALEC[j]
            (self._bs_sphere_matrix_type2[3*i + 0, j],
             self._bs_sphere_matrix_type2[3*i + 1, j],
             self._bs_sphere_matrix_type2[3*i + 2, j]) = np.hstack(
              bs_sphere(
                (np.hstack((self._rvecs_type2_iono[0].flat[j],
                            self._rvecs_type2_fac[0].flat[j],
                            self._rvecs_type2_equator[0].flat[j])),
                 np.hstack((self._rvecs_type2_iono[1].flat[j],
                            self._rvecs_type2_fac[1].flat[j],
                            self._rvecs_type2_equator[1].flat[j])),
                 np.hstack((self._rvecs_type2_iono[2].flat[j],
                            self._rvecs_type2_fac[2].flat[j],
                            self._rvecs_type2_equator[2].flat[j])) ),
                (np.hstack((self._Jvecs_type2_iono[0].flat[j],
                            self._Jvecs_type2_fac[0].flat[j],
                            self._Jvecs_type2_equator[0].flat[j])),
                 np.hstack((self._Jvecs_type2_iono[1].flat[j],
                            self._Jvecs_type2_fac[1].flat[j],
                            self._Jvecs_type2_equator[1].flat[j])),
                 np.hstack((self._Jvecs_type2_iono[2].flat[j],
                            self._Jvecs_type2_fac[2].flat[j],
                            self._Jvecs_type2_equator[2].flat[j])) ),
                np.hstack((self._dvecs_type2_iono.flat[j],
                           self._dvecs_type2_fac.flat[j],
                           self._dvecs_type2_equator.flat[j])),
                (obs_phis.flat[i], obs_thetas.flat[i], obs_rhos.flat[i])
              )
            ) # np.hstack required because bs_sphere() returns array of arrays

      # matrix is defined, now use it
      if type1:
        amps_type1 = np.nan_to_num(self.amps_type1.reshape(-1,1))
        (obs_Bphis1,
         obs_Bthetas1,
         obs_Brhos1) = np.matmul(self._bs_sphere_matrix_type1,
                                 amps_type1).reshape(-1,3).T
        obs_Bphis1 = obs_Bphis1.reshape(obs_phis.shape)
        obs_Bthetas1 = obs_Bthetas1.reshape(obs_thetas.shape)
        obs_Brhos1 = obs_Brhos1.reshape(obs_rhos.shape)
      else:
        obs_Bphis1 = np.zeros(obs_phis.shape)
        obs_Bthetas1 = np.zeros(obs_thetas.shape)
        obs_Brhos1 = np.zeros(obs_rhos.shape)

      if type2:
        amps_type2 = np.nan_to_num(self.amps_type2.reshape(-1,1))
        (obs_Bphis2,
         obs_Bthetas2,
         obs_Brhos2) = np.matmul(self._bs_sphere_matrix_type2,
                                 amps_type2).reshape(-1,3).T
        obs_Bphis2 = obs_Bphis2.reshape(obs_phis.shape)
        obs_Bthetas2 = obs_Bthetas2.reshape(obs_thetas.shape)
        obs_Brhos2 = obs_Brhos2.reshape(obs_rhos.shape)
      else:
        obs_Bphis2 = np.zeros(obs_phis.shape)
        obs_Bthetas2 = np.zeros(obs_thetas.shape)
        obs_Brhos2 = np.zeros(obs_rhos.shape)

    else:
      # generate deltaB at observation sites using full Biot-Savart
      
      # deltaB from type1 Bostrom loops
      if type1:
        (obs_Bphis1,
         obs_Bthetas1,
         obs_Brhos1) = bs_sphere(
          self.rvecs_type1,
          self.Jvecs_type1,
          self.dvecs_type1,
          (obs_phis, obs_thetas, obs_rhos)
        )
      else:
        obs_Bphis1 = np.zeros(obs_phis.shape)
        obs_Bthetas1 = np.zeros(obs_thetas.shape)
        obs_Brhos1 = np.zeros(obs_rhos.shape)
      
      # deltaB from type 2 Bostrom loops
      if type2:
        (obs_Bphis2,
         obs_Bthetas2,
         obs_Brhos2) = bs_sphere(
          self.rvecs_type2,
          self.Jvecs_type2,
          self.dvecs_type2,
          (obs_phis, obs_thetas, obs_rhos)
        )
      else:
        obs_Bphis2 = np.zeros(obs_phis.shape)
        obs_Bthetas2 = np.zeros(obs_thetas.shape)
        obs_Brhos2 = np.zeros(obs_rhos.shape)
    
    return (obs_Bphis1 + obs_Bphis2,
            obs_Bthetas1 + obs_Bthetas2,
            obs_Brhos1 + obs_Brhos2)



  def bs_cart(self, obs_x_y_z, matrix=False,
              type1=True, type2=True):
    """
    Integrate Biot-Savart in Cartesian coordinates
    """
    obs_x = obs_x_y_z[0]
    obs_y = obs_x_y_z[1]
    obs_z = obs_x_y_z[2]

    if matrix:
      if (self._bs_cart_matrix_type1 is None or
          self._bs_cart_matrix_type2 is None):
        
        print("Calculating BS matrices")
        
        # pre-allocate type1 BS matrix
        self._bs_cart_matrix_type1 = np.empty(
          (3 * obs_x.size,
           self.amps_type1.size)
        )
        # pre-allocate type2 BS matrix
        self._bs_cart_matrix_type2 = np.empty(
          (3 * obs_x.size,
           self.amps_type2.size)
        )
        
        # loop over each DALEC/Obs pair to fill out the scaling matrices
        for i in np.arange(obs_x.size):
          for j in np.arange(self.amps_type1.size):

            # calculate deltaB at Obs[i] given type1 DALEC[j]
            (self._bs_cart_matrix_type1[3*i + 0, j],
             self._bs_cart_matrix_type1[3*i + 1, j],
             self._bs_cart_matrix_type1[3*i + 2, j]) = np.hstack(
               bs_cart(
                (np.hstack((self._rvecs_type1_iono[0].flat[j],
                            self._rvecs_type1_fac[0].flat[j],
                            self._rvecs_type1_equator[0].flat[j])),
                 np.hstack((self._rvecs_type1_iono[1].flat[j],
                            self._rvecs_type1_fac[1].flat[j],
                            self._rvecs_type1_equator[1].flat[j])),
                 np.hstack((self._rvecs_type1_iono[2].flat[j],
                            self._rvecs_type1_fac[2].flat[j],
                            self._rvecs_type1_equator[2].flat[j])) ),
                (np.hstack((self._Jvecs_type1_iono[0].flat[j],
                            self._Jvecs_type1_fac[0].flat[j],
                            self._Jvecs_type1_equator[0].flat[j])),
                 np.hstack((self._Jvecs_type1_iono[1].flat[j],
                            self._Jvecs_type1_fac[1].flat[j],
                            self._Jvecs_type1_equator[1].flat[j])),
                 np.hstack((self._Jvecs_type1_iono[2].flat[j],
                            self._Jvecs_type1_fac[2].flat[j],
                            self._Jvecs_type1_equator[2].flat[j])) ),
                np.hstack((self._dvecs_type1_iono.flat[j],
                           self._dvecs_type1_fac.flat[j],
                           self._dvecs_type1_equator.flat[j])),
                (obs_x.flat[i], obs_y.flat[i], obs_z.flat[i])
               )
            ) # np.hstack required because bs_cart() returns array of arrays

            # calculate deltaB at Obs[i] given type2 DALEC[j]
            (self._bs_cart_matrix_type2[3*i + 0, j],
             self._bs_cart_matrix_type2[3*i + 1, j],
             self._bs_cart_matrix_type2[3*i + 2, j]) = np.hstack(
              bs_cart(
                (np.hstack((self._rvecs_type2_iono[0].flat[j],
                            self._rvecs_type2_fac[0].flat[j],
                            self._rvecs_type2_equator[0].flat[j])),
                 np.hstack((self._rvecs_type2_iono[1].flat[j],
                            self._rvecs_type2_fac[1].flat[j],
                            self._rvecs_type2_equator[1].flat[j])),
                 np.hstack((self._rvecs_type2_iono[2].flat[j],
                            self._rvecs_type2_fac[2].flat[j],
                            self._rvecs_type2_equator[2].flat[j])) ),
                (np.hstack((self._Jvecs_type2_iono[0].flat[j],
                            self._Jvecs_type2_fac[0].flat[j],
                            self._Jvecs_type2_equator[0].flat[j])),
                 np.hstack((self._Jvecs_type2_iono[1].flat[j],
                            self._Jvecs_type2_fac[1].flat[j],
                            self._Jvecs_type2_equator[1].flat[j])),
                 np.hstack((self._Jvecs_type2_iono[2].flat[j],
                            self._Jvecs_type2_fac[2].flat[j],
                            self._Jvecs_type2_equator[2].flat[j])) ),
                np.hstack((self._dvecs_type2_iono.flat[j],
                           self._dvecs_type2_fac.flat[j],
                           self._dvecs_type2_equator.flat[j])),
                (obs_x.flat[i], obs_y.flat[i], obs_z.flat[i])
              )
            ) # np.hstack required because bs_cart() returns array of arrays

      # matrix is defined, now use it
      if type1:
        # any NaN currents should just be set to zero before matmul
        amps_type1 = np.nan_to_num(self.amps_type1.reshape(-1,1))
        (obs_Bx1,
         obs_By1,
         obs_Bz1) = np.matmul(self._bs_cart_matrix_type1,
                              amps_type1).reshape(-1,3).T
        obs_Bx1 = obs_Bx1.reshape(obs_x.shape)
        obs_By1 = obs_By1.reshape(obs_y.shape)
        obs_Bz1 = obs_Bz1.reshape(obs_z.shape)
      else:
        obs_Bx1 = np.zeros(obs_x.shape)
        obs_By1 = np.zeros(obs_y.shape)
        obs_Bz1 = np.zeros(obs_z.shape)

      if type2:
        amps_type2 = np.nan_to_num(self.amps_type2.reshape(-1,1))
        (obs_Bx2,
         obs_By2,
         obs_Bz2) = np.matmul(self._bs_cart_matrix_type2,
                              amps_type2).reshape(-1,3).T
        obs_Bx2 = obs_Bx2.reshape(obs_x.shape)
        obs_By2 = obs_By2.reshape(obs_y.shape)
        obs_Bz2 = obs_Bz2.reshape(obs_z.shape)
      else:
        obs_Bx2 = np.zeros(obs_x.shape)
        obs_By2 = np.zeros(obs_y.shape)
        obs_Bz2 = np.zeros(obs_z.shape)

    else:
      # generate deltaB at observation sites using full Biot-Savart
      
      # deltaB from type1 Bostrom loops
      if type1:
        (obs_Bx1,
         obs_By1,
         obs_Bz1) = bs_cart(
          self.rvecs_type1,
          self.Jvecs_type1,
          self.dvecs_type1,
          (obs_x, obs_y, obs_z)
        )
      else:
        obs_Bx1 = np.zeros(obs_x.shape)
        obs_By1 = np.zeros(obs_y.shape)
        obs_Bz1 = np.zeros(obs_z.shape)
      
      # deltaB from type 2 Bostrom loops
      if type2:
        (obs_Bx2,
         obs_By2,
         obs_Bz2) = bs_cart(
          self.rvecs_type2,
          self.Jvecs_type2,
          self.dvecs_type2,
          (obs_x, obs_y, obs_z)
        )
      else:
        obs_Bx2 = np.zeros(obs_x.shape)
        obs_By2 = np.zeros(obs_y.shape)
        obs_Bz2 = np.zeros(obs_z.shape)
    
    return (obs_Bx1 + obs_Bx2,
            obs_By1 + obs_By2,
            obs_Bz1 + obs_Bz2)


class ralecs(object):
  """
  Class for Radial-Aligned Loop Equivalent Current System (RALECS). This is a
  thin wrapper for ralecs_sphere. See ralecs_sphere docstring for details.
  
  Parameters
  ----------
  ion_phi_theta : tuple or list of longitude and colatitude meshgrids
                  (required)
  ion_rho       : scalar indicating a spical ionosphere radius
                  (default=6500e3)
  nrad          : integer number of discrete radial segments
                  (default=10)
  isI           : True if currents are expected (not densities)
                  (default=False)
  """
  def __init__(self, ion_phi_theta, ion_rho=6500e3, nrad=10, isI=False):
    
    # for now, RALECS must be defined in spherical coordinates, although once
    # defined, they can be converted to Cartesian and back again
    self.coords = "spherical"

    # ion_phi and ion_theta are grids of longitude-like and colatitude-like
    # coordinates; they should be compatible with numpy's meshgrid()
    self.ion_phi = ion_phi_theta[0]
    self.ion_theta = ion_phi_theta[1]
    
    # ion_radius - ionosphere radius
    self.ion_rho = ion_rho
    
    # nrad - number of discrete dipole-aligned segments
    self.nrad = nrad
    
    # is_I - are we working with current (True) or current density (False)
    self.isI = isI
    
    # generate min/max longitude and colatitude values for cells
    rion_min, rion_max = _edgeGrid((self.ion_phi, self.ion_theta))
    
    # define radius min/max so that nothing gets trimmed
    rion_min.append(np.zeros(self.ion_phi.shape) + self.ion_rho)
    rion_max.append(np.zeros(self.ion_phi.shape) + np.Inf)

    # generate full type1 and type2 current loops
    print('Initializing type1 RALEC loops')
    (self.rvecs_type1, 
     self.Jvecs_type1, 
     self.dvecs_type1) = ralecs_sphere(rion_min, rion_max,
                                       (np.ones(self.ion_phi.shape),
                                        np.zeros(self.ion_theta.shape)),
                                       n=self.nrad,
                                       isI=self.isI,
                                       type2=False)
    print('Initializing type2 RALEC loops')
    (self.rvecs_type2, 
     self.Jvecs_type2, 
     self.dvecs_type2) = ralecs_sphere(rion_min, rion_max,
                                       (np.zeros(self.ion_phi.shape),
                                        np.ones(self.ion_theta.shape)),
                                       n=self.nrad,
                                       isI=self.isI,
                                       type1=False)    
    
  
  def copy(self):
    """
    Return a deep copy of this object
    """
    import copy
    return copy.deepcopy(self)
    
  
  def trim(self,
           phi_min=None,
           phi_max=None,
           theta_min=None,
           theta_max=None,
           rho_min=None,
           rho_max=None):
    """
    Return a ralecs object with unwanted phi, theta, or rho ranges removed
    """
    out = self.copy()
    
    # _trim_phi_theta_rho() the copies of rvecs*, Jvecs*, and dvecs*
    (out.rvecs_type1,
     out.Jvecs_type1,
     out.dvecs_type1) = _trim_phi_theta_rho(
      out.rvecs_type1, out.Jvecs_type1, out.dvecs_type1,
      phi_min=phi_min, phi_max=phi_max,
      theta_min=theta_min, theta_max=theta_max,
      rho_min=rho_min, rho_max=rho_max
    )
    (out.rvecs_type2,
     out.Jvecs_type2,
     out.dvecs_type2) = _trim_phi_theta_rho(
      out.rvecs_type2, out.Jvecs_type2, out.dvecs_type2,
      phi_min=phi_min, phi_max=phi_max,
      theta_min=theta_min, theta_max=theta_max,
      rho_min=rho_min, rho_max=rho_max
    )
    
    return out
    
    
  def scale(self, Jion):
    """
    Return a ralecs object with Jvecs scaled by Jion 
    """
    out = self.copy()
    
    if (np.shape(Jion[0]) != np.shape(self.Jvecs_type1[0]) or
        np.shape(Jion[0]) != np.shape(self.Jvecs_type2[0]) or
        np.shape(Jion[1]) != np.shape(self.Jvecs_type1[1]) or
        np.shape(Jion[1]) != np.shape(self.Jvecs_type2[1])):
      raise Exception('Input Jion dimensions do not match Jvecs')
    else:
      for i in range(len(out.Jvecs_type1[0].flat)):
        out.Jvecs_type1[0].flat[i] = out.Jvecs_type1[0].flat[i] * Jion[0].flat[i]
        out.Jvecs_type1[1].flat[i] = out.Jvecs_type1[1].flat[i] * Jion[0].flat[i]
        out.Jvecs_type1[2].flat[i] = out.Jvecs_type1[2].flat[i] * Jion[0].flat[i]
      for i in range(len(out.Jvecs_type2[1].flat)):
        out.Jvecs_type2[0].flat[i] = out.Jvecs_type2[0].flat[i] * Jion[1].flat[i]
        out.Jvecs_type2[1].flat[i] = out.Jvecs_type2[1].flat[i] * Jion[1].flat[i]
        out.Jvecs_type2[2].flat[i] = out.Jvecs_type2[2].flat[i] * Jion[1].flat[i]
    
    return out

  @property
  def rvecs(self):
    """
    Return rvecs for Type1 and Type2 current systems as single object array
    """
    phis1, thetas1, rhos1 = self.rvecs_type1
    phis2, thetas2, rhos2 = self.rvecs_type2
    
    phis = np.empty(phis1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(phis1.flat, phis2.flat)):
      phis.flat[i] = np.hstack((one, two))
    
    thetas = np.empty(thetas1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(thetas1.flat, thetas2.flat)):
      thetas.flat[i] = np.hstack((one, two))
      
    rhos = np.empty(rhos1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(rhos1.flat, rhos2.flat)):
      rhos.flat[i] = np.hstack((one, two))
    
    return phis, thetas, rhos
  
  
  @property
  def Jvecs(self):
    """
    Return Jvecs for Type1 and Type2 current systems as single object array
    """
    Jphis1, Jthetas1, Jrhos1 = self.Jvecs_type1
    Jphis2, Jthetas2, Jrhos2 = self.Jvecs_type2
    
    Jphis = np.empty(Jphis1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(Jphis1.flat, Jphis2.flat)):
      Jphis.flat[i] = np.hstack((one, two))
    
    Jthetas = np.empty(Jthetas1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(Jthetas1.flat, Jthetas2.flat)):
      Jthetas.flat[i] = np.hstack((one, two))
      
    Jrhos = np.empty(Jrhos1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(Jrhos1.flat, Jrhos2.flat)):
      Jrhos.flat[i] = np.hstack((one, two))
    
    return Jphis, Jthetas, Jrhos
  

  @property
  def dvecs(self):
    """
    Return dvecs for Type1 and Type2 current systems as single object array
    """
    ds1 = self.dvecs_type1
    ds2 = self.dvecs_type2
    
    ds = np.empty(ds1.shape, dtype='object')
    for i, (one, two) in enumerate(zip(ds1.flat, ds2.flat)):
      ds.flat[i] = np.hstack((one, two))
        
    return ds


  def cartesian(self):
    """
    Convert rvecs for Type1 and Type2 current systems Cartesian coordinates
    """

    if self.coords == "cartesian":
      print("RALECS already in Cartesian coordinates.")

    else:
      self.coords = "cartesian"
      for i in range(len(self.rvecs_type1[0])):
        # convert Jvecs first, since we'll change rvecs
        (self.Jvecs_type1[0].flat[i],
         self.Jvecs_type1[1].flat[i],
         self.Jvecs_type1[2].flat[i]) = _sp2cart_dir(
           (self.Jvecs_type1[0].flat[i],
            self.Jvecs_type1[1].flat[i],
            self.Jvecs_type1[2].flat[i]),          
           (self.rvecs_type1[0].flat[i],
            self.rvecs_type1[1].flat[i],
            self.rvecs_type1[2].flat[i])
        )
        (self.Jvecs_type2[0].flat[i],
         self.Jvecs_type2[1].flat[i],
         self.Jvecs_type2[2].flat[i]) = _sp2cart_dir(
           (self.Jvecs_type2[0].flat[i],
            self.Jvecs_type2[1].flat[i],
            self.Jvecs_type2[2].flat[i]),          
           (self.rvecs_type2[0].flat[i],
            self.rvecs_type2[1].flat[i],
            self.rvecs_type2[2].flat[i])
        )
        # now we can change rvecs
        (self.rvecs_type1[0].flat[i],
         self.rvecs_type1[1].flat[i],
         self.rvecs_type1[2].flat[i]) = _sp2cart_pos(
           (self.rvecs_type1[0].flat[i],
            self.rvecs_type1[1].flat[i],
            self.rvecs_type1[2].flat[i])
        )
        (self.rvecs_type2[0].flat[i],
         self.rvecs_type2[1].flat[i],
         self.rvecs_type2[2].flat[i]) = _sp2cart_pos(
           (self.rvecs_type2[0].flat[i],
            self.rvecs_type2[1].flat[i],
            self.rvecs_type2[2].flat[i])
        )


  def spherical(self):
    """
    Convert rvecs for Type1 and Type2 current systems spherical coordinates
    """
    if self.coords == "spherical":
      print("RALECS already in spherical coordinates.")

    else:
      self.coords = "spherical"
      for i in range(len(self.rvecs_type1[0])):
        # convert Jvecs first, since we'll change rvecs
        (self.Jvecs_type1[0].flat[i],
         self.Jvecs_type1[1].flat[i],
         self.Jvecs_type1[2].flat[i]) = _cart2sp_dir(
           (self.Jvecs_type1[0].flat[i],
            self.Jvecs_type1[1].flat[i],
            self.Jvecs_type1[2].flat[i]),          
           (self.rvecs_type1[0].flat[i],
            self.rvecs_type1[1].flat[i],
            self.rvecs_type1[2].flat[i])
        )
        (self.Jvecs_type2[0].flat[i],
         self.Jvecs_type2[1].flat[i],
         self.Jvecs_type2[2].flat[i]) = _cart2sp_dir(
           (self.Jvecs_type2[0].flat[i],
            self.Jvecs_type2[1].flat[i],
            self.Jvecs_type2[2].flat[i]),          
           (self.rvecs_type2[0].flat[i],
            self.rvecs_type2[1].flat[i],
            self.rvecs_type2[2].flat[i])
        )
        # now we can change rvecs
        (self.rvecs_type1[0].flat[i],
         self.rvecs_type1[1].flat[i],
         self.rvecs_type1[2].flat[i]) = _cart2sp_pos(
           (self.rvecs_type1[0].flat[i],
            self.rvecs_type1[1].flat[i],
            self.rvecs_type1[2].flat[i])
        )
        (self.rvecs_type2[0].flat[i],
         self.rvecs_type2[1].flat[i],
         self.rvecs_type2[2].flat[i]) = _cart2sp_pos(
           (self.rvecs_type2[0].flat[i],
            self.rvecs_type2[1].flat[i],
            self.rvecs_type2[2].flat[i])
        )




def dalecs_sphere(rion_min, rion_max, Jion, n=10, isI=False,
                  type1=True, type2=True,
                  iono=True, fac=True, equator=True):
  """
  Build a 3D Dipole-Aligned Loop Equivalent Current System (DALECS) in spherical
  coordinates given ionospheric current (density) on a 2D spherical shell. Each
  DALEC is comprised of a pair of discretized Bostrom loops: the type 1 loop is
  purely zonal in the ionosphere and equator, closing along dipole field-aligned
  currents; the type 2 loop is purely meridional in the ionosphere, radial at
  the quator, and closes along dipole field-aligned currents.

  INPUTS:
  - rion_min is a 3-tuple/list of scalars or same-shaped ND arrays that hold:
     phi_min   - minimum longitude (0 points to sun) of ionospheric current
     theta_min - minimum colatitude of ionospheric current
     rho_min   - minimum radius of returned currents
                 ** if rho_min is larger than rho_max, rho_max is taken
                    as the ionospheric radius, and rho_min sets the min
                    radius for returned DALEC elements, possibly reducing
                    the number of elements
  - rion_max is a 3-tuple/list of scalars or same-shaped ND arrays that hold:
     phi_max   - maximum longitude (0 points to sun) of ionospheric current
     theta_max - maximum colatitude of ionospheric current
     rho_max   - maximum radius of returned currents
                 ** if rho_max is larger than rho_min, rho_min is taken
                    as the ionospheric radius, and rho_max sets the max
                    radius for returned DALEC elements, possibly reducing
                    the number of elements
  - Jion is a 2-tuple/list of scalars or same-shaped ND arrays that hold:
     Jphi      - ionospheric current (density) in phi direction
     Jtheta    - ionospheric current (density) in theta direction
  - n is optional number of elements used to discretize each FAC; with
    4 FACs per DALEC, 2 ionospheric sheets, and 2 equatorial sheets, the
    total number of elements per DALEC is (4*n+4). Default n=10.
  - isI is optional flag to treat (jtheta,jphi) as a wire current, NOT a
    current density; this simply changes the output dlav so that it is a
    pathlength, not a differential area, and sets all DALEC elements equal to
    the same electric current, ensuring zero divergence. Default isI=False.
  - type1 is an optional flag to limit which type Bostrom loops are created
  - type2 is an optional flag to limit which type Bostrom loops are created
  - iono is an optional flag to enable/disable ionosphere currents
  - fac is an optional flag to enable/disable field aligned currents
  - equator is an optional flag to enable/disable equatorial currents

  OUTPUTS:
  - rvecs is a 3-tuple/list of NumPy object arrays (phis, thetas, rhos), each
    with the same dimensions as rion_min, with each object holding a 1D array
    of discrete DALEC element position vector components:
     phis[:]   - vector of phi coordinates of DALEC element centers
     thetas[:] - vector of theta coordinates of DALEC element centers
     rhos[:]   - vector of rho coordinates of DALEC element centers
  - Jvecs is a 3-tuple/list of NumPy object arrays (Jphis, Jthetas, Jrhos), each
    with the same dimensions as rion_min, with each object holding a 1D array
    of discrete DALEC element current (density) vector components:
     Jphis[:]   - vector of Jphi components of current (density)
     Jthetas[:] - vector of Jtheta coordinates of current (density)
     Jrhos[:]   - vector of Jrho coordinates of current (density)
  - dvecs is an object array with the same dimensions as rion_min, with each
    object holding a 1D array of discrete DALEC element areas (if isI==False),
    or DALEC element pathlengths (if isI==True).

  """


  """
  NOTES:
    A) rho_min/rho_max usage is admittedly confusing, so here are a few
       use-cases intended to clarify how these inputs should be used:
       1) if rho_min is the ionospheric radius, and rho_max is larger than
          rho_min, all DALECS current segments outside rho_max will be
          discarded...this is probably the most common use-case;
       2) if rho_min and rho_max are equal, the ionospheric current segments
          (and ONLY ionospheric segments) will always be returned...this is
          mostly redundant, since ionospheric current segments are already
          defined by inputs phi_min|max, theta_min|max, and Jion, however it
          may be used to obtain the differential lengths/areas/volumes
          (i.e., dvecs) for each ionospheric cell;
       3) if rho_max is the ionospheric radius, and rho_min is larger than
          rho_max, all DALECS current segments inside rho_min will be
          discarded...this is probably only useful for diagnostic purposes
          (e.g., isolating effects of ionospheric and magnetosperic currents).

    B) all currents between rho_min and rho_max are calculated before they
       are pruned, so these options cannot currently be used to make this
       function run any faster.

    C) This program was written to explicitly model Bostrom current loops,
       and it does this well, if somewhat inefficiently. In fact, if one's
       objective is actually to create (partial) hemispheric grids of FACs,
       it is much more efficient to calculate a numerical divergence of the
       ionospheric current vector field, then map these values up/out along
       magnetic field lines to the equator; this requires only 1 field-line
       mapping per ionospheric element instead of 4.
       With this in mind, consider modifying this program to:
       1) check if divergence of Jion can be calculated numerically (i.e.,
          it must have a dimension equal to at least 2 in the direction a
          divergence is desired, and really 3 for a more robust central
          differenc-based divergence);
       2) check if rho_min and rho_max have identical phi,theta coordinates;
          if they do, calculate divergence using something like the utility
          function _div2d_sph() in MIXCalcs.py to get FAC distribution;
       3) map FACs back up the field lines, scaling current density according
          to Slava Merkin's 2005 notes on the LFM's inner MHD boundary.
       The only obvious downside to this is possible error introduced by the
       numerical diverence calculation, most of which will manifest as a
       smoothing of the FAC field; if the ionospheric currents are themselves
       already a product of numerical gradients, this smoothing may become
       unacceptable.

  REFERENCES:
  Bonnevier et al. (1970), "A three-dimensional model current system for
    polar magnetic substorms",
  Kisabeth & Rostoker (1977), "Modelling of three-dimensional current systems
    associated with magnetospheric substorms",

  """

  #
  # Pre-process inputs
  #

  # start by converting all expected list-like inputs into actual lists;
  # this simplifies input checking, and allows users to pass multi-component
  # inputs as lists, tuples, or even arrays-of-objects
  rion_min = list(rion_min)
  rion_max = list(rion_max)
  Jion = list(Jion)


  # check first required input parameter
  if not(len(rion_min) == 3):
    raise Exception('1st input must contain 3 items')
  elif (np.isscalar(rion_min[0]) and
        np.isscalar(rion_min[1]) and
        np.isscalar(rion_min[2])):
    # convert scalars to 0-rank arrays
    rion_min[0] = np.array(rion_min[0])
    rion_min[1] = np.array(rion_min[1])
    rion_min[2] = np.array(rion_min[2])
  elif not(type(rion_min[0]) is np.ndarray and
           type(rion_min[1]) is np.ndarray and
           type(rion_min[2]) is np.ndarray):
    raise Exception('1st input must hold all scalars or all ndarrays')
  elif not(rion_min[0].shape == rion_min[1].shape and
           rion_min[0].shape == rion_min[2].shape):
    raise Exception('1st input arrays must all have same shape')


  # check second required input parameter
  if not(len(rion_max) == 3):
    raise Exception('2nd input must contain 3 items')
  elif (np.isscalar(rion_max[0]) and
        np.isscalar(rion_max[1]) and
        np.isscalar(rion_max[2])):
    # convert scalars to 0-rank arrays
    rion_max[0] = np.array(rion_max[0])
    rion_max[1] = np.array(rion_max[1])
    rion_max[2] = np.array(rion_max[2])
  elif not(type(rion_max[0]) is np.ndarray and
           type(rion_max[1]) is np.ndarray and
           type(rion_max[2]) is np.ndarray):
    raise Exception('2nd input must hold all scalars or all ndarrays')
  elif not(rion_max[0].shape == rion_max[1].shape and
           rion_max[0].shape == rion_max[2].shape):
    raise Exception('2nd input arrays must all have same shape')


  # check third required input parameter
  if not(len(Jion) == 2):
    raise Exception('3rd input must contain 2 items')
  elif (np.isscalar(Jion[0]) and
        np.isscalar(Jion[1])):
    # convert scalars to 0-rank arrays
    Jion[0] = np.array(Jion[0])
    Jion[1] = np.array(Jion[1])
  elif not(type(Jion[0]) is np.ndarray and
           type(Jion[1]) is np.ndarray):
    raise Exception('3rd input must hold all scalars or all ndarrays')
  elif not(Jion[0].shape == Jion[1].shape):
    raise Exception('3rd input arrays must all have same shape')


  # components of all required inputs must have same dimensions
  # NOTE: this will raise exceptions when comparing scalars, 1xNone
  #       arrays, or 1x1[...x1] arrays; there is no simple way to
  #       avoid this in NumPy/Python -EJR 9/2013
  if not(rion_min[0].shape == rion_max[0].shape and
         rion_min[0].shape == Jion[0].shape):
    raise Exception('All input components must have same shape')


  #
  # Start actual algorithm
  #

  # pre-allocate arrays of objects to hold single temporary DALEC
  rvec = [np.empty((1,), dtype=object),
           np.empty((1,), dtype=object),
           np.empty((1,), dtype=object)]
  Jvec = [np.empty((1,), dtype=object),
           np.empty((1,), dtype=object),
           np.empty((1,), dtype=object)]
  dvec = np.empty((1,), dtype=object)
  
  # pre-allocate arrays of objects to hold all output DALECs
  dims = rion_min[0].shape
  rvecs = [np.empty(dims, dtype=object),
           np.empty(dims, dtype=object),
           np.empty(dims, dtype=object)]
  Jvecs = [np.empty(dims, dtype=object),
           np.empty(dims, dtype=object),
           np.empty(dims, dtype=object)]
  dvecs = np.empty(dims, dtype=object)



  for i in range(np.size(rion_min[0].flat)):

    # copy input elements into temporary variables for each iteration
    phi_min = rion_min[0].flat[i]
    phi_max = rion_max[0].flat[i]
    theta_min = rion_min[1].flat[i]
    theta_max = rion_max[1].flat[i]
    rho_min = rion_min[2].flat[i]
    rho_max = rion_max[2].flat[i]
    Jphi = Jion[0].flat[i]
    Jtheta = Jion[1].flat[i]


    # call _bostromType1() to generate type 1 discretized Bostrom loops
    if type1:
      (phis1, thetas1, rhos1,
       Jphis1, Jthetas1, Jrhos1,
       dl_para1, dl_perp1) = _bostromType1(min(phi_min, phi_max),
                                           max(phi_min, phi_max),
                                           min(theta_min, theta_max),
                                           max(theta_min, theta_max),
                                           min(rho_min, rho_max),
                                           max(rho_min, rho_max),
                                           Jphi, Jtheta, n, isI,
                                           iono=iono, fac=fac,
                                           equator=equator)
    else:
      (phis1, thetas1, rhos1, 
       Jphis1, Jthetas1, Jrhos1,
       dl_para1, dl_perp1) = (np.array([]) for iter in range(8))
       
    # call _bostromType2() to generate type 2 discretized Bostrom loops
    if type2:
      (phis2, thetas2, rhos2,
       Jphis2, Jthetas2, Jrhos2,
       dl_para2, dl_perp2) = _bostromType2(min(phi_min, phi_max),
                                           max(phi_min, phi_max),
                                           min(theta_min, theta_max),
                                           max(theta_min, theta_max),
                                           min(rho_min, rho_max),
                                           max(rho_min, rho_max),
                                           Jphi, Jtheta, n, isI,
                                           iono=iono, fac=fac,
                                           equator=equator)
    else:
      (phis2, thetas2, rhos2, 
       Jphis2, Jthetas2, Jrhos2,
       dl_para2, dl_perp2) = (np.array([]) for iter in range(8))
    
    # combine and trim individual DALECs

    # even though this is only a single DALEC, we still need to place it
    # into an object array for the trim_phi_theta_rho() function to work
    rvec[0][0] = np.concatenate((phis1, phis2))
    rvec[1][0] = np.concatenate((thetas1, thetas2))
    rvec[2][0] = np.concatenate((rhos1, rhos2))
    Jvec[0][0] = np.concatenate((Jphis1, Jphis2))
    Jvec[1][0] = np.concatenate((Jthetas1, Jthetas2))
    Jvec[2][0] = np.concatenate((Jrhos1, Jrhos2))
    dvec = np.empty((1,), dtype=object)
    dvec[0] = np.concatenate(((dl_para1 * dl_perp1),
                            (dl_para2 * dl_perp2)))
    
    # trim DALEC
    rvec, Jvec, dvec = _trim_phi_theta_rho(
      rvec, Jvec, dvec,
      rho_min=rho_min, rho_max=rho_max)
    
    # place trimmed DALEC into output array of DALECs
    rvecs[0].flat[i] = rvec[0][0]
    rvecs[1].flat[i] = rvec[1][0]
    rvecs[2].flat[i] = rvec[2][0]
    Jvecs[0].flat[i] = Jvec[0][0]
    Jvecs[1].flat[i] = Jvec[1][0]
    Jvecs[2].flat[i] = Jvec[2][0]
    dvecs.flat[i] = dvec[0]

  # endfor i in loops
  
  
  
  # If inputs were scalars, do NOT return arrays of objects
  if rvecs[0].shape is ():
    rvecs[0] = rvecs[0].flat[0]
    rvecs[1] = rvecs[1].flat[0]
    rvecs[2] = rvecs[2].flat[0]
    Jvecs[0] = Jvecs[0].flat[0]
    Jvecs[1] = Jvecs[1].flat[0]
    Jvecs[2] = Jvecs[2].flat[0]
    dvecs = dvecs.flat[0]


  return (rvecs, Jvecs, dvecs)



def ralecs_sphere(rion_min, rion_max, Jion, n=10, isI=False,
                  type1=True, type2=True,
                  iono=True, fac=True, outer=True):
  """
  Construct a 3D Radially Aligned Loop Equivalent Current System (RALECS) in
  spherical coordinates given ionospheric current (density) on a 2D spherical
  shell. Each RALEC contains a pair of discretized loops: the "type 1" loop is
  purely zonal in the ionosphere; the "type 2" loop is purely meridional. When
  the current reaches the edge of its cell, it flows out along a radial "field
  line", and returns along the radial "field line" at the opposite edge. The
  "field lines" are discretized in log10-space in order to place more elements
  near the ionosphere. The loops close at a radius 100x the ionospheric radius.

  While this function has a similar interface to dalecs_sphere(), it is not
  especially suited for "real" physics It is rather provided to help simulate
  some of the more cartoon-ish models of ionosphere-magnetosphere coupling.

  INPUTS:
  - rion_min is a 3-tuple/list of scalars or same-shaped ND arrays that hold:
     phi_min   - minimum longitude (0 points to sun) of ionospheric current
     theta_min - minimum colatitude of ionospheric current
     rho_min   - minimum radius of returned currents
                 ** if rho_min is larger than rho_max, rho_max is taken
                    as the ionospheric radius, and rho_min sets the min
                    radius for returned RALEC elements, possibly reducing
                    the number of elements
  - rion_max is a 3-tuple/list of scalars or same-shaped ND arrays that hold:
     phi_max   - maximum longitude (0 points to sun) of ionospheric current
     theta_max - maximum colatitude of ionospheric current
     rho_max   - maximum radius of returned currents
                 ** if rho_max is larger than rho_min, rho_min is taken
                    as the ionospheric radius, and rho_max sets the max
                    radius for returned RALEC elements, possibly reducing
                    the number of elements
  - Jion is a 2-tuple/list of scalars or same-shaped ND arrays that hold:
     Jphi      - ionospheric current (density) in phi direction
     Jtheta    - ionospheric current (density) in theta direction
  - n is optional number of elements used to discretize each FAC; with
    4 FACs per RALEC, 2 ionospheric sheets, and 2 distant sheets, the
    total number of elements per RALEC is (4*n+4). Default n=10.
  - isI is optional flag to treat (jtheta,jphi) as a wire current, NOT a
    current density; this simply changes the output dlav so that it is a
    pathlength, not a differential area, and sets all DALEC elements equal to
    the same electric current, ensuring zero divergence. Default isI=False.
  - type1 is an optional flag to limit which type RALEC loops are created
  - type2 is an optional flag to limit which type RALEC loops are created
  - iono is an optional flag to enable/disable ionosphere currents
  - fac is an optional flag to enable/disable field aligned currents
  - outer is an optional flag to enable/disable outer sphere currents


  OUTPUTS:
  - rvecs is a 3-tuple/list of NumPy object arrays (phis, thetas, rhos), each
    with the same dimensions as rion_min, with each object holding a 1D array
    of discrete RALEC element position vector components:
     phis[:]   - vector of phi coordinates of RALEC element centers
     thetas[:] - vector of theta coordinates of RALEC element centers
     rhos[:]   - vector of rho coordinates of RALEC element centers
  - Jvecs is a 3-tuple/list of NumPy object arrays (Jphis, Jthetas, Jrhos), each
    with the same dimensions as rion_min, with each object holding a 1D array
    of discrete RALEC element current (density) vector components:
     Jphis[:]   - vector of Jphi components of current (density)
     Jthetas[:] - vector of Jtheta coordinates of current (density)
     Jrhos[:]   - vector of Jrho coordinates of current (density)
  - dvecs is an object array with the same dimensions as rion_min, with each
    object holding a 1D array of discrete RALEC element areas (if isI==False),
    or RALEC element pathlengths (if isI==True).

  """


  """
  NOTES:
    A) rho_min/rho_max usage is admittedly confusing, so here are a few
       use-cases intended to clarify how these inputs should be used:
       1) if rho_min is the ionospheric radius, and rho_max is larger than
          rho_min, all RALEC current segments outside rho_max will be
          discarded...this is probably the most common use-case;
       2) if rho_min and rho_max are equal, the ionospheric current segments
          (and ONLY ionospheric segments) will always be returned...this is
          mostly redundant, since ionospheric current segments are already
          defined by inputs phi_min|max, theta_min|max, and Jion, however it
          may be used to obtain the differential lengths/areas/volumes
          (i.e., dvecs) for each ionospheric cell;
       3) if rho_max is the ionospheric radius, and rho_min is larger than
          rho_max, all RALEC current segments inside rho_min will be
          discarded...this is probably only useful for diagnostic purposes
          (e.g., isolating effects of ionospheric and magnetosperic currents).

    B) all currents between rho_min and rho_max are calculated before they
       are pruned, so these options cannot currently be used to make this
       function run any faster.
  """

  #
  # Pre-process inputs
  #

  # start by converting all expected list-like inputs into actual lists;
  # this simplifies input checking, and allows users to pass multi-component
  # inputs as lists, tuples, or even arrays-of-objects
  rion_min = list(rion_min)
  rion_max = list(rion_max)
  Jion = list(Jion)


  # check first required input parameter
  if not(len(rion_min) == 3):
    raise Exception('1st input must contain 3 items')
  elif (np.isscalar(rion_min[0]) and
        np.isscalar(rion_min[1]) and
        np.isscalar(rion_min[2])):
    # convert scalars to 0-rank arrays
    rion_min[0] = np.array(rion_min[0])
    rion_min[1] = np.array(rion_min[1])
    rion_min[2] = np.array(rion_min[2])
  elif not(type(rion_min[0]) is np.ndarray and
           type(rion_min[1]) is np.ndarray and
           type(rion_min[2]) is np.ndarray):
    raise Exception('1st input must hold all scalars or all ndarrays')
  elif not(rion_min[0].shape == rion_min[1].shape and
           rion_min[0].shape == rion_min[2].shape):
    raise Exception('1st input arrays must all have same shape')


  # check second required input parameter
  if not(len(rion_max) == 3):
    raise Exception('2nd input must contain 3 items')
  elif (np.isscalar(rion_max[0]) and
        np.isscalar(rion_max[1]) and
        np.isscalar(rion_max[2])):
    # convert scalars to 0-rank arrays
    rion_max[0] = np.array(rion_max[0])
    rion_max[1] = np.array(rion_max[1])
    rion_max[2] = np.array(rion_max[2])
  elif not(type(rion_max[0]) is np.ndarray and
           type(rion_max[1]) is np.ndarray and
           type(rion_max[2]) is np.ndarray):
    raise Exception('2nd input must hold all scalars or all ndarrays')
  elif not(rion_max[0].shape == rion_max[1].shape and
           rion_max[0].shape == rion_max[2].shape):
    raise Exception('2nd input arrays must all have same shape')


  # check third required input parameter
  if not(len(Jion) == 2):
    raise Exception('3rd input must contain 2 items')
  elif (np.isscalar(Jion[0]) and
        np.isscalar(Jion[1])):
    # convert scalars to 0-rank arrays
    Jion[0] = np.array(Jion[0])
    Jion[1] = np.array(Jion[1])
  elif not(type(Jion[0]) is np.ndarray and
           type(Jion[1]) is np.ndarray):
    raise Exception('3rd input must hold all scalars or all ndarrays')
  elif not(Jion[0].shape == Jion[1].shape):
    raise Exception('3rd input arrays must all have same shape')


  # components of all required inputs must have same dimensions
  # NOTE: this will raise exceptions when comparing scalars, 1xNone
  #       arrays, or 1x1[...x1] arrays; there is no simple way to
  #       avoid this in NumPy/Python -EJR 9/2013
  if not(rion_min[0].shape == rion_max[0].shape and
         rion_min[0].shape == Jion[0].shape):
    raise Exception('All input components must have same shape')


  #
  # Start actual algorithm
  #

  # pre-allocate arrays of objects to hold single temporary RALEC
  rvec = [np.empty((1,), dtype=object),
           np.empty((1,), dtype=object),
           np.empty((1,), dtype=object)]
  Jvec = [np.empty((1,), dtype=object),
           np.empty((1,), dtype=object),
           np.empty((1,), dtype=object)]
  dvec = np.empty((1,), dtype=object)
  
  # pre-allocate arrays of objects to hold all output RALECs
  dims = rion_min[0].shape
  rvecs = [np.empty(dims, dtype=object),
           np.empty(dims, dtype=object),
           np.empty(dims, dtype=object)]
  Jvecs = [np.empty(dims, dtype=object),
           np.empty(dims, dtype=object),
           np.empty(dims, dtype=object)]
  dvecs = np.empty(dims, dtype=object)


  for i in range(np.size(rion_min[0].flat)):

    # copy input elements into temporary variables for each iteration
    phi_min = rion_min[0].flat[i]
    phi_max = rion_max[0].flat[i]
    theta_min = rion_min[1].flat[i]
    theta_max = rion_max[1].flat[i]
    rho_min = rion_min[2].flat[i]
    rho_max = rion_max[2].flat[i]
    Jphi = Jion[0].flat[i]
    Jtheta = Jion[1].flat[i]


    # call _ralecType1() to generate type 1 discretized RALEC loops
    if type1:
      (phis1, thetas1, rhos1,
       Jphis1, Jthetas1, Jrhos1,
       dl_para1, dl_perp1) = _ralecType1(min(phi_min, phi_max),
                                         max(phi_min, phi_max),
                                         min(theta_min, theta_max),
                                         max(theta_min, theta_max),
                                         min(rho_min, rho_max),
                                         max(rho_min, rho_max),
                                         Jphi, Jtheta, n, isI,
                                         iono=iono, fac=fac,
                                         outer=outer)
    else:
      (phis1, thetas1, rhos1, 
       Jphis1, Jthetas1, Jrhos1,
       dl_para1, dl_perp1) = (np.array([]) for iter in range(8))
       
    # call _ralecType2() to generate type 2 discretized RALEC loops
    if type2:
      (phis2, thetas2, rhos2,
       Jphis2, Jthetas2, Jrhos2,
       dl_para2, dl_perp2) = _ralecType2(min(phi_min, phi_max),
                                         max(phi_min, phi_max),
                                         min(theta_min, theta_max),
                                         max(theta_min, theta_max),
                                         min(rho_min, rho_max),
                                         max(rho_min, rho_max),
                                         Jphi, Jtheta, n, isI,
                                         iono=iono, fac=fac,
                                         outer=outer)
    else:
      (phis2, thetas2, rhos2, 
       Jphis2, Jthetas2, Jrhos2,
       dl_para2, dl_perp2) = (np.array([]) for iter in range(8))

    
    # combine and trim individual RALECs

    # even though this is only a single RALEC, we still need to place it
    # into an object array for the trim_phi_theta_rho() function to work
    rvec[0][0] = np.concatenate((phis1, phis2))
    rvec[1][0] = np.concatenate((thetas1, thetas2))
    rvec[2][0] = np.concatenate((rhos1, rhos2))
    Jvec[0][0] = np.concatenate((Jphis1, Jphis2))
    Jvec[1][0] = np.concatenate((Jthetas1, Jthetas2))
    Jvec[2][0] = np.concatenate((Jrhos1, Jrhos2))
    dvec = np.empty((1,), dtype=object)
    dvec[0] = np.concatenate(((dl_para1 * dl_perp1),
                            (dl_para2 * dl_perp2)))
    
    # trim RALEC
    rvec, Jvec, dvec = _trim_phi_theta_rho(
      rvec, Jvec, dvec,
      rho_min=rho_min, rho_max=rho_max)
    
    # place trimmed RALEC into output array of RALECs
    rvecs[0].flat[i] = rvec[0][0]
    rvecs[1].flat[i] = rvec[1][0]
    rvecs[2].flat[i] = rvec[2][0]
    Jvecs[0].flat[i] = Jvec[0][0]
    Jvecs[1].flat[i] = Jvec[1][0]
    Jvecs[2].flat[i] = Jvec[2][0]
    dvecs.flat[i] = dvec[0]

  # endfor i in loops
  
  
  
  # If inputs were scalars, do NOT return arrays of objects
  if rvecs[0].shape is ():
    rvecs[0] = rvecs[0].flat[0]
    rvecs[1] = rvecs[1].flat[0]
    rvecs[2] = rvecs[2].flat[0]
    Jvecs[0] = Jvecs[0].flat[0]
    Jvecs[1] = Jvecs[1].flat[0]
    Jvecs[2] = Jvecs[2].flat[0]
    dvecs = dvecs.flat[0]


  return (rvecs, Jvecs, dvecs)



def _edgeGrid(rcenter):
   """
   Generates meshgrid of cell boundaries (i.e., rmin and rmax position vectors)
   given meshgrid of cell centers. It assumes cell boundaries are all adjacent,
   and that spacing for outer-most rows/columns is regular.
   """

   # rcenter must be list-like object
   if not (type(rcenter) is list or
           type(rcenter) is tuple or
           (type(rcenter) is np.ndarray and rcenter.dtype == object)):
      raise Exception('Input must be list-like')
   else:
      # make it a list to lessen future difficulty
      rcenter = list(rcenter)


   nd = len(rcenter)

   if nd > 3:
      raise Exception('Too many dimensions in input meshgrid')
   elif nd > 2:
      raise Exception('3D meshgrids not implemented yet')
   elif nd > 1:
      #raise Exception, '2D meshgrids not implemented yet'
      
      # unwrap rcenter (np.unwrap() assumes values are in radians)
      unwrap_0 = np.unwrap(rcenter[0], axis=0)
      unwrap_1 = np.unwrap(rcenter[1], axis=1)
      
      # calculate grid cell boundaries
      rmin = [None] * nd # initialize empty 3 list
      rmin[0] = np.zeros(rcenter[0].shape)
      rmin[0][1:,:] = rcenter[0][1:,:] - np.diff(unwrap_0, axis=0)/2.
      rmin[0][0,:] = rcenter[0][0,:] - np.diff(unwrap_0[0:2,:], axis=0).squeeze()/2.

      rmin[1] = np.zeros(rcenter[1].shape)
      rmin[1][:,1:] = rcenter[1][:,1:] - np.diff(unwrap_1, axis=1)/2.
      rmin[1][:,0] = rcenter[1][:,0] - np.diff(unwrap_1[:,0:2], axis=1).squeeze()/2.

      rmax = [None] * nd # initialize empty 3 list
      rmax[0] = np.zeros(rcenter[0].shape)
      rmax[0][:-1,:] = rcenter[0][:-1,:] + np.diff(unwrap_0, axis=0)/2.
      rmax[0][-1,:] = rcenter[0][-1,:] + np.diff(unwrap_0[-2:,:], axis=0).squeeze()/2.
      
      rmax[1] = np.zeros(rcenter[1].shape)
      rmax[1][:,:-1] = rcenter[1][:,:-1] + np.diff(unwrap_1, axis=1)/2.
      rmax[1][:,-1] = rcenter[1][:,-1] + np.diff(unwrap_1[:,-2:], axis=1).squeeze()/2.
      
   else:
      # raise Exception, '1D meshgrids not implemented yet'

      # unwrap rcenter (np.unwrap() assumes values are in radians)
      unwrap_0 = np.unwrap(rcenter[0], axis=0)
      
      # calculate grid cell boundaries
      rmin = [None] * nd # initialize empty 3 list
      rmin[0] = np.zeros(rcenter[0].shape)
      rmin[0][1:] = rcenter[0][1:] - np.diff(unwrap_0, axis=0)/2.
      rmin[0][0] = rcenter[0][0] - np.diff(unwrap_0[0:2], axis=0).squeeze()/2.

      rmax = [None] * nd # initialize empty 3 list
      rmax[0] = np.zeros(rcenter[0].shape)
      rmax[0][:-1] = rcenter[0][:-1] + np.diff(unwrap_0, axis=0)/2.
      rmax[0][-1] = rcenter[0][-1] + np.diff(unwrap_0[-2:], axis=0).squeeze()/2.

   return rmin,rmax



def _trim_phi_theta_rho(rvecs, Jvecs, dvecs,
                        phi_min=None, phi_max=None,
                        theta_min=None, theta_max=None,
                        rho_min=None, rho_max=None):
   """
   Remove unwanted DALEC/RALEC elements by filtering on rvecs.
   NOTE: Even though we assume phi and theta are angles, there is no simple
         way to know, when a _min or _max is None, what the default periodic
         range should be (e.g., is phi {0:2pi} or {-pi:pi}?). So, if _min is
         None, assume there is no minimum value, or min=-infinity. Similarly,
         if _max is None, assume there is no max, or max=infinity. If the
         user specifies a _min or _max, we assume they know what they want.
   NOTE: Comparisons are made after recasting rvecs to the same data type as
         *_min/*_max. This way the user can force a comparison at a lower
         precision (e.g., 32-bit float), which can be helpful given the 
         loss of precision that tends to occur with lots of trigonometric
         operations common with spherical coordinates.
   """
   if phi_min is None:
      phi_min = -np.inf
   if phi_max is None:
      phi_max = np.inf
   
   if theta_min is None:
      theta_min = -np.inf
   if theta_max is None:
      theta_max = np.inf
   
   if rho_min is None:
      rho_min = 0.
   if rho_max is None:
      rho_max = np.inf
   
   
   # loop over each horizontal current element
   for i in range(np.size(dvecs.flat)):
            
      if phi_min <= phi_max:
        # if min<=max, return all between min and max
        good_phi = (np.greater_equal(rvecs[0].flat[i], phi_min, 
                                     dtype=type(phi_min)) &
                    np.less_equal(rvecs[0].flat[i], phi_max, 
                                  dtype=type(phi_max)))

      else:
        # if max<min, return all outside of min-max range
        good_phi = (np.greater_equal(rvecs[0].flat[i], phi_min, 
                                     dtype=type(phi_min)) |
                    np.less_equal(rvecs[0].flat[i], phi_max, 
                                  dtype=type(phi_max)))
      
      if theta_min <= theta_max:
        # if min<=max, return all between min and max
        good_theta = (np.greater_equal(rvecs[1].flat[i], theta_min,
                                       dtype=type(theta_min)) &
                      np.less_equal(rvecs[1].flat[i], theta_max,
                                    dtype=type(theta_max)))

      else:
        # if max<min, return all outside of min-max range
        good_theta = (np.greater_equal(rvecs[1].flat[i], theta_min,
                                       dtype=type(theta_min)) |
                      np.less_equal(rvecs[1].flat[i], theta_max,
                                    dtype=type(theta_max)))
      
      if rho_min <= rho_max:
        # if min<=max, return all between min and max
        good_rho = (np.greater_equal(rvecs[2].flat[i], rho_min,
                                     dtype=type(rho_min)) &
                    np.less_equal(rvecs[2].flat[i], rho_max,
                                  dtype=type(rho_max)))
      else:
        # if max<min, return all outside of min-max range
        good_rho = (np.greater_equal(rvecs[2].flat[i], rho_min,
                                     dtype=type(rho_min)) |
                    np.less_equal(rvecs[2].flat[i], rho_max,
                                  dtype=type(rho_max)))

      
      good = good_phi & good_theta & good_rho
      
      rvecs[0].flat[i] = rvecs[0].flat[i][good]
      rvecs[1].flat[i] = rvecs[1].flat[i][good]
      rvecs[2].flat[i] = rvecs[2].flat[i][good]
    
      Jvecs[0].flat[i] = Jvecs[0].flat[i][good]
      Jvecs[1].flat[i] = Jvecs[1].flat[i][good]
      Jvecs[2].flat[i] = Jvecs[2].flat[i][good]
     
      dvecs.flat[i] = dvecs.flat[i][good]
      
   return rvecs, Jvecs, dvecs



def _bostromType1(phi_min, phi_max,
                  theta_min, theta_max,
                  rho_min, rho_max,
                  Jphi, Jtheta,
                  n, isI,
                  iono=True, fac=True, equator=True):
  """
  Discretize a Type 1 Bostrom current loop, and return results in 1D arrays.
  """
  #
  # Logic: create a discretized current loop using linear segments in dipole
  #        coordinates, then converting these into current (density) segments
  #        in spherical coordinates according to Swisdak (2006), "Notes on
  #        the Dipole Coordinate System", arXiv:physics/0606044v1.
  #

  #
  # Derive some simple quantities needed later
  # 
  dphi = (phi_max-phi_min)
  dtheta=(theta_max-theta_min)
  phi_avg = (phi_max + phi_min) / 2
  theta_avg = (theta_max + theta_min) / 2

  #
  # Convert ionospheric current density into current
  #
  if isI:
    Iphi = Jphi
  else:
    Iphi = Jphi * (rho_min * dtheta) # phi component

  #
  # Initialize arrays to hold data for discrete Bostrom loop elements
  #
  qs = np.full(2*n+2, np.nan)
  ps = np.full(2*n+2, np.nan)
  phis = np.full(2*n+2, np.nan)
  thetas = np.full(2*n+2, np.nan)
  rhos = np.full(2*n+2, np.nan)
  dl_perp = np.full(2*n+2, np.nan)
  dl_para = np.full(2*n+2, np.nan)
  Iqs = np.full(2*n+2, np.nan)
  Ips = np.full(2*n+2, np.nan)
  Iphis = np.full(2*n+2, np.nan)
  Ithetas = np.full(2*n+2, np.nan)
  Irhos = np.full(2*n+2, np.nan)

  #
  # Caclulate ionospheric components in spherical coordinates
  #
  if iono:
    phis[0] = phi_avg
    thetas[0] = theta_avg
    rhos[0] = rho_min
    dl_para[0] = rhos[0] * np.sin(thetas[0]) * dphi
    if isI:
      dl_perp[0] = 1
    else:
      dl_perp[0] = rhos[0] * dtheta
    Iphis[0] = Jphi * dl_perp[0]
    Ithetas[0] = 0
    Irhos[0] = 0


  #
  # Generate position vectors for centers of current (density) elements
  #

  # # ionospheric element
  # if iono:
  #   qs[0] = np.cos((theta_max+theta_min)/2.) / rho_min**2.
  #   ps[0] = rho_min / np.sin((theta_max+theta_min)/2.)**2.
  #   phis[0] = (phi_max+phi_min)/2.

  # equatorial element
  if equator:
    qs[1*n+1] = 0
    ps[1*n+1] = rho_min / np.sin(theta_avg)**2.
    phis[1*n+1] = phi_avg

  if fac:
    # E FAC elements
    q_avg = np.cos(theta_avg) / rho_min**2.
    qs[0*n+1:1*n+1] = (np.linspace(q_avg, 0, n+1)[:-1] +
                       np.linspace(q_avg, 0, n+1)[1:]) / 2.
    ps[0*n+1:1*n+1] = rho_min / np.sin(theta_avg)**2.
    phis[0*n+1:1*n+1] = phi_max

    # W FAC elements
    q_avg = np.cos(theta_avg) / rho_min**2.
    qs[1*n+2:2*n+2] = (np.linspace(0, q_avg, n+1)[:-1] +
                        np.linspace(0, q_avg, n+1)[1:]) / 2.
    ps[1*n+2:2*n+2] = rho_min / np.sin(theta_avg)**2.
    phis[1*n+2:2*n+2] = phi_min

  # convert qs and ps to rhos and thetas...phis remain unchanged
  ignore = np.isnan(qs)
  (rhos[~ignore], thetas[~ignore]) = _dp2sp_pos(qs[~ignore], ps[~ignore])


  #
  # Calculate pathlengths perpendicular to current density element at elements'
  # positions; this is a cross-section that defines the current *sheet* density
  #
  if isI:
    # don't waste cpu cycles if current density is not requested
    dl_perp[:] = 1.
  else:

    # if iono:
    #   # ionospheric element
    #   dl_perp[0] = rho_min * (theta_max-theta_min)

    dp = rho_min / np.sin(theta_min)**2 - rho_min / np.sin(theta_max)**2.
    
    if equator:
      # equatorial element
      dl_perp[1*n+1] = (dp * np.sin(thetas[1*n+1])**3. /
                        np.sqrt(1. + 3.*np.cos(thetas[1*n+1])**2.) )

    if fac:
      # E FAC elements
      dl_perp[0*n+1:1*n+1] = (dp * np.sin(thetas[0*n+1:1*n+1])**3. /
                              np.sqrt(1. + 3.*np.cos(thetas[0*n+1:1*n+1])**2.) )

      # W FAC elements
      dl_perp[1*n+2:2*n+2] = (dp * np.sin(thetas[1*n+2:2*n+2])**3. /
                              np.sqrt(1. + 3.*np.cos(thetas[1*n+2:2*n+2])**2.) )

  # set Inf values to NaN, and treat as missing data in subsequent processing
  dl_perp[np.isinf(dl_perp)] = np.nan


  #
  # Calculate pathlengths parallel to current (density) element at elements'
  # positions; this is a path along which a line integral would be calculated
  # in, for example, the Biot-Savart equations.
  #


  # if iono:
  #   # ionospheric element
  #   dl_para[0] = rho_min * np.sin((theta_max+theta_min)/2.) * dphi

  if equator:
    # equatorial element
    dl_para[1*n+1] = dphi * rhos[1*n+1]  * np.sin(thetas[1*n+1])

  if fac:
    # E FAC elements
    dqE = qs[1] - qs[2]
    dl_para[0*n+1:1*n+1] = (dqE * rhos[0*n+1:1*n+1]**3. /
                            np.sqrt(1. + 3.*np.cos(thetas[0*n+1:1*n+1])**2.) )

    # W FAC elements
    dqW = qs[1*n+3] - qs[1*n+2]
    dl_para[1*n+2:2*n+2] = (dqW * rhos[1*n+2:2*n+2]**3. /
                            np.sqrt(1. + 3.*np.cos(thetas[1*n+2:2*n+2])**2.) )


  # set Inf values to NaN, and treat as missing data in subsequent processing.
  dl_para[np.isinf(dl_para)] = np.nan


  #
  # Generate current vectors in dipole coordinates
  #

  # # ionospheric element
  # Iqs[0] = 0
  # Ips[0] = 0
  # Iphis[0] = Iphi

  # equatorial element
  Iqs[1*n+1] = 0
  Ips[1*n+1] = 0
  Iphis[1*n+1] = -Iphi

  # E FAC elements
  Iqs[0*n+1:1*n+1] = -Iphi
  Ips[0*n+1:1*n+1] = 0
  Iphis[0*n+1:1*n+1] = 0

  # W FAC elements
  Iqs[1*n+2:2*n+2] = Iphi
  Ips[1*n+2:2*n+2] = 0
  Iphis[1*n+2:2*n+2] = 0

  # #
  # # Remove NaNs prior to calling _dp2sp_dir (not ones that were Infs)
  # #
  # phis = phis[~remove]
  # thetas = thetas[~remove]
  # rhos = rhos[~remove]
  # Iqs = Iqs[~remove]
  # Ips = Ips[~remove]
  # Iphis = Iphis[~remove]
  # dl_para = dl_para[~remove]
  # dl_perp = dl_perp[~remove]

  #
  # Convert currents in dipole coordinates to currents in spherical
  #
  (Irhos[~ignore],
   Ithetas[~ignore]) = _dp2sp_dir(
    Iqs[~ignore], Ips[~ignore], thetas[~ignore]
  )
  
  #
  # Finally, divide current vectors by dl_perp to create current densities
  #
  Jrhos = Irhos / dl_perp
  Jthetas = Ithetas / dl_perp
  Jphis = Iphis / dl_perp

  # rhos less than rho_min should not be possible in this function, but they
  # do arise due to finite numerical precision and lots of trigonometry; they
  # are always just barely less than rho_min, so we will force them to rho_min
  rhos[rhos < rho_min] = rho_min
  
  # trim NaNs
  keep = ~np.isnan(Jphis) & ~np.isnan(Jthetas) & ~np.isnan(Jrhos)

  return (phis[keep], thetas[keep], rhos[keep], 
          Jphis[keep], Jthetas[keep], Jrhos[keep],
          dl_para[keep], dl_perp[keep])


def _bostromType2(phi_min, phi_max,
                  theta_min, theta_max,
                  rho_min, rho_max,
                  Jphi, Jtheta,
                  n, isI,
                  iono=True, fac=True, equator=True):
  """
  Discretize a Type 2 Bostrom current loop, and return results in 1D arrays.
  """
  #
  # Logic: create a discretized current loop using linear segments in dipole
  #        coordinates, then converting these into current (density) segments
  #        in spherical coordinates according to Swisdak (2006), "Notes on
  #        the Dipole Coordinate System", arXiv:physics/0606044v1.
  #

  #
  # Derive some simple quantities needed later
  # 
  dphi = (phi_max-phi_min)
  dtheta = (theta_max-theta_min)
  phi_avg = (phi_max + phi_min) / 2
  theta_avg = (theta_max + theta_min) / 2

  #
  # Convert ionospheric current density into current
  #
  if isI:
    Itheta = Jtheta
  else:
    Itheta = Jtheta * (rho_min * np.sin(theta_avg) * dphi)

  #
  # Initialize arrays to hold data for discrete Bostrom loop elements
  #
  qs = np.full(2*n+2, np.nan)
  ps = np.full(2*n+2, np.nan)
  phis = np.full(2*n+2, np.nan)
  thetas = np.full(2*n+2, np.nan)
  rhos = np.full(2*n+2, np.nan)
  dl_perp = np.full(2*n+2, np.nan)
  dl_para = np.full(2*n+2, np.nan)
  Iqs = np.full(2*n+2, np.nan)
  Ips = np.full(2*n+2, np.nan)
  Iphis = np.full(2*n+2, np.nan)
  Ithetas = np.full(2*n+2, np.nan)
  Irhos = np.full(2*n+2, np.nan)


  #
  # Caclulate ionospheric components in spherical coordinates
  #
  if iono:
    phis[0] = phi_avg
    thetas[0] = theta_avg
    rhos[0] = rho_min
    dl_para[0] = rhos[0] * dtheta
    if isI:
      dl_perp[0] = 1
    else:
      dl_perp[0] = rhos[0] * np.sin(theta_avg) * dphi
    Iphis[0] = 0
    Ithetas[0] = Jtheta * dl_perp[0]
    Irhos[0] = 0


  #
  # First, generate position vectors for centers of current (density) elements
  #

  # # ionospheric element
  # if iono:
  #   qs[0] = np.cos((theta_max+theta_min)/2.) / rho_min**2.
  #   ps[0] = rho_min / np.sin((theta_max+theta_min)/2.)**2.
  #   phis[0] = (phi_max+phi_min)/2.

  # equatorial element
  if equator:
    qs[1*n+1] = 0
    ps[1*n+1] = rho_min / np.sin(theta_avg)**2.
    phis[1*n+1] = phi_avg

  if fac:
    # S FAC elements
    q_max = np.cos(theta_max) / rho_min**2.
    qs[0*n+1:1*n+1] = (np.linspace(q_max, 0, n+1)[:-1] +
                        np.linspace(q_max, 0, n+1)[1:]) / 2.

    ps [0*n+1:1*n+1] = rho_min / np.sin(theta_max)**2.
    phis [0*n+1:1*n+1] = phi_avg

    # N FAC elements
    q_min = np.cos(theta_min) / rho_min**2.
    qs[1*n+2:2*n+2] = (np.linspace(0, q_min, n+1)[:-1] +
                        np.linspace(0, q_min, n+1)[1:]) / 2.

    ps[1*n+2:2*n+2] = rho_min / np.sin(theta_min)**2.
    phis[1*n+2:2*n+2] = phi_avg



  # convert qs and ps to rhos and thetas...phis remain unchanged
  ignore = np.isnan(qs)
  (rhos[~ignore], thetas[~ignore]) = _dp2sp_pos(qs[~ignore], ps[~ignore])


  #
  # Calculate pathlengths perpendicular to current density element at
  # elements' positions; this is the cross-section that defines the current
  # *sheet* density
  #
  if isI:

    # don't waste cpu cycles if current density is not requested
    dl_perp[:] = 1.

  else:

    # if iono:
    #   # ionospheric element
    #   dl_perp[0] = rho_min * np.sin((theta_max+theta_min)/2.) * dphi

    if equator:
      # equatorial element
      dl_perp[1*n+1] = dphi * rhos[1*n+1] * np.sin(thetas[1*n+1])

    if fac:
      # S FAC elements
      dl_perp[0*n+1:1*n+1] = dphi * rhos[0*n+1:1*n+1] * np.sin(thetas[0*n+1:1*n+1])

      # N FAC elements
      dl_perp[1*n+2:2*n+2] = dphi * rhos[1*n+2:2*n+2] * np.sin(thetas[1*n+2:2*n+2])


  # for now, just set all Inf values to NaN, adn treat as missing data in
  # any subsequent processing
  dl_perp[np.isinf(dl_perp)] = np.nan



  #
  # Next, calculate pathlengths parallel to current (density) element at
  # elements' positions;
  #


  # if iono:
  #   # ionospheric element
  #   dl_para[0] = rho_min * dtheta
  
  dp = rho_min / np.sin(theta_min)**2 - rho_min / np.sin(theta_max)**2.

  if equator:
    # equatorial element
    dl_para[1*n+1] = (dp * np.sin(thetas[1*n+1])**3. /
                      np.sqrt(1. + 3.*np.cos(thetas[1*n+1])**2.) )

  if fac:
    # S FAC elements
    dqS = qs[0*n+1] - qs[0*n+2]
    dl_para[0*n+1:1*n+1] = (dqS * rhos[0*n+1:1*n+1]**3. /
                            np.sqrt(1. + 3.*np.cos(thetas[0*n+1:1*n+1])**2.) )

    # N FAC elements
    dqN = qs[1*n+3] - qs[1*n+2]
    dl_para[1*n+2:2*n+2] = (dqN * rhos[1*n+2:2*n+2]**3. /
                            np.sqrt(1. + 3.*np.cos(thetas[1*n+2:2*n+2])**2.) )


  # For now, just set all Inf values to NaN, and treat as missing data in
  # any subsequent processing.
  dl_para[np.isinf(dl_para)] = np.nan


  #
  # Next, generate current vectors in dipole coordinates
  #



  # # NOTE: in nature, ionospheric current is NOT thought to flow on constant
  # #       q or p, but rather it has components in both "directions". However,
  # #       FACs do flow along lines of constant p, and equatorial currents flow
  # #       along a line of constant q (i.e., q=0), We force this onto our
  # #       current loops, which should transform into a purely theta current.
  # #       This calculation is not actually required. -EJR

  # # ionospheric element
  # Iqs[0] = -np.sin((theta_max+theta_min)/2.) / np.sqrt(1. + 3.*np.cos((theta_max+theta_min)/2.)**2.) * Itheta
  # Ips[0] = -2*np.cos((theta_max+theta_min)/2.) / np.sqrt(1. + 3.*np.cos((theta_max+theta_min)/2.)**2.) * Itheta
  # Iphis[0] = 0

  # equatorial element
  Iqs[1*n+1] = 0
  Ips[1*n+1] = Itheta
  Iphis[1*n+1] = 0

  # S FAC elements
  Iqs[0*n+1:1*n+1] = -Itheta
  Ips[0*n+1:1*n+1] = 0
  Iphis[0*n+1:1*n+1] = 0

  # N FAC elements
  Iqs[1*n+2:2*n+2] = Itheta
  Ips[1*n+2:2*n+2] = 0
  Iphis[1*n+2:2*n+2] = 0

  # #
  # # Remove NaNs prior to calling _dp2sp_dir (not ones that were Infs)
  # #
  # phis = phis[~remove]
  # thetas = thetas[~remove]
  # rhos = rhos[~remove]
  # Iqs = Iqs[~remove]
  # Ips = Ips[~remove]
  # Iphis = Iphis[~remove]
  # dl_para = dl_para[~remove]
  # dl_perp = dl_perp[~remove]

  #
  # Convert currents in dipole coordinates to currents in spherical
  #
  (Irhos[~ignore],
   Ithetas[~ignore]) = _dp2sp_dir(
    Iqs[~ignore], Ips[~ignore], thetas[~ignore]
  )
  
  #
  # Finally, divide current vectors by dl_perp to create current densities
  #
  Jrhos = Irhos / dl_perp
  Jthetas = Ithetas / dl_perp
  Jphis = Iphis / dl_perp
    
  # rhos less than rho_min should not be possible in this function, but they
  # do arise due to finite numerical precision and lots of trigonometry; they
  # are always just barely less than rho_min, so we will force them to rho_min
  rhos[rhos < rho_min] = rho_min

  # trim NaNs
  keep = ~np.isnan(Jphis) & ~np.isnan(Jthetas) & ~np.isnan(Jrhos)
  
  return (phis[keep], thetas[keep], rhos[keep], 
          Jphis[keep], Jthetas[keep], Jrhos[keep],
          dl_para[keep], dl_perp[keep])



def _ralecType1(phi_min, phi_max,
                theta_min, theta_max,
                rho_min, rho_max,
                Jphi, Jtheta,
                n, isI):
  """
  Discretize a Type 1 RALEC current loop, and return results in 1D arrays.
  """

  # initialize arrays to hold data for discrete Bostrom loop elements
  rhos = np.zeros(2*n+2)
  thetas = np.zeros(2*n+2)
  phis = np.zeros(2*n+2)
  dl_perp = np.zeros(2*n+2)
  dl_para = np.zeros(2*n+2)
  Irhos = np.zeros(2*n+2)
  Ithetas = np.zeros(2*n+2)
  Iphis = np.zeros(2*n+2)

  #
  # First, generate position vectors for centers of current (density) elements
  #

  # ionospheric element
  rhos[0] = rho_min
  thetas[0] = (theta_max+theta_min)/2.
  phis[0] = (phi_max+phi_min)/2.

  # quasi-infinite element
  rhos[1*n+1] = 10**2 * rho_min
  thetas[1*n+1] = (theta_max+theta_min)/2.
  phis[1*n+1] = (phi_max+phi_min)/2.
  
  # E FAC elements (place more elements closer to Earth)
  rhos[0*n+1:1*n+1] = ((10**(np.linspace(0, 1, n+1))**2)[:-1] +
                       (10**(np.linspace(0, 1, n+1))**2)[1:]) / 2. * rho_min
  thetas[0*n+1:1*n+1] = (theta_max+theta_min)/2.
  phis[0*n+1:1*n+1] = phi_max

  # W FAC elements (place more elements closer to Earth)
  rhos[1*n+2:2*n+2] = ((10**(np.linspace(1, 0, n+1))**2)[:-1] +
                       (10**(np.linspace(1, 0, n+1))**2)[1:]) / 2. * rho_min

  thetas[1*n+2:2*n+2] = (theta_max+theta_min)/2.
  phis[1*n+2:2*n+2] = phi_min


  #
  # Next, calculate pathlengths perpendicular to current density element at
  # elements' positions; this is the cross-section that defines the current
  # *sheet* density
  #

  if isI:

     # don't waste cpu cycles if current density is not requested
     dl_perp[:] = 1.

  else:

     # ionospheric element
     dl_perp[0] = np.abs(rho_min * (theta_max-theta_min))

     # quasi-infinite element
     dl_perp[1*n+1] = np.abs(rhos[1*n+1] * (theta_max-theta_min))

     # E FAC elements
     dl_perp[0*n+1:1*n+1] = np.abs(rhos[0*n+1:1*n+1] * (theta_max-theta_min))

     # W FAC elements
     dl_perp[1*n+2:2*n+2] = np.abs(rhos[1*n+2:2*n+2] * (theta_max-theta_min))


  # for now, just set all Inf values to NaN, adn treat as missing data in
  # any subsequent processing
  dl_perp[np.isinf(dl_perp)] = np.nan


  #
  # Next, calculate pathlengths parallel to current (density) element at
  # elements' positions; this is the path along which a line integral would
  # be calculated in, for example, the Biot-Savart equations.
  #

  # ionospheric element
  dl_para[0] = np.abs(rho_min * np.sin((theta_max+theta_min)/2.) * (phi_max-phi_min))

  # quasi-infinite element
  dl_para[1*n+1] = np.abs(rhos[1*n+1] * np.sin((theta_max+theta_min)/2.) * (phi_max-phi_min))

  # E FAC elements
  dl_para[0*n+1:1*n+1] = np.abs(np.diff(10**(np.linspace(0, 1, n+1)**2) * rho_min) )

  # W FAC elements
  dl_para[1*n+2:2*n+2] = np.abs(-np.diff(10**(np.linspace(1, 0, n+1)**2) * rho_min) )


  # For now, just set all Inf values to NaN, and treat as missing data in
  # any subsequent processing.
  dl_para[np.isinf(dl_para)] = np.nan


  #
  # Next, generate current vectors in dipole coordinates
  #

  # convert ionospheric current density into simple current

  if isI:
     #Itheta = Jtheta
     Iphi = Jphi
  else:
     #Itheta = Jtheta * (rho_min * np.sin((theta_max+theta_min)/2.) * dphi)
     Iphi = Jphi * (rho_min * (theta_max-theta_min)) # phi component

  # ionospheric element
  Irhos[0] = 0
  Ithetas[0] = 0
  Iphis[0] = Iphi

  # quasi-infinite element
  Irhos[1*n+1] = 0
  Ithetas[1*n+1] = 0
  Iphis[1*n+1] = -Iphi

  # E FAC elements
  Irhos[0*n+1:1*n+1] = Iphi
  Ithetas[0*n+1:1*n+1] = 0
  Iphis[0*n+1:1*n] = 0

  # W FAC elements
  Irhos[1*n+2:2*n+2] = -Iphi
  Ithetas[1*n+2:2*n+2] = 0
  Iphis[1*n+2:2*n+2] = 0


  #
  # Finally, divide current vectors by dl_perp to create current densities
  #
  Jrhos = Irhos / dl_perp
  Jthetas = Ithetas / dl_perp
  Jphis = Iphis / dl_perp


  return (phis, thetas, rhos, Jphis, Jthetas, Jrhos, dl_para, dl_perp)



def _ralecType2(phi_min, phi_max,
                theta_min, theta_max,
                rho_min, rho_max,
                Jphi, Jtheta,
                n, isI):
  """
  Discretize a Type 2 RALEC, and return results in 1D arrays.
  """

  # initialize arrays to hold data for discrete Bostrom loop elements
  rhos = np.zeros(2*n+2)
  thetas = np.zeros(2*n+2)
  phis = np.zeros(2*n+2)
  dl_perp = np.zeros(2*n+2)
  dl_para = np.zeros(2*n+2)
  Irhos = np.zeros(2*n+2)
  Ithetas = np.zeros(2*n+2)
  Iphis = np.zeros(2*n+2)

  #
  # First, generate position vectors for centers of current (density) elements
  #
  # ionospheric element
  rhos[0] = rho_min
  thetas[0] = (theta_max+theta_min)/2.
  phis[0] = (phi_max+phi_min)/2.

  # quasi-infinite element
  rhos[1*n+1] = 10**2 * rho_min
  thetas[1*n+1] = (theta_max+theta_min)/2.
  phis[1*n+1] = (phi_max+phi_min)/2.

  # S FAC elements (place more elements closer to Earth)
  rhos[0*n+1:1*n+1] = ((10**(np.linspace(0, 1, n+1))**2)[:-1] +
                       (10**(np.linspace(0, 1, n+1))**2)[1:]) / 2. * rho_min

  thetas[0*n+1:1*n+1] = theta_max
  phis [0*n+1:1*n+1] = (phi_max+phi_min)/2.

  # N FAC elements (place more elements closer to Earth)
  rhos[1*n+2:2*n+2] = ((10**(np.linspace(1, 0, n+1))**2)[:-1] +
                       (10**(np.linspace(1, 0, n+1))**2)[1:]) / 2. * rho_min

  thetas[1*n+2:2*n+2] = theta_min
  phis[1*n+2:2*n+2] = (phi_max+phi_min)/2.



  #
  # Next, calculate pathlengths perpendicular to current density element at
  # elements' positions;
  #

  if isI:

     # don't waste cpu cycles if current density is not requested
     dl_perp[:] = 1.

  else:

    # ionospheric element
    dl_perp[0] = np.abs(rho_min * np.sin((theta_max+theta_min)/2.) * (phi_max-phi_min))

    # quasi-infinite element
    dl_perp[1*n+1] = np.abs(rhos[1*n+1] * np.sin(thetas[1*n+1]) * (phi_max-phi_min))

    # S FAC elements
    dl_perp[0*n+1:1*n+1] = np.abs(rhos[0*n+1:1*n+1] * np.sin(thetas[0*n+1:1*n+1]) * (phi_max-phi_min))

    # N FAC elements
    dl_perp[1*n+2:2*n+2] = np.abs(rhos[1*n+2:2*n+2] * np.sin(thetas[1*n+2:2*n+2]) * (phi_max-phi_min))


  # for now, just set all Inf values to NaN, adn treat as missing data in
  # any subsequent processing
  dl_perp[np.isinf(dl_perp)] = np.nan



  #
  # Next, calculate pathlengths parallel to current (density) element at
  # elements' positions;
  #

  # ionospheric element
  dl_para[0] = np.abs(rho_min * (theta_max-theta_min))

  # quasi-infinite element
  dl_para[1*n+1] = np.abs(rhos[1*n+1] * (theta_max-theta_min))

  # S FAC elements
  dl_para[0*n+1:1*n+1] = np.abs(np.diff(10**(np.linspace(0, 1, n+1)**2) * rho_min) )

  # N FAC elements
  dl_para[1*n+2:2*n+2] = np.abs(-np.diff(10**(np.linspace(1, 0, n+1)**2) * rho_min) )


  # For now, just set all Inf values to NaN, and treat as missing data in
  # any subsequent processing.
  dl_para[np.isinf(dl_para)] = np.nan


  #
  # Next, generate current vectors in dipole coordinates
  #

  # convert ionospheric current density into simple current

  if isI:
     Itheta = Jtheta
     #Iphi = Jphi
  else:
     Itheta = Jtheta * rho_min * np.sin((theta_max+theta_min)/2.) * (phi_max-phi_min)
     #Iphi = Jphi * (rho_min * dtheta) # phi component

  # ionospheric element
  Irhos[0] = 0
  Ithetas[0] = Itheta
  Iphis[0] = 0

  # quasi-infinite element
  Irhos[1*n+1] = 0
  Ithetas[1*n+1] = -Itheta
  Iphis[1*n+1] = 0

  # S FAC elements
  Irhos[0*n+1:1*n+1] = Itheta
  Ithetas[0*n+1:1*n+1] = 0
  Iphis[0*n+1:1*n+1] = 0

  # N FAC elements
  Irhos[1*n+2:2*n+2] = -Itheta
  Ithetas[1*n+2:2*n+2] = 0
  Iphis[1*n+2:2*n+2] = 0


  #
  # Finally, divide current vectors by dl_perp to create current densities
  #
  Jrhos = Irhos / dl_perp
  Jthetas = Ithetas / dl_perp
  Jphis = Iphis / dl_perp


  return (phis, thetas, rhos, Jphis, Jthetas, Jrhos, dl_para, dl_perp)



def bs_cart(rvecs, Jvecs, dvecs, observs):
   """
   Calculate magnetic perturbation in local Cartesian coordinates, caused by
   currents measured in their own local Cartesian coordinates, using the well-
   known Biot-Savart relationship.


   Inputs
     - rvecs is 3-tuple/list of position vector components, or arrays of position
       vector components, or arrays of objects that reference position vector
       components of current element center(s):
        xs    - x position vector component(s)
        ys    - y position vector component(s)
        zs    - z position vector component(s)
     - Jvecs is 3-tuple/list of current (density) components, or arrays of
       current (density) components, or arrays of objects that reference current
       (density) components at positions specified via rvecs:
        Jxs   - x current (density) component(s)
        Jys   - y current (density) component(s)
        Jzs   - z current (density) component(s)
     - dvecs is a differential length/area/volume, or array of differential
       lengths/areas/volumes, or array of objects that reference differential
       lengths/areas/volumes, depending on if Jvecs describes line currents (A),
       current sheet densities (A/m), or current volume densities (A/m/m)
     - observs is 3-tuple/list or same-shaped ND array of observatory location(s):
        obs_xs   - x components of observatory location(s)
        obs_ys   - y components of observatory location(s)
        obs_zs   - z components of observatory location(s)

   Outputs
     - deltaB is 3-tuple/list or same-shaped ND array of magnetic perturbation(s):
        dBxs  - x components of deltaB at locations specified in observ
        dBys  - y components of deltaB at locations specified in observ
        dBzs  - z components of deltaB at locations specified in observ
   """

   """
   NOTES:
   This function was originally intended only to validate bs_sphere() (e.g., do
   a Biot-Savart integration in Cartesian coordinates, then transform the output
   into spherical coordinates to be compared with output from bs_sphere()). It
   turns out it is significantly more efficient in Python to convert spherical
   coordinates into Cartesian, run this function, and convert the outputs back
   into spherical coordinates. So, we choose to discard the approach inspired
   by Kisabeth and Rostoker (1977), and simply make bs_sphere() a wrapper for
   this function. Someday it may be worth looking into this more closely, since
   our original expectation was that the K&R approach would be more efficient.
   -EJR 10/2013


   REFERENCES:
   Kisabeth & Rostoker (1977), "Modelling of three-dimensional current systems
     associated with magnetospheric substorms",
  """

   # start by converting all expected list-like inputs into lists;
   # this simplifies input checking, and allows users to pass multi-component
   # inputs as lists, tuples, or even arrays-of-objects
   rvecs = list(rvecs)
   Jvecs = list(Jvecs)
   observs = list(observs)


   # check first required input parameter
   if not(len(rvecs) == 3):
      raise Exception('1st input must contain 3 items')
   elif all([np.isscalar(rvecs[i]) for i in range(3)]):
      # convert scalars to 0-rank arrays
      rvecs[0] = np.array(rvecs[0])
      rvecs[1] = np.array(rvecs[1])
      rvecs[2] = np.array(rvecs[2])
   elif not(all([type(rvecs[i]) is np.ndarray for i in range(3)])):
      raise Exception('1st input must hold all scalars or all ndarrays')
   elif not(rvecs[0].shape == rvecs[1].shape and
            rvecs[0].shape == rvecs[2].shape):
      raise Exception('1st input arrays must all have same shape')

   if all([rvecs[i].dtype == object for i in range(3)]):
      # if here, all input arrays are object arrays
      if not(all([rvecs[i].flat[j].ndim == 1 for i in range(3) for j in range(rvecs[i].size) ])):
         raise Exception('1st input object array elements must be 1D')
   elif not(all([rvecs[i].dtype != object for i in range(3)])):
      raise Exception('1st input must contain all contiguous or all object arrays')


   # check second required input parameter
   if not(len(Jvecs) == 3):
      raise Exception('1st input must contain 3 items')
   elif all([np.isscalar(Jvecs[i]) for i in range(3)]):
      # convert scalars to 0-rank arrays
      Jvecs[0] = np.array(Jvecs[0])
      Jvecs[1] = np.array(Jvecs[1])
      Jvecs[2] = np.array(Jvecs[2])
   elif not(all([type(Jvecs[i]) is np.ndarray for i in range(3)])):
      raise Exception('2nd input must hold all scalars or all ndarrays')
   elif not(Jvecs[0].shape == Jvecs[1].shape and
            Jvecs[0].shape == Jvecs[2].shape):
      raise Exception('2nd input arrays must all have same shape')

   if all([Jvecs[i].dtype == object for i in range(3)]):
      # if here, all input arrays are object arrays
      if not(all([Jvecs[i].flat[j].ndim == 1
                  for i in range(3) for j in range(Jvecs[i].size) ])):
         raise Exception('2nd input object array elements must be 1D')
   elif not(all([Jvecs[i].dtype != object for i in range(3)])):
      raise Exception('2nd input must contain all contiguous or all object arrays')


   # check third required input parameter
   if np.isscalar(dvecs):
      # convert scalar to 0-rank array
      dvecs = np.array(dvecs)
   elif not(type(dvecs) is np.ndarray):
      raise Exception('3rd input must be a scalar or ndarray')

   if dvecs.dtype == object:
      if not(all([dvecs.flat[j].ndim == 1 for j in range(dvecs.size)])):
         raise Exception('3rd input object array elemetns must be 1D')


   # all components of 1st three inputs must have identical dimensions
   # NOTE: for object arrays, this only compares the shapes of the object arrays,
   #       NOT any of the arrays referenced by each object, which are allowed to
   #       differ in shape as long as they are 1D
   if not(rvecs[0].shape == Jvecs[0].shape and
          rvecs[0].shape == dvecs.shape):
      raise Exception('All components of first 3 input components must all have same shape')


   # check fourth required input parameter
   if not(len(observs) == 3):
      raise Exception('4th input must contain 3 items')
   elif all([np.isscalar(observs[i]) for i in range(3)]):
      # convert scalars to 0-rank arrays
      observs[0] = np.array(observs[0])
      observs[1] = np.array(observs[1])
      observs[2] = np.array(observs[2])
   elif not(all([type(observs[i]) is np.ndarray for i in range(3)])):
      raise Exception('4th input must hold all scalars or all ndarrays')
   elif not(observs[0].shape == observs[1].shape and
            observs[0].shape == observs[2].shape):
      raise Exception('4th input arrays must all have same shape')



   #
   # Start actual algorithm
   #

   # copy and flatten rvecs, Jvecs, and dvecs into individual component arrays
   # NOTE: this creates contiguous memory blocks for each variable, allowing
   #       NumPy's efficient array vectorization, which should be faster than
   #       any parallelization given adequate memory; this function can still
   #       be parallelized by processing each observatory separately
   # FIXME: weird things happen with concatenate() and object arrays; it would
   #        be much more readable if we could avoid this conditional block
   if rvecs[0].dtype == object:
      xs = np.concatenate(rvecs[0].flatten())
      ys = np.concatenate(rvecs[1].flatten())
      zs = np.concatenate(rvecs[2].flatten())
      Jxs = np.concatenate(Jvecs[0].flatten())
      Jys = np.concatenate(Jvecs[1].flatten())
      Jzs = np.concatenate(Jvecs[2].flatten())
      dlavs = np.concatenate(dvecs.flatten())
   else:
      xs = rvecs[0].flatten()
      ys = rvecs[1].flatten()
      zs = rvecs[2].flatten()
      Jxs = Jvecs[0].flatten()
      Jys = Jvecs[1].flatten()
      Jzs = Jvecs[2].flatten()
      dlavs = dvecs.flatten()


   # copy and flatten observ[*] into individual observatory arrays
   obs_xs = observs[0].flatten()
   obs_ys = observs[1].flatten()
   obs_zs = observs[2].flatten()

   # get number of observatories
   nobs = np.size(obs_xs)


   # pre-allocate output arrays
   # NOTE: these are flattened for now, we will reshape them to match
   #       observs on exit
   dBxs = np.zeros(obs_xs.shape)
   dBys = np.zeros(obs_ys.shape)
   dBzs = np.zeros(obs_zs.shape)


   #############################################################################
   ##
   ## This block is a NumPy vectorized version of the following loop. It was
   ## tested, and generates results identical to those from the following loop.
   ## We commented it out and use the loop because when more than a few hundred
   ## observatories are processed), the memory management overhead slows things
   ## considerably on a Macintosh laptop with 8GB RAM.
   ##
   ## Tested the vectorized code on NCAR's Geyser, which has two terabytes (!)
   ## of RAM, but it is still slower than the following loop. Some day I may
   ## investigate this further, but for now stick with the loop. Some notes and
   ## ideas on possible causes of the unexpected slowdown:
   ##
   ## * Some very basic profiling suggests that the major bottleneck occurs when
   ##   calculating dr3; some moderate speed improvements were achieved when the
   ##   old-school numerical trick of multiplying values 2 or 3 times, rather
   ##   than relying on power() functions (or ^ or ** operators), was used, but
   ##   the loop was still faster.
   ## * NCAR's Geyser is a distributed memory architecture...amazingly, I think
   ##   Geyser might actually be shared memory, according to online docs
   ## * NumPy's "broadcasting" is not as efficient as one might believe; try
   ##   creating full matrices manually (use tile() command(?)).
   ## * Try multiplying matrices in-place using a *= operator...doesn't do much
   ##   when in loop mode, but maybe for large arrays it will be faster.
   ##
   #############################################################################
   ##
   ## # attempt to vectorize over observatories too
   ## dxs = obs_xs.reshape(1,obs_xs.size) - xs.reshape(xs.size,1)
   ## dys = obs_ys.reshape(1,obs_ys.size) - ys.reshape(ys.size,1)
   ## dzs = obs_zs.reshape(1,obs_zs.size) - zs.reshape(zs.size,1)
   ##
   ## #dr3 = np.sqrt(dxs**2 + dys**2 + dzs**2)**3
   ## dr = np.sqrt(dxs*dxs + dys*dys + dzs*dzs)
   ## dr3 = dr * dr * dr
   ##
   ##
   ## dBxs = 1e-7 * np.nansum( (Jys.reshape(Jys.size,1) * dzs -
   ##                          Jzs.reshape(Jzs.size,1) * dys) /
   ##                         dr3 * dlavs.reshape(dlavs.size,1), axis=0).flatten()
   ## dBys = 1e-7 * np.nansum( (Jzs.reshape(Jzs.size,1) * dxs -
   ##                          Jxs.reshape(Jxs.size,1) * dzs) /
   ##                         dr3 * dlavs.reshape(dlavs.size,1), axis=0).flatten()
   ## dBzs = 1e-7 * np.nansum( (Jxs.reshape(Jxs.size,1) * dys -
   ##                          Jys.reshape(Jys.size,1) * dxs) /
   ##                         dr3 * dlavs.reshape(dlavs.size,1), axis=0).flatten()
   ##
   #############################################################################

   # loop over observatories
   for j in range(nobs):


      # get displacement vectors from jth observatory to each current segment
      dxs = obs_xs[j] - xs
      dys = obs_ys[j] - ys
      dzs = obs_zs[j] - zs

      # cube the magnitude of displacment vector
      #dr3 = np.sqrt(dxs**2 + dys**2 + dzs**2)**3
      dr = np.sqrt(dxs*dxs + dys*dys + dzs*dzs)
      dr3 = dr*dr*dr # avoiding powers leads to 2-3x speed improvement

      # integrate to get delta_B
      # NOTE: mu_naught = 4 * pi * 10^-7 Webers/(A*m), which if used here, leads
      #       to a flux density in units of Tesla. Given there is a 4*pi normalizing
      #       constant in the Biot-Savart relationsip, we can simply use 10^-7 to
      #       produce results in Tesla.
      dBxs[j] = 1e-7 * np.nansum( (Jys * dzs - Jzs * dys) / dr3 * dlavs)
      dBys[j] = 1e-7 * np.nansum( (Jzs * dxs - Jxs * dzs) / dr3 * dlavs)
      dBzs[j] = 1e-7 * np.nansum( (Jxs * dys - Jys * dxs) / dr3 * dlavs)


   return (dBxs.reshape(observs[0].shape),
           dBys.reshape(observs[1].shape),
           dBzs.reshape(observs[2].shape))





def bs_sphere(rvecs, Jvecs, dvecs, observs):
   """
   Calculate magnetic perturbation in local spherical coordinates, caused by
   currents measured in their own local sperical coordinates, using well-known
   Biot-Savart relationship.

   Note: inputs are converted into Cartesian coordinates before calling
         bs_cart(), whose results are then converted back into spherical
         coordinates; so, don't use this if your inputs are already in
         Cartesian coordinates.


   Inputs
     - rvecs is 3-tuple/list of position vector components, or arrays of position
       vector components, or arrays of objects that reference position vector
       components of current element center(s):
        phis    - phi position vector component(s)
        thetas  - theta position vector component(s)
        rhos    - rho position vector component(s)
     - Jvecs is 3-tuple/list of current (density) components, or arrays of
       current (density) components, or arrays of objects that reference current
       (density) components at positions specified via rvecs:
        Jphis   - phi current (density) component(s)
        Jthetas - theta current (density) component(s)
        Jrhos   - rho current (density) component(s)
     - dvecs is a differential length/area/volume, or array of differential
       lengths/areas/volumes, or array of objects that reference differential
       lengths/areas/volumes, depending on if Jvecs describes line currents (A),
       current sheet densities (A/m), or current volume densities (A/m/m)
     - observs is 3-tuple/list or same-shaped ND array of observatory location(s):
        obs_phis   - phi components of observatory location(s)
        obs_thetas - theta components of observatory location(s)
        obs_rhos   - rho components of observatory location(s)

   Outputs
     - deltaB is 3-tuple/list or same-shaped ND array of magnetic perturbation(s):
        dBphis  - phi components of deltaB at locations specified in observ
        dBthetas- theta components of deltaB at locations specified in observ
        dBrhos  - rho components of deltaB at locations specified in observ
   """

   """
   NOTES:
   This function was originally implemented as a direct tranlation of Kisabeth
   and Rostoker (1977)'s "matrix" formulation for integrating the Biot-Savart
   equation in local spherical coordinates. It was expected that this would be
   more numerically efficient than: 1) tranforming spherical coordinates into
   Cartesian; 2) integrating B-S equation; and 3) transforming results back
   into spherical coordinates. It turned out to be ~6x slower! It is likely
   that there exists a simple tweak for the K-R algorithm as implemented below
   that will improve performance, but until I can investigate this further, I
   simply wrap bs_cart() here. -EJR 10/2013


   REFERENCES:
   Kisabeth & Rostoker (1977), "Modelling of three-dimensional current systems
     associated with magnetospheric substorms",

   """

   #
   # Pre-process inputs
   #

   # start by converting all expected list-like inputs into lists;
   # this simplifies input checking, and allows users to pass multi-component
   # inputs as lists, tuples, or even arrays-of-objects
   rvecs = list(rvecs)
   Jvecs = list(Jvecs)
   observs = list(observs)


   # check first required input parameter
   if not(len(rvecs) == 3):
      raise Exception('1st input must contain 3 items')
   elif all([np.isscalar(rvecs[i]) for i in range(3)]):
      # convert scalars to 0-rank arrays
      rvecs[0] = np.array(rvecs[0])
      rvecs[1] = np.array(rvecs[1])
      rvecs[2] = np.array(rvecs[2])
   elif not(all([type(rvecs[i]) is np.ndarray for i in range(3)])):
      raise Exception('1st input must hold all scalars or all ndarrays')
   elif not(rvecs[0].shape == rvecs[1].shape and
            rvecs[0].shape == rvecs[2].shape):
      raise Exception('1st input arrays must all have same shape')

   if all([rvecs[i].dtype == object for i in range(3)]):
      # if here, all input arrays are object arrays
      if not(all([rvecs[i].flat[j].ndim == 1 for i in range(3) for j in range(rvecs[i].size) ])):
         raise Exception('1st input object array elements must be 1D')
   elif not(all([rvecs[i].dtype != object for i in range(3)])):
      raise Exception('1st input must contain all contiguous or all object arrays')


   # check second required input parameter
   if not(len(Jvecs) == 3):
      raise Exception('1st input must contain 3 items')
   elif all([np.isscalar(Jvecs[i]) for i in range(3)]):
      # convert scalars to 0-rank arrays
      Jvecs[0] = np.array(Jvecs[0])
      Jvecs[1] = np.array(Jvecs[1])
      Jvecs[2] = np.array(Jvecs[2])
   elif not(all([type(Jvecs[i]) is np.ndarray for i in range(3)])):
      raise Exception('2nd input must hold all scalars or all ndarrays')
   elif not(Jvecs[0].shape == Jvecs[1].shape and
            Jvecs[0].shape == Jvecs[2].shape):
      raise Exception('2nd input arrays must all have same shape')

   if all([Jvecs[i].dtype == object for i in range(3)]):
      # if here, all input arrays are object arrays
      if not(all([Jvecs[i].flat[j].ndim == 1
                  for i in range(3) for j in range(Jvecs[i].size) ])):
         raise Exception('2nd input object array elements must be 1D')
   elif not(all([Jvecs[i].dtype != object for i in range(3)])):
      raise Exception('2nd input must contain all contiguous or all object arrays')


   # check third required input parameter
   if np.isscalar(dvecs):
      # convert scalar to 0-rank array
      dvecs = np.array(dvecs)
   elif not(type(dvecs) is np.ndarray):
      raise Exception('3rd input must be a scalar or ndarray')

   if dvecs.dtype == object:
      if not(all([dvecs.flat[j].ndim == 1 for j in range(dvecs.size)])):
         raise Exception('3rd input object array elemetns must be 1D')


   # all components of 1st three inputs must have identical dimensions
   # NOTE: for object arrays, this only compares the shapes of the object arrays,
   #       NOT any of the arrays referenced by each object, which are allowed to
   #       differ in shape as long as they are 1D
   if not(rvecs[0].shape == Jvecs[0].shape and
          rvecs[0].shape == dvecs.shape):
      raise Exception('All components of first 3 input components must all have same shape')


   # check fourth required input parameter
   if not(len(observs) == 3):
      raise Exception('4th input must contain 3 items')
   elif all([np.isscalar(observs[i]) for i in range(3)]):
      # convert scalars to 0-rank arrays
      observs[0] = np.array(observs[0])
      observs[1] = np.array(observs[1])
      observs[2] = np.array(observs[2])
   elif not(all([type(observs[i]) is np.ndarray for i in range(3)])):
      raise Exception('4th input must hold all scalars or all ndarrays')
   elif not(observs[0].shape == observs[1].shape and
            observs[0].shape == observs[2].shape):
      raise Exception('4th input arrays must all have same shape')




   #
   # Start algorithm
   #


   # copy and flatten rvecs, Jvecs, and dvecs into individual component arrays
   # NOTE: this creates contiguous memory blocks for each variable, allowing
   #       NumPy's efficient array vectorization, which should be faster than
   #       any parallelization given adequate memory; this function can still
   #       be parallelized by processing each observatory separately
   # FIXME: weird things happen with concatenate() and object arrays; it would
   #        be much more readable if we could avoid this conditional block
   if rvecs[0].dtype == object:
      phis = np.concatenate(rvecs[0].flatten())
      thetas = np.concatenate(rvecs[1].flatten())
      rhos = np.concatenate(rvecs[2].flatten())
      Jphis = np.concatenate(Jvecs[0].flatten())
      Jthetas = np.concatenate(Jvecs[1].flatten())
      Jrhos = np.concatenate(Jvecs[2].flatten())
      dlavs = np.concatenate(dvecs.flatten())
   else:
      phis = rvecs[0].flatten()
      thetas = rvecs[1].flatten()
      rhos = rvecs[2].flatten()
      Jphis = Jvecs[0].flatten()
      Jthetas = Jvecs[1].flatten()
      Jrhos = Jvecs[2].flatten()
      dlavs = dvecs.flatten()


   # copy and flatten observ[*] into individual observatory arrays
   obs_phis = observs[0].flatten()
   obs_thetas = observs[1].flatten()
   obs_rhos = observs[2].flatten()

   # get number of observatories
   nobs = np.size(obs_phis)



   #############################################################################
   ##
   ## Wrap bs_cart() after tranforming all input coordinates from spherical into
   ## Cartesian, then transform results back into spherical coordinates
   ##
   #############################################################################

   # geopack_08 doesn't handle vectors at the origin (0,0,0) very well, so we
   # will use the home-built transforms found in this module instead

   # transform input current segment positions into Cartesian coordinates
   #xs, ys, zs, Jxs, Jys, Jzs = pyLTR.transform.SPHtoCAR(phis, thetas, rhos,
   #                                                     Jphis, Jthetas, Jrhos)
   xs, ys, zs = _sp2cart_pos((phis, thetas, rhos))
   Jxs, Jys, Jzs = _sp2cart_dir((Jphis, Jthetas, Jrhos), (phis, thetas, rhos))

   # transform input observatory locations into Cartesian coordinates
   #obs_xs, obs_ys, obs_zs = pyLTR.transform.SPHtoCAR(obs_phis, obs_thetas, obs_rhos)
   obs_xs, obs_ys, obs_zs = _sp2cart_pos((obs_phis, obs_thetas, obs_rhos))

   # call bs_cart()
   dBxs, dBys, dBzs = bs_cart((xs,ys,zs),(Jxs,Jys,Jzs),dlavs,(obs_xs,obs_ys,obs_zs))

   # # transform outputs back into spherical
   # _,_,_, dBphis, dBthetas, dBrhos = pyLTR.transform.CARtoSPH(obs_xs, obs_ys, obs_zs,
   #                                                            dBxs, dBys, dBzs)
   dBphis, dBthetas, dBrhos = _cart2sp_dir((dBxs, dBys, dBzs),
                                           (obs_xs, obs_ys, obs_zs))

   #############################################################################
   ##
   ## K&R-1977 algorithm implemented below...we chose to simply wrap bs_cart()
   ## with appropriate coordinate transformations because it is so much faster.
   ## Leave it intact so that it may be used for testing, or possibly optimized
   ## at a later date.
   ##
   #############################################################################
   ##
   ##
   ## # pre-allocate output arrays
   ## # NOTE: these are flattened for now, we will reshape them to match
   ## #       observs on exit
   ## dBphis = np.zeros(obs_phis.shape)
   ## dBthetas = np.zeros(obs_thetas.shape)
   ## dBrhos = np.zeros(obs_rhos.shape)
   ##
   ##
   ##
   ## # loop over observatories
   ## for j in range(nobs):
   ##
   ##    print '\rObservatory',j+1,' of ',nobs,
   ##    sys.stdout.flush()
   ##
   ##    # rotation matrix elements a_{ij}
   ##    # NOTE: we don't actually create the full matrix, or multiply by it; this
   ##    # is probably somewhat inefficient, but it follows the algorithm described
   ##    # by K&R(1977) exactly...it probably doesn't lose too much efficiency.
   ##    a11 = (np.sin(obs_thetas[j]) * np.sin(thetas) * np.cos(obs_phis[j] - phis) +
   ##           np.cos(obs_thetas[j]) * np.cos(thetas))
   ##    a12 = (np.sin(obs_thetas[j]) * np.cos(thetas) * np.cos(obs_phis[j] - phis) -
   ##           np.cos(obs_thetas[j]) * np.sin(thetas))
   ##    a13 =  np.sin(obs_thetas[j]) * np.sin(obs_phis[j] - phis)
   ##    a21 = (np.cos(obs_thetas[j]) * np.sin(thetas) * np.cos(obs_phis[j] - phis) -
   ##           np.sin(obs_thetas[j]) * np.cos(thetas))
   ##    a22 = (np.cos(obs_thetas[j]) * np.cos(thetas) * np.cos(obs_phis[j] - phis) +
   ##           np.sin(obs_thetas[j]) * np.sin(thetas))
   ##    a23 =  np.cos(obs_thetas[j]) * np.sin(obs_phis[j] - phis)
   ##    a31 = -np.sin(thetas) * np.sin(obs_phis[j] - phis)
   ##    a32 = -np.cos(thetas) * np.sin(obs_phis[j] - phis)
   ##    a33 = np.cos(obs_phis[j] - phis);
   ##
   ##
   ##    # create rotation matrix elements k_{ij}
   ##    invRmag = 1 / np.sqrt(rhos**2 + obs_rhos[j]**2 - 2 * rhos * obs_rhos[j] * a11)
   ##
   ##    # if Rmag==zero, invRmag==Inf...assume magnetic field cancels within the
   ##    # volume, just like at the center of a wire with uniform current
   ##    invRmag[np.isinf(invRmag)] = 0;
   ##
   ##    k11 = invRmag**3 * np.zeros(np.size(a11));
   ##    k12 = invRmag**3 * rhos * a13;
   ##    k13 = invRmag**3 * -rhos * a12;
   ##    k21 = invRmag**3 * obs_rhos[j] * a31;
   ##    k22 = invRmag**3 * (rhos * a23 + obs_rhos[j] * a32);
   ##    k23 = invRmag**3 * (obs_rhos[j] * a33 - rhos * a22);
   ##    k31 = invRmag**3 * -obs_rhos[j] * a21;
   ##    k32 = invRmag**3 * (rhos * a33 - obs_rhos[j] * a22);
   ##    k33 = invRmag**3 * -(obs_rhos[j] * a23 + rhos * a32);
   ##
   ##
   ##    # integrate to get delta_B in local spherical coordinates
   ##    # NOTE: mu_naught = 4 * pi * 10^-7 Webers/(A*m), which if used here, leads
   ##    #       to a flux density in units of Tesla. Given there is a 4*pi normalizing
   ##    #       constant in the Biot-Savart relationsip, we can simply use 10^-7 to
   ##    #       produce results in Tesla.
   ##    dBrhos[j]   = dBrhos[j] +   1e-7 * np.nansum((k11 * Jrhos +
   ##                                                 k12 * Jthetas +
   ##                                                 k13 * Jphis) * dlavs)
   ##    dBthetas[j] = dBthetas[j] + 1e-7 * np.nansum((k21 * Jrhos +
   ##                                                 k22 * Jthetas +
   ##                                                 k23 * Jphis) * dlavs)
   ##    dBphis[j]   = dBphis[j] +   1e-7 * np.nansum((k31 * Jrhos +
   ##                                                 k32 * Jthetas +
   ##                                                 k33 * Jphis) * dlavs)
   ##
   ## # end of 'for j in range(nobs')
   ##
   ## print
   ##
   ## return (dBphis.reshape(observs[0].shape),
   ##         dBthetas.reshape(observs[1].shape),
   ##         dBrhos.reshape(observs[2].shape))
   ##
   #############################################################################


   # reshape dB's to match original observatory inputs(s)
   return (dBphis.reshape(observs[0].shape),
           dBthetas.reshape(observs[1].shape),
           dBrhos.reshape(observs[2].shape))



def _dp2sp_pos(qs, ps):
  """
  This support function transforms position vectors in a dipole coordinate
  system into position vectors in a spherical coordinate system according to
  Swisdak (2006), "Notes on the Dipole Coordinate System". It converts dipole
  coordinates {q,p} to {rho,theta} (phi is unchanged when transforming between
  these coordinates).
  NOTE: this algorithm is a root solver optimized for numerical stability; as
        such, there is numerical error at precisions somewhat less than typical
        double-precision limits (e.g., for q==0, p==2, which corresponds to
        the equatorial crossing of a dipole field line traced to theta=pi/4,
        the answer differs from the expected theta=pi/2 by ~1e-8; I do not
        understand this. -EJR 8/2013).
  """

  # just convert qs and ps to >=1D numpy arrays if they aren't already
  qs = np.atleast_1d(qs)
  ps = np.atleast_1d(ps)

  alpha = 256./27. * qs**2. * ps**4.

  beta = (1 + np.sqrt(1+alpha))**(2./3.)
  gamma = alpha**(1./3.)
  mu = 0.5 * ((beta*beta + beta*gamma + gamma*gamma) / beta )**(3./2.)

  rhos = (4*mu) / ((1+mu) * (1+np.sqrt(2.*mu-1))) * ps
  thetas = np.arcsin(np.sqrt(rhos / ps))

  # account for negative qs (i.e., theta>pi/2)
  thetas[qs<0] = np.pi - thetas[qs<0]

  return (rhos, thetas)


def _dp2sp_dir(vqs, vps, thetas):
  """
  This support function transforms directional vectors in a dipole coordinate
  system into vectors in a spherical coordinate system using relationships from
  Swisdak (2006), "Notes on the Dipole Coordinate System". It converts dipole
  vectors {vq,vp} to {vrho,vtheta} (vphi is unchanged when transforming between
  these coordinates). To do this, the theta position of the directional vector
  must be known, which can be obtained using _dp2sp_pos().
  NOTE: I find it interesting that the rho position of the directional vector
        is not required along with theta, but it is not found in the equations
        Swisdak (2006), and the outputs seem correct. -EJR 8/2013
  """

  ## make this more efficient
  #vrhos = (-2.*np.cos(thetas) / np.sqrt(1.+3.*np.cos(thetas)**2.) * vqs +
  #         np.sin(thetas) / np.sqrt(1.+3.*np.cos(thetas)**2.) * vps)
  #
  #vthetas = (-np.sin(thetas) / np.sqrt(1.+3.*np.cos(thetas)**2.) * vqs -
  #           2.*np.cos(thetas) / np.sqrt(1.+3.*np.cos(thetas)**2.) * vps )

  c_thetas = np.cos(thetas)
  s_thetas = np.sin(thetas)
  delta = np.sqrt(1 + 3 * c_thetas * c_thetas)

  vrhos = (-2. * c_thetas / delta * vqs + s_thetas / delta * vps)
  vthetas = (-s_thetas / delta * vqs - 2. * c_thetas / delta * vps)

  return (vrhos, vthetas)


def _sp2cart_pos(r):
   """
   convert spherical position coordinates to cartesian
   """
   phis, thetas, rhos = r

   xs = rhos * np.sin(thetas) * np.cos(phis)
   ys = rhos * np.sin(thetas) * np.sin(phis)
   zs = rhos * np.cos(thetas)

   return (xs, ys, zs)


def _sp2cart_dir(v, r):
   """
   convert spherical direction vectors to cartesian
  """
   dphis, dthetas, drhos = v
   phis, thetas, rhos = r

   dxs = (np.sin(thetas) * np.cos(phis) * drhos +
          np.cos(thetas) * np.cos(phis) * dthetas -
          np.sin(phis) * dphis)

   dys = (np.sin(thetas) * np.sin(phis) * drhos +
          np.cos(thetas) * np.sin(phis) * dthetas +
          np.cos(phis) * dphis)

   dzs = (np.cos(thetas) * drhos - np.sin(thetas) * dthetas)


   return (dxs, dys, dzs)


def _cart2sp_pos(r):
   """
   convert cartesian position coordinates to spherical
   """
   xs, ys, zs = r

   # use arctan2
   phis = np.arctan2(ys, xs)

   # first determine cylindrical radius, or else we won't be able to
   # get a reasonable theta if/when rsph is zero
   rhos = np.sqrt(xs**2 + ys**2)

   # calculate thetas
   thetas = np.arctan2(rhos,zs)

   # now, calculate spherical radius
   rhos = np.sqrt(rhos**2 + zs**2)


   return (phis, thetas, rhos)


def _cart2sp_dir(v, r):
   """
   convert cartesian direction vectors to cartesian
   """
   dxs, dys, dzs = v
   xs, ys, zs = r

   # first, find position vectors in spherical coordinates
   (phis, thetas, rhos) = _cart2sp_pos((xs, ys, zs))

   drhos = (np.sin(thetas) * np.cos(phis) * dxs +
            np.sin(thetas) * np.sin(phis) * dys +
            np.cos(thetas) * dzs)

   dthetas = (np.cos(thetas) * np.cos(phis) * dxs +
              np.cos(thetas) * np.sin(phis) * dys -
              np.sin(thetas) * dzs)

   dphis = (np.cos(phis) * dys - np.sin(phis) * dxs)


   return (dphis, dthetas, drhos)





###
### Various tests follow; these should work with the nose package, which
### means their name should start with "Test", or "test", followed by an
### underscore
###

def test_line_current_bs_cart():
   """
   bs_cart() approximately reproduces magnetic field from infinite line current
   as determined using Ampere's Law
   """

   import numpy as np

   # generate a really long line along the Z axis
   zs = np.linspace(-100000,100000,200001)
   xs = zs * 0
   ys = zs * 0

   # specify 1-amp current along the Z axis
   Jzs = np.ones(zs.size)
   Jxs = Jzs * 0
   Jys = Jzs * 0

   # specify contiguous differential lengths
   dvecs = np.ones(zs.size)

   # define positions in xy plane at which to "observe" magnetic field
   obs_zs = np.array([0,0,0,0])
   obs_xs = np.array([10,0,-10,0])
   obs_ys = np.array([0,10,0,-10])

   # call bs_cart() to generate magnetic fields at observed locations
   dBxs, dBys, dBzs = bs_cart((xs,ys,zs), (Jxs,Jys,Jzs), dvecs,
                              (obs_xs,obs_ys,obs_zs))

   # use Ampere's law to get analytic expression for infinite line current
   dBtot = (4*np.pi*1e-7 * 1.) / (2*np.pi*10.)
   dBzs_anal = np.array([0,0,0,0])
   dBxs_anal = np.array([0,-dBtot,0,dBtot])
   dBys_anal = np.array([dBtot,0,-dBtot,0])

   # compare BS with analytic results, properly accounting for expected zeros
   np.testing.assert_allclose(dBxs[np.abs(dBxs_anal) >= 1e-20],
                              dBxs_anal[np.abs(dBxs_anal) >= 1e-20],
                              rtol=1e-7, atol=0)
   np.testing.assert_allclose(dBys[np.abs(dBys_anal) >= 1e-20],
                              dBys_anal[np.abs(dBys_anal) >= 1e-20],
                              rtol=1e-7, atol=0)
   np.testing.assert_allclose(dBzs[np.abs(dBzs_anal) >= 1e-20],
                              dBzs_anal[np.abs(dBzs_anal) >= 1e-20],
                              rtol=1e-7, atol=0)

   np.testing.assert_allclose(dBxs[np.abs(dBxs_anal) < 1e-20],
                              dBxs_anal[np.abs(dBxs_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBys[np.abs(dBys_anal) < 1e-20],
                              dBys_anal[np.abs(dBys_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBzs[np.abs(dBzs_anal) < 1e-20],
                              dBzs_anal[np.abs(dBzs_anal) < 1e-20],
                              rtol=0, atol=1e-12)


def test_line_current_bs_sphere():
   """
   bs_sphere() approximately reproduces magnetic field from infinite line current
   as determined using Ampere's Law
   """

   import numpy as np

   # generate a really long line along the Z axis
   rhos = np.abs(np.linspace(-100000,100000,200001))
   phis = rhos * 0
   thetas = np.concatenate((np.tile(np.pi,100000), np.tile(0, 100001)) )

   # specify 1-amp current along the Z axis
   Jrhos = np.concatenate((-np.ones(100000), np.ones(100001)) )
   Jphis = Jrhos * 0
   Jthetas = Jrhos * 0

   # specify contiguous differential lengths
   dvecs = np.ones(rhos.size)

   # define positions in theta=pi/2 plane at which to "observe" magnetic field
   obs_rhos = np.array([10, 10, 10, 10])
   obs_phis = np.array([0, np.pi/2., np.pi, 3.*np.pi/2.])
   obs_thetas = np.array([np.pi/2., np.pi/2., np.pi/2., np.pi/2.])

   # call bs_sphere() to generate magnetic fields at observed locations
   dBphis, dBthetas, dBrhos = bs_sphere((phis,thetas,rhos), (Jphis,Jthetas,Jrhos), dvecs,
                              (obs_phis,obs_thetas,obs_rhos))

   # use Ampere's law to get analytic expression for infinite line current
   dBtot = (4*np.pi*1e-7 * 1.) / (2*np.pi*10.)
   dBrhos_anal = np.array([0,0,0,0])
   dBphis_anal = np.array([dBtot,dBtot,dBtot,dBtot])
   dBthetas_anal = np.array([0,0,0,0])


   # compare BS with analytic results, properly accounting for expected zeros
   np.testing.assert_allclose(dBphis[np.abs(dBphis_anal) >= 1e-20],
                              dBphis_anal[np.abs(dBphis_anal) >= 1e-20],
                              rtol=1e-7, atol=0)
   np.testing.assert_allclose(dBthetas[np.abs(dBthetas_anal) >= 1e-20],
                              dBthetas_anal[np.abs(dBthetas_anal) >= 1e-20],
                              rtol=1e-7, atol=0)
   np.testing.assert_allclose(dBrhos[np.abs(dBrhos_anal) >= 1e-20],
                              dBrhos_anal[np.abs(dBrhos_anal) >= 1e-20],
                              rtol=1e-7, atol=0)

   np.testing.assert_allclose(dBphis[np.abs(dBphis_anal) < 1e-20],
                              dBphis_anal[np.abs(dBphis_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBthetas[np.abs(dBthetas_anal) < 1e-20],
                              dBthetas_anal[np.abs(dBthetas_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBrhos[np.abs(dBrhos_anal) < 1e-20],
                              dBrhos_anal[np.abs(dBrhos_anal) < 1e-20],
                              rtol=0, atol=1e-12)


def test_loop_current_bs_sphere():
   """
   bs_sphere() approximately reproduces magnetic field from a loop current
   as determined using technique at http://www.netdenizen.com/emagnettest/
   """

   import numpy as np
   from scipy.special import ellipk, ellipe

   # generate a loop current in the theta=pi/2 plane
   rhos = np.zeros(100000) + 10
   phis = np.linspace(0, 2.*np.pi, 100001)[:-1]
   thetas = np.zeros(100000) + np.pi/2.

   # specify 1-amp loop current
   Jrhos = np.zeros(100000)
   Jphis = np.ones(100000)
   Jthetas = np.zeros(100000)

   # specify contiguous differential arclengths of loop segments
   dvecs = np.zeros(100000) + 2. * 10 * np.pi / 100000.

   # define positions in theta=pi/2 plane at which to "observe" magnetic field
   obs_rhos = np.linspace(0, 20, 10)
   obs_phis = np.zeros(obs_rhos.size)
   obs_thetas = np.zeros(obs_rhos.size) + np.pi/2.

   # append positions along theta=0 line at which to observe magnetic field
   obs_rhos = np.concatenate((obs_rhos, obs_rhos))
   obs_phis = np.concatenate((obs_phis, obs_phis))
   obs_thetas = np.concatenate((obs_thetas, np.zeros(obs_thetas.size) ))

   # call bs_sphere() to generate magnetic fields at observed locations
   dBphis, dBthetas, dBrhos = bs_sphere((phis,thetas,rhos), (Jphis,Jthetas,Jrhos), dvecs,
                                        (obs_phis,obs_thetas,obs_rhos))

   # use formula(s) from http://www.netdenizen.com/emagnettest/ for "analytic"
   # solutions for loop current (using "" because I'm not sure if elliptical
   # integral functions ellipe() and ellipk() are truly analytic)
   dBrhos_anal = np.zeros(obs_rhos.size)
   dBphis_anal = np.zeros(obs_rhos.size)
   dBthetas_anal = np.zeros(obs_rhos.size)
   for i in range(obs_rhos.size):

      # convert observatory locations to cylindrical coordinates
      xtmp, ytmp, ztmp = _sp2cart_pos((obs_phis[i], obs_thetas[i], obs_rhos[i]))
      rtmp = np.sqrt(xtmp**2. + ytmp**2.)

      # calculate parameters needed for solution
      alpha = rtmp / 10.
      beta = ztmp / 10.
      if rtmp == 0:
         gamma = 0
      else:
         gamma = ztmp / rtmp
      q = ((1+alpha)**2. + beta**2.)
      k = np.sqrt(4*alpha/q)

      # calculate delta B at center of loop
      B0 = (4 * np.pi * 1e-7 * 1) / (2.*10.) * 1.

      # calculate delta B analytically using elliptical integral functions
      dBztmp =  B0 / (np.pi * np.sqrt(q)) * \
         (ellipe(k**2.) * ((1-alpha**2.-beta**2.)/(q-4*alpha)) + ellipk(k**2.))
      dBrtmp = B0 * gamma / (np.pi * np.sqrt(q)) * \
         (ellipe(k**2.) * ((1+alpha**2.+beta**2.)/(q-4*alpha)) - ellipk(k**2.))

      # convert back to spherical coordinates
      dBphis_anal[i], dBthetas_anal[i], dBrhos_anal[i] =  _cart2sp_dir((dBrtmp, 0., dBztmp),
                                                                       (xtmp, ytmp, ztmp))


   # compare BS with analytic results, properly accounting for expected zeros
   np.testing.assert_allclose(dBphis[np.abs(dBphis_anal) >= 1e-20],
                              dBphis_anal[np.abs(dBphis_anal) >= 1e-20],
                              rtol=1e-7, atol=0)
   np.testing.assert_allclose(dBthetas[np.abs(dBthetas_anal) >= 1e-20],
                              dBthetas_anal[np.abs(dBthetas_anal) >= 1e-20],
                              rtol=1e-7, atol=0)
   np.testing.assert_allclose(dBrhos[np.abs(dBrhos_anal) >= 1e-20],
                              dBrhos_anal[np.abs(dBrhos_anal) >= 1e-20],
                              rtol=1e-7, atol=0)

   np.testing.assert_allclose(dBphis[np.abs(dBphis_anal) < 1e-20],
                              dBphis_anal[np.abs(dBphis_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBthetas[np.abs(dBthetas_anal) < 1e-20],
                              dBthetas_anal[np.abs(dBthetas_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBrhos[np.abs(dBrhos_anal) < 1e-20],
                              dBrhos_anal[np.abs(dBrhos_anal) < 1e-20],
                              rtol=0, atol=1e-12)


def test_fukushima_planar():
   """
   Confirm Fukushima's theorum for a planar geometry (i.e., uniformly expanding
   currents, from a point source in a plane, generate a magnetic field that
   cancels out a vertical line current that ends at that point source for all
   points below the plane).
   """

   import numpy as np

   # generate a really long line along the positive Z axis
   rhos = np.linspace(0,100000, 100001)
   phis = rhos * 0
   thetas = rhos * 0

   # specify a 1-amp current along the Zaxis
   Jrhos = -np.ones(rhos.size)
   Jphis = Jrhos * 0
   Jthetas = Jrhos * 0

   # specify contiguous differential lengths
   dvecs = np.ones(rhos.size)

   # generate a really large current sheet, expanding radially from a point
   # source of 1-amp
   # NOTE: exclude the origin, where density is infinite; this is allowed, since
   #       calculated magnetic fields would cancel out everywhere anyway
   meshPhis, meshThetas, meshRhos = np.meshgrid(np.linspace(0,2*np.pi,37)[:-1], # .1 degree longitude bins
                                                np.pi/2., # co-latitude
                                                np.linspace(0,100000, 100001)[1:], # radius, w/o origin
                                                indexing='ij')
   phis = np.concatenate((phis, meshPhis.flatten()))
   thetas = np.concatenate((thetas, meshThetas.flatten()))
   rhos = np.concatenate((rhos, meshRhos.flatten()))

   Jphis = np.concatenate((Jphis, np.zeros(meshPhis.size) ))
   Jthetas = np.concatenate((Jthetas, np.zeros(meshThetas.size) ))

   # calculate current density by dividing by current segment arclength
   Jrhos = np.concatenate((Jrhos, np.tile(1./36., 36*100000) /
                           (2*np.pi/36. * meshRhos.flatten()) ) )

   # dvecs is the area of current segments, from rho-.5 to rho+.5
   dvecs = np.concatenate((dvecs, (np.pi/36. * ((meshRhos.flatten()+.5)**2 -
                                                (meshRhos.flatten()-.5)**2) ) ) )


   # specify a line of observation points that is 1000 meters off the Z-axis,
   # equally distant to azimuthal rays, and which passes through the theta=pi/2
   # plane from top to bottom
   x,y,z = _sp2cart_pos((meshPhis[:2,0,0].mean(), np.pi/2., 1000))

   obs_xs = np.ones(10)*x
   obs_ys = np.ones(10)*y
   obs_zs = np.linspace(5000,-5000,10) # ~500m above/below plane
   obs_phis, obs_thetas, obs_rhos = _cart2sp_pos((obs_xs, obs_ys, obs_zs))

   # call bs_sphere()
   dBphisL, dBthetasL, dBrhosL = bs_sphere((phis[:100001], thetas[:100001], rhos[:100001]),
                                           (Jphis[:100001], Jthetas[:100001], Jrhos[:100001]),
                                           dvecs[:100001],
                                           (obs_phis, obs_thetas, obs_rhos))

   dBphisS, dBthetasS, dBrhosS = bs_sphere((phis[100001:], thetas[100001:], rhos[100001:]),
                                           (Jphis[100001:], Jthetas[100001:], Jrhos[100001:]),
                                           dvecs[100001:],
                                           (obs_phis, obs_thetas, obs_rhos))

   # expected results
   dBphis_anal = np.concatenate((np.tile(-2e-10, 5), np.zeros(5) ))
   dBthetas_anal = np.zeros(10)
   dBrhos_anal = np.zeros(10)


   # compare BS with analytic results, properly accounting for expected zeros
   np.testing.assert_allclose((dBphisL+dBphisS)[np.abs(dBphis_anal) >= 1e-20],
                              dBphis_anal[np.abs(dBphis_anal) >= 1e-20],
                              rtol=1e-3, atol=0)
   np.testing.assert_allclose((dBthetasL+dBthetasS)[np.abs(dBthetas_anal) >= 1e-20],
                              dBthetas_anal[np.abs(dBthetas_anal) >= 1e-20],
                              rtol=1e-3, atol=0)
   np.testing.assert_allclose((dBrhosL+dBrhosS)[np.abs(dBrhos_anal) >= 1e-20],
                              dBrhos_anal[np.abs(dBrhos_anal) >= 1e-20],
                              rtol=1e-3, atol=0)

   np.testing.assert_allclose((dBphisL+dBphisS)[np.abs(dBphis_anal) < 1e-20],
                              dBphis_anal[np.abs(dBphis_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose((dBthetasL+dBthetasS)[np.abs(dBthetas_anal) < 1e-20],
                              dBthetas_anal[np.abs(dBthetas_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose((dBrhosL+dBrhosS)[np.abs(dBrhos_anal) < 1e-20],
                              dBrhos_anal[np.abs(dBrhos_anal) < 1e-20],
                              rtol=0, atol=1e-12)



def test_SECS_DivFree():
   """
   Confirm divergence free spherical elementary current returns expected results
   as taken from Amm & Viljanen (Earth Platets Space, 1999).
   """

   import numpy as np

   # first define <=1 degree grid-boundaries in latitude and longitude
   # (fine grid needed so discretized areas are at least close to correct
   #  when estimating current *density* later on)
   rhos = np.array([6500e3])
   phis = np.linspace(-180,180,360*5+1) * np.pi/180.
   thetas = np.linspace(0,180,180*5+1) * np.pi/180.

   # then find the mid-points...avoids singularities at the poles
   phiGrid, thetaGrid, rhoGrid = np.meshgrid((phis[1:] + phis[:-1])/2,
                                             (thetas[1:] + thetas[:-1])/2,
                                             rhos, indexing='ij')


   # 10000-amp I_0, like A&V-1999
   I0 = 10000.

   # Earth radius so that ionospheric height is 100km, like A&V-1999
   R0 = 6400e3


   # SECS div-free current density for each cell in the grid, like A&V-1999
   JphiGrid = I0 / (4*np.pi*rhoGrid) * 1 / np.tan(thetaGrid/2.)
   JthetaGrid = JphiGrid * 0.


   # # differential areas for each grid cell
   # # (these are approximations, but on a fine enough grid, they are OK)
   dl1 = (rhoGrid * np.sin(thetaGrid) * # length of current sheet elements
          np.tile(phis[1:] - phis[:-1], phiGrid.shape[1]).reshape(phiGrid.shape) )
   dl2 = (rhoGrid * # cross sections of current sheet elements
          np.tile(thetas[1:] - thetas[:-1], thetaGrid.shape[0]).reshape(thetaGrid.shape) )
   dAs =  dl1 * dl2 # area of current sheet element
   # # (these are exact...and lead to almost indistinguisable results)
   # abs_dsintheta = np.abs(np.sin(np.pi/2 - thetas[1:]) -
   #                        np.sin(np.pi/2 - thetas[:-1]))
   # abs_dphi = np.abs(phis[1:] - phis[:-1])
   # dAs = (rhoGrid**2 *
   #        np.atleast_3d(abs_dsintheta.reshape(1,abs_dsintheta.size) *
   #                      abs_dphi.reshape(abs_dphi.size,1) ) )

   # specify a set of virtual observatories at which to sample deltaB
   obs_phis = np.tile(0, 10)
   obs_thetas = np.linspace(1, 5000e3/R0, 10)
   obs_rhos = np.tile(R0, 10)


   # caclulate deltaB using B-S
   dBphi, dBtheta, dBrho = bs_sphere((phiGrid, thetaGrid, rhoGrid),
                                     (JphiGrid, JthetaGrid, np.zeros(JphiGrid.shape)),
                                     dAs,
                                     (obs_phis,obs_thetas,obs_rhos))


   # calculate deltaB analytically (from A&V-1999)
   dBphi_anal = 0. * obs_phis
   dBtheta_anal = (-(1 / 1e7 * I0 / (1 * obs_rhos * np.sin(obs_thetas))) *
                   ((obs_rhos/rhos[0] - np.cos(obs_thetas)) /
                    np.sqrt(1 - 2 * obs_rhos * np.cos(obs_thetas) / rhos[0] +
                            (obs_rhos/rhos[0])**2 ) +
                    np.cos(obs_thetas) ) )
   dBrho_anal = ((1 / 1e7 * I0 / (1 * obs_rhos)) *
                 (1 / np.sqrt(1 - 2 * obs_rhos * np.cos(obs_thetas) / rhos[0] +
                             (obs_rhos/rhos[0])**2 ) - 1) )


   # compare BS with analytic results, properly accounting for expected zeros
   np.testing.assert_allclose(dBphi[np.abs(dBphi_anal) >= 1e-20],
                              dBphi_anal[np.abs(dBphi_anal) >= 1e-20],
                              rtol=1e-3, atol=0)
   np.testing.assert_allclose(dBtheta[np.abs(dBtheta_anal) >= 1e-20],
                              dBtheta_anal[np.abs(dBtheta_anal) >= 1e-20],
                              rtol=1e-3, atol=0)
   np.testing.assert_allclose(dBrho[np.abs(dBrho_anal) >= 1e-20],
                              dBrho_anal[np.abs(dBrho_anal) >= 1e-20],
                              rtol=1e-3, atol=0)

   np.testing.assert_allclose(dBphi[np.abs(dBphi_anal) < 1e-20],
                              dBphi_anal[np.abs(dBphi_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBtheta[np.abs(dBtheta_anal) < 1e-20],
                              dBtheta_anal[np.abs(dBtheta_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBrho[np.abs(dBrho_anal) < 1e-20],
                              dBrho_anal[np.abs(dBrho_anal) < 1e-20],
                              rtol=0, atol=1e-12)



def test_SECS_CurlFree():
   """
   Confirm curl free spherical elementary current returns expected results
   as taken from Amm & Viljanen (Earth Platets Space, 1999).
   """

   import numpy as np

   # First, poloidal currents on the sphere

   # define <=1 degree grid-boundaries in latitude and longitude
   # (fine grid needed so discretized areas are at least close to correct
   #  when estimating current *density* later on)
   rhos = np.array([6500e3])
   phis = np.linspace(-180,180,360*5+1) * np.pi/180.
   thetas = np.linspace(0,180,180*5+1) * np.pi/180.

   # then find the mid-points...avoids singularities at the poles
   phiGrid, thetaGrid, rhoGrid = np.meshgrid((phis[1:] + phis[:-1])/2,
                                             (thetas[1:] + thetas[:-1])/2,
                                             rhos, indexing='ij')


   # 10000-amp I_0, like A&V-1999
   I0 = 10000.

   # Earth radius so that ionospheric height is 100km, like A&V-1999
   R0 = 6400e3


   # SECS curl-free current density for each cell in the grid
   # (actually, this is the superposition of two curl-free SECs, in/out at
   #  opposite poles, thus allowing us to ignore the constant divergence
   #  that normally exists everywhere but the poles when we do the B-S)
   JthetaGrid = (I0 / (4*np.pi*rhoGrid) / np.tan(thetaGrid/2.) +
                 I0 / (4*np.pi*rhoGrid) / np.tan((np.pi - thetaGrid)/2.) )
   JphiGrid = JthetaGrid * 0.


   # # differential areas for each grid cell
   # # (these are approximations, but on a fine enough grid, they are OK)
   dl1 = (rhoGrid * np.sin(thetaGrid) * # length of current sheet elements
          np.tile(phis[1:] - phis[:-1], phiGrid.shape[1]).reshape(phiGrid.shape) )
   dl2 = (rhoGrid * # cross sections of current sheet elements
          np.tile(thetas[1:] - thetas[:-1], thetaGrid.shape[0]).reshape(thetaGrid.shape) )
   dAs =  dl1 * dl2 # area of current sheet element
   # # (these are exact...and lead to almost indistinguisable results)
   # abs_dsintheta = np.abs(np.sin(np.pi/2 - thetas[1:]) -
   #                        np.sin(np.pi/2 - thetas[:-1]))
   # abs_dphi = np.abs(phis[1:] - phis[:-1])
   # dAs = (rhoGrid**2 *
   #        np.atleast_3d(abs_dsintheta.reshape(1,abs_dsintheta.size) *
   #                      abs_dphi.reshape(abs_dphi.size,1) ) )


   # Now, create quasi-infinite line currents into and out of poles
   rhosFAC = np.concatenate((np.linspace(10000000e3, rhos, 100001).reshape(-1,1),
                             np.linspace(rhos, 10000000e3, 100001).reshape(-1,1)))
   dvecs = np.concatenate(( np.abs(np.diff(rhosFAC[:100001], axis=0)),
                            np.abs(np.diff(rhosFAC[100001:], axis=0)) ) )
   rhosFAC = np.concatenate(( (rhosFAC[:100000]+rhosFAC[1:100001])/2,
                              (rhosFAC[100001:-1]+rhosFAC[100002:])/2 ) )

   phisFAC = rhosFAC * 0
   thetasFAC = np.concatenate((np.tile(0, [100000,1]), np.tile(np.pi, [100000,1])))

   # specify current along the Zaxis
   JrhosFAC = np.concatenate((-np.ones((100000,1)) * I0, np.ones((100000,1)) * I0) )
   JphisFAC = JrhosFAC * 0
   JthetasFAC = JrhosFAC * 0



   # specify a set of virtual observatories *outside* the ionosphere at which
   # to sample deltaB
   obs_phis = np.tile(0, 10)
   obs_thetas = np.linspace(np.pi/2 - 5000e3/R0, np.pi/2 + 5000e3/R0, 10)
   obs_rhos = np.tile(R0 * 2., 10)


   # caclulate deltaB using B-S
   dBphiGrid, dBthetaGrid, dBrhoGrid = bs_sphere((phiGrid, thetaGrid, rhoGrid),
                                     (JphiGrid, JthetaGrid, np.zeros(JphiGrid.shape)),
                                     dAs,
                                     (obs_phis,obs_thetas,obs_rhos))
   print(phisFAC.shape)
   print(thetasFAC.shape)
   print(rhosFAC.shape)
   dBphiFAC, dBthetaFAC, dBrhoFAC = bs_sphere((phisFAC, thetasFAC, rhosFAC),
                                     (JphisFAC, JthetasFAC, JrhosFAC),
                                     dvecs,
                                     (obs_phis,obs_thetas,obs_rhos))

   dBphi = dBphiGrid + dBphiFAC
   dBtheta = dBthetaGrid + dBthetaFAC
   dBrho = dBrhoGrid + dBrhoFAC

   # calculate deltaB analytically (from Fukushima-1976)
   # NOTE: this applies Eq. 6 from Fukushima-1976 twice: once for incoming
   #       current, and once for the outgoing current; it does NOT use Eq. 4,
   #       which is wrong (i.e., 1/cos(theta) should be 1/sin(theta)). -EJR
   dBphi_anal = (-I0 / (1e7 * obs_rhos) / np.tan(obs_thetas/2.) -
                  I0 / (1e7 * obs_rhos) / np.tan((np.pi - obs_thetas)/2.) )
   dBtheta_anal = 0. * obs_thetas
   dBrho_anal = 0. * obs_rhos


   # compare BS with analytic results, properly accounting for expected zeros
   np.testing.assert_allclose(dBphi[np.abs(dBphi_anal) >= 1e-20],
                              dBphi_anal[np.abs(dBphi_anal) >= 1e-20],
                              rtol=1e-3, atol=0)
   np.testing.assert_allclose(dBtheta[np.abs(dBtheta_anal) >= 1e-20],
                              dBtheta_anal[np.abs(dBtheta_anal) >= 1e-20],
                              rtol=1e-3, atol=0)
   np.testing.assert_allclose(dBrho[np.abs(dBrho_anal) >= 1e-20],
                              dBrho_anal[np.abs(dBrho_anal) >= 1e-20],
                              rtol=1e-3, atol=0)

   np.testing.assert_allclose(dBphi[np.abs(dBphi_anal) < 1e-20],
                              dBphi_anal[np.abs(dBphi_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBtheta[np.abs(dBtheta_anal) < 1e-20],
                              dBtheta_anal[np.abs(dBtheta_anal) < 1e-20],
                              rtol=0, atol=1e-12)
   np.testing.assert_allclose(dBrho[np.abs(dBrho_anal) < 1e-20],
                              dBrho_anal[np.abs(dBrho_anal) < 1e-20],
                              rtol=0, atol=1e-12)





def test_Type2_AllLongitude():
  """
  A Type 2 Bostrom loop whose boundaries extend 360 degrees (2*pi) should
  result in a surface perturbation of zero. This is stated by Bonnevier et
  al. (1970), although I don't fully follow their proof. Really, it just
  comes down to Ampere's Law for a toroidal solenoid.
  """
  import numpy as np

  # create Type 2 toroid in 1-degree bins
  phis = np.linspace(0,360,361) * np.pi/180.
  thetas = np.linspace(20,30,11) * np.pi/180.
  rhos = np.array([6500e3, np.Inf])

  rion_min = np.meshgrid(phis[:-1], thetas[:-1], rhos[:-1], indexing='ij')
  rion_max = np.meshgrid(phis[1:], thetas[1:], rhos[1:], indexing='ij')

  Iphi = 0. / 10. + 0. * rion_min[0]
  Itheta = 1e6 / 360. + 0. * rion_min[0]

  # generate grid of dalecs
  (r_B2, J_B2, d_B2) = dalecs_sphere(rion_min, rion_max, (Iphi, Itheta), 24, True)

  # calculate deltaBs for meridional line of virtual observatories
  phi_obs, theta_obs, rho_obs = np.meshgrid(0 * np.pi/180.,
                                            np.linspace(0,np.pi,91),
                                            6378e3,
                                            indexing='ij')
  (dBphi, dBtheta, dBrho) = bs_sphere(r_B2, J_B2, d_B2,
                                      (phi_obs, theta_obs, rho_obs) )

  # all B-field components should be zero
  dBphi_anal = dBphi * 0
  dBtheta_anal = dBtheta * 0
  dBrho_anal = dBrho * 0

  # compare numerical with analytic results, properly accounting for expected zeros
  np.testing.assert_allclose(dBphi[np.abs(dBphi_anal) >= 1e-20],
                             dBphi_anal[np.abs(dBphi_anal) >= 1e-20],
                             rtol=1e-3, atol=0)
  np.testing.assert_allclose(dBtheta[np.abs(dBtheta_anal) >= 1e-20],
                             dBtheta_anal[np.abs(dBtheta_anal) >= 1e-20],
                             rtol=1e-3, atol=0)
  np.testing.assert_allclose(dBrho[np.abs(dBrho_anal) >= 1e-20],
                             dBrho_anal[np.abs(dBrho_anal) >= 1e-20],
                             rtol=1e-3, atol=0)

  np.testing.assert_allclose(dBphi[np.abs(dBphi_anal) < 1e-20],
                             dBphi_anal[np.abs(dBphi_anal) < 1e-20],
                             rtol=0, atol=1e-9)
  np.testing.assert_allclose(dBtheta[np.abs(dBtheta_anal) < 1e-20],
                             dBtheta_anal[np.abs(dBtheta_anal) < 1e-20],
                             rtol=0, atol=1e-9)
  np.testing.assert_allclose(dBrho[np.abs(dBrho_anal) < 1e-20],
                             dBrho_anal[np.abs(dBrho_anal) < 1e-20],
                             rtol=0, atol=1e-9)
