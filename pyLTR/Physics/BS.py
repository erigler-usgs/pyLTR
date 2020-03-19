import numpy as np

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
      if not(all([rvecs[i].flat[j].ndim == 1
                  for j in range(rvecs[i].size) for i in range(3)])):
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
                  for j in range(Jvecs[i].size) for i in range(3)])):
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
   rhosFAC = np.concatenate((np.linspace(10000000e3, rhos, 100001),
                             np.linspace(rhos, 10000000e3, 100001)) )
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
