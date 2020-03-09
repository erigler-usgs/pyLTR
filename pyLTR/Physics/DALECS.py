import numpy as np

def dalecs_sphere(rion_min, rion_max, Jion, n=10, isI=False):
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
          rho_min, all SSECS current segments outside rho_max will be
          discarded...this is probably the most common use-case;
       2) if rho_min and rho_max are equal, the ionospheric current segments
          (and ONLY ionospheric segments) will always be returned...this is
          mostly redundant, since ionospheric current segments are already
          defined by inputs phi_min|max, theta_min|max, and Jion, however it
          may be used to obtain the differential lengths/areas/volumes
          (i.e., dvecs) for each ionospheric cell;
       3) if rho_max is the ionospheric radius, and rho_min is larger than
          rho_max, all SSECS current segments inside rho_min will be
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

  # pre-allocate arrays of objects to hold 1D output vectors
  dims = rion_min[0].shape
  rvecs = [np.zeros(dims, dtype=object),
           np.zeros(dims, dtype=object),
           np.zeros(dims, dtype=object)]
  Jvecs = [np.zeros(dims, dtype=object),
           np.zeros(dims, dtype=object),
           np.zeros(dims, dtype=object)]
  dvecs = np.zeros(dims, dtype=object)



  for i in range(np.size(rion_min[0].flat)):

    # copy input elements into temporary variables for each iteration,
    # enforcing proper min/max sort orders where appropriate
    phi_min = min(rion_min[0].flat[i], rion_max[0].flat[i])
    phi_max = max(rion_min[0].flat[i], rion_max[0].flat[i])
    theta_min = min(rion_min[1].flat[i], rion_max[1].flat[i])
    theta_max = max(rion_min[1].flat[i], rion_max[1].flat[i])
    rho_min = min(rion_min[2].flat[i], rion_max[2].flat[i])
    rho_max = max(rion_min[2].flat[i], rion_max[2].flat[i])
    Jphi = Jion[0].flat[i]
    Jtheta = Jion[1].flat[i]


    # this boolean tracks whether the original rho_min was less than
    # the original rho_max, which allows filtering currents *below*
    # the original rho_min instead of *above* the original rho_max
    minLTEmax = rion_min[2].flat[i] <= rion_max[2].flat[i]


    # call _bostromType1() to generate type 1 discretized Bostrom loops
    (phis1, thetas1, rhos1,
     Jphis1, Jthetas1, Jrhos1,
     dl_para1, dl_perp1) = _bostromType1(phi_min, phi_max,
                                         theta_min, theta_max,
                                         rho_min, rho_max,
                                         Jphi, Jtheta, n, isI)

    # call _bostromType2() to generate type 2 discretized Bostrom loops
    (phis2, thetas2, rhos2,
     Jphis2, Jthetas2, Jrhos2,
     dl_para2, dl_perp2) = _bostromType2(phi_min, phi_max,
                                         theta_min, theta_max,
                                         rho_min, rho_max,
                                         Jphi, Jtheta, n, isI)


    # copy local variables into output array, removing undesired elements
    # NOTE: numerical precision can be an issue here, so we apply a kludge...
    #       basically, a tolerance equal to 10x double-precision machine epsilon
    #       is used, ensuring(?) that when rho_range[0] and rho_range[1] are
    #       the same, this function *will* return the ionospheric currents,
    #       but nothing else. -EJR 9/2013
    if minLTEmax:
      good1 = np.logical_not((rhos1 / rho_max) > (1. + 10 * np.finfo(np.float64).eps))
      good2 = np.logical_not((rhos2 / rho_max) > (1. + 10 * np.finfo(np.float64).eps))
    else:
      good1 = np.logical_not((rhos1 / rho_max) < (1. + 10 * np.finfo(np.float64).eps))
      good2 = np.logical_not((rhos2 / rho_max) < (1. + 10 * np.finfo(np.float64).eps))

    rvecs[0].flat[i] = np.concatenate((phis1[good1],
                                       phis2[good2]))
    rvecs[1].flat[i] = np.concatenate((thetas1[good1],
                                       thetas2[good2]))
    rvecs[2].flat[i] = np.concatenate((rhos1[good1],
                                       rhos2[good2]))
    Jvecs[0].flat[i] = np.concatenate((Jphis1[good1],
                                       Jphis2[good2]))
    Jvecs[1].flat[i] = np.concatenate((Jthetas1[good1],
                                       Jthetas2[good2]))
    Jvecs[2].flat[i] = np.concatenate((Jrhos1[good1],
                                       Jrhos2[good2]))
    dvecs.flat[i] = np.concatenate(((dl_para1 * dl_perp1)[good1],
                                    (dl_para2 * dl_perp2)[good2]))

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


def ralecs_sphere(rion_min, rion_max, Jion, n=10, isI=False):
  """
  Construct a 3D Radially Aligned Loop Equivalent Current System (RALECS) in
  spherical coordinates given ionospheric current (density) on a 2D spherical
  shell. Each RALEC contains a pair of discretized loops: the "type 1" loop is
  purely zonal in the ionosphere; the "type 2" loop is purely meridional. When
  the current reaches the edge of its cell, it flows out along a radial "field
  line", and returns along the radial "field line" at the opposite edge. The
  "field lines" are discretized in log10-space in order to place more elements
  near the ionosphere. The loops close at a radius 10x the ionospheric radius.

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


  OUTPUTS:
  - rvecs is a 3-tuple/list of NumPy object arrays (phis, thetas, rhos), each
    with the same dimensions as rion_min, with each object holding a 1D array
    of discrete RALEC element position vector components:
     phis[:]   - vector of phi coordinates of RALEC element centers
     thetas[:] - vector of theta coordinates of RALEC element centers
     rhos[:]   - vector of rho coordinates of RALEC element centers
  - Jvecs is a 3-tuple/list of NumPy object arrays (Jphis, Jthetas, Jrhos), each
    with the same dimensions as rion_min, with each object holding a 1D array
    of discrete DALEC element current (density) vector components:
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

  # pre-allocate arrays of objects to hold 1D output vectors
  dims = rion_min[0].shape
  rvecs = [np.zeros(dims, dtype=object),
           np.zeros(dims, dtype=object),
           np.zeros(dims, dtype=object)]
  Jvecs = [np.zeros(dims, dtype=object),
           np.zeros(dims, dtype=object),
           np.zeros(dims, dtype=object)]
  dvecs = np.zeros(dims, dtype=object)



  for i in range(np.size(rion_min[0].flat)):

    # copy input elements into temporary variables for each iteration,
    # enforcing proper min/max sort orders where appropriate
    phi_min = min(rion_min[0].flat[i], rion_max[0].flat[i])
    phi_max = max(rion_min[0].flat[i], rion_max[0].flat[i])
    theta_min = min(rion_min[1].flat[i], rion_max[1].flat[i])
    theta_max = max(rion_min[1].flat[i], rion_max[1].flat[i])
    rho_min = min(rion_min[2].flat[i], rion_max[2].flat[i])
    rho_max = max(rion_min[2].flat[i], rion_max[2].flat[i])
    Jphi = Jion[0].flat[i]
    Jtheta = Jion[1].flat[i]


    # this boolean tracks whether the original rho_min was less than
    # the original rho_max, which allows filtering currents *below*
    # the original rho_min instead of *above* the original rho_max
    minLTEmax = rion_min[2].flat[i] <= rion_max[2].flat[i]



    # call _ralecType1() to generate type 1 discretized ralec loops
    (phis1, thetas1, rhos1,
     Jphis1, Jthetas1, Jrhos1,
     dl_para1, dl_perp1) = _ralecType1(phi_min, phi_max,
                                         theta_min, theta_max,
                                         rho_min, rho_max,
                                         Jphi, Jtheta, n, isI)

    # call _ralecType2() to generate type 2 discretized ralec loops
    (phis2, thetas2, rhos2,
     Jphis2, Jthetas2, Jrhos2,
     dl_para2, dl_perp2) = _ralecType2(phi_min, phi_max,
                                         theta_min, theta_max,
                                         rho_min, rho_max,
                                         Jphi, Jtheta, n, isI)


    # copy local variables into output array, removing undesired elements
    # NOTE: numerical precision can be an issue here, so we apply a kludge...
    #       basically, a tolerance equal to 10x double-precision machine epsilon
    #       is used, ensuring(?) that when rho_range[0] and rho_range[1] are
    #       the same, this function *will* return the ionospheric currents,
    #       but nothing else. -EJR 9/2013
    if minLTEmax:
      good1 = np.logical_not((rhos1 / rho_max) > (1. + 10 * np.finfo(np.float64).eps))
      good2 = np.logical_not((rhos2 / rho_max) > (1. + 10 * np.finfo(np.float64).eps))
    else:
      good1 = np.logical_not((rhos1 / rho_max) < (1. + 10 * np.finfo(np.float64).eps))
      good2 = np.logical_not((rhos2 / rho_max) < (1. + 10 * np.finfo(np.float64).eps))

    rvecs[0].flat[i] = np.concatenate((phis1[good1],
                                       phis2[good2]))
    rvecs[1].flat[i] = np.concatenate((thetas1[good1],
                                       thetas2[good2]))
    rvecs[2].flat[i] = np.concatenate((rhos1[good1],
                                       rhos2[good2]))
    Jvecs[0].flat[i] = np.concatenate((Jphis1[good1],
                                       Jphis2[good2]))
    Jvecs[1].flat[i] = np.concatenate((Jthetas1[good1],
                                       Jthetas2[good2]))
    Jvecs[2].flat[i] = np.concatenate((Jrhos1[good1],
                                       Jrhos2[good2]))
    dvecs.flat[i] = np.concatenate(((dl_para1 * dl_perp1)[good1],
                                    (dl_para2 * dl_perp2)[good2]))

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



def _bostromType1(phi_min, phi_max,
                  theta_min, theta_max,
                  rho_min, rho_max,
                  Jphi, Jtheta,
                  n, isI):
  """
  Discretize a Type 1 Bostrom current loop, and return results in 1D arrays.
  """

  #
  # Logic: create a discretized current loop using linear segments in dipole
  #        coordinates, then converting these into current (density) segments
  #        in spherical coordinates according to Swisdak (2006), "Notes on
  #        the Dipole Coordinate System", arXiv:physics/0606044v1.
  #

  if rho_min == rho_max:
    # bypass any dipole coordinate stuff if it's going to be trimmed later
    phis = np.atleast_1d((phi_min + phi_max) / 2.)
    thetas = np.atleast_1d((theta_min + theta_max) / 2.)
    rhos = np.atleast_1d(rho_min)
    Jphis = np.atleast_1d(Jphi)
    Jthetas = np.atleast_1d(0.)
    Jrhos = np.atleast_1d(0.)
    dl_para = np.atleast_1d(rhos * np.sin(thetas) * (phi_max - phi_min))
    if isI:
      dl_perp = np.atleast_1d(1.)
    else:
      dl_perp = np.atleast_1d(rhos * (theta_max - theta_min))

  else:
    
    # initialize arrays to hold data for discrete Bostrom loop elements
    qs = np.zeros(2*n+2)
    ps = np.zeros(2*n+2)
    phis = np.zeros(2*n+2)
    dl_perp = np.zeros(2*n+2)
    dl_para = np.zeros(2*n+2)
    Iqs = np.zeros(2*n+2)
    Ips = np.zeros(2*n+2)
    Iphis = np.zeros(2*n+2)

    #
    # First, generate position vectors for centers of current (density) elements
    #

    # ionospheric element
    qs[0] = np.cos((theta_max+theta_min)/2.) / rho_min**2.
    ps[0] = rho_min / np.sin((theta_max+theta_min)/2.)**2.
    phis[0] = (phi_max+phi_min)/2.

    # equatorial element
    qs[1*n+1] = 0
    ps[1*n+1] = rho_min / np.sin((theta_max+theta_min)/2.)**2.
    phis[1*n+1] = (phi_max+phi_min)/2.

    # E FAC elements
    q_avg = np.cos((theta_min + theta_max) / 2.) / rho_min**2.
    qs[0*n+1:1*n+1] = (np.linspace(q_avg, 0, n+1)[:-1] +
                       np.linspace(q_avg, 0, n+1)[1:]) / 2.
    ps[0*n+1:1*n+1] = rho_min / np.sin((theta_max+theta_min)/2.)**2.
    phis[0*n+1:1*n+1] = phi_max

    # W FAC elements
    q_avg = np.cos((theta_min + theta_max) / 2.) / rho_min**2.
    qs[1*n+2:2*n+2] = (np.linspace(0, q_avg, n+1)[:-1] +
                       np.linspace(0, q_avg, n+1)[1:]) / 2.
    ps[1*n+2:2*n+2] = rho_min / np.sin((theta_max+theta_min)/2.)**2.
    phis[1*n+2:2*n+2] = phi_min

    # convert qs and ps to rhos and thetas...phis remain unchanged
    (rhos, thetas) = _dp2sp_pos(qs, ps)


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
      dl_perp[0] = rho_min * (theta_max-theta_min)

      # equatorial element
      dp = rho_min / np.sin(theta_min)**2 - rho_min / np.sin(theta_max)**2.
      dl_perp[1*n+1] = (dp * np.sin(thetas[1*n+1])**3. /
                        np.sqrt(1. + 3.*np.cos(thetas[1*n+1])**2.) )

      # E FAC elements
      dl_perp[0*n+1:1*n+1] = (dp * np.sin(thetas[0*n+1:1*n+1])**3. /
                              np.sqrt(1. + 3.*np.cos(thetas[0*n+1:1*n+1])**2.) )

      # W FAC elements
      dl_perp[1*n+2:2*n+2] = (dp * np.sin(thetas[1*n+2:2*n+2])**3. /
                              np.sqrt(1. + 3.*np.cos(thetas[1*n+2:2*n+2])**2.) )


    # for now, just set all Inf values to NaN, adn treat as missing data in
    # any subsequent processing
    dl_perp[np.isinf(dl_perp)] = np.nan


    #
    # Next, calculate pathlengths parallel to current (density) element at
    # elements' positions; this is the path along which a line integral would
    # be calculated in, for example, the Biot-Savart equations.
    #

    dphi = (phi_max-phi_min)
    dtheta=(theta_max-theta_min)

    # ionospheric element
    dl_para[0] = rho_min * np.sin((theta_max+theta_min)/2.) * dphi

    # equatorial element
    dl_para[1*n+1] = dphi * rhos[1*n+1]  * np.sin(thetas[1*n+1])

    # E FAC elements
    dqE = qs[1] - qs[2]
    dl_para[0*n+1:1*n+1] = (dqE * rhos[0*n+1:1*n+1]**3. /
                            np.sqrt(1. + 3.*np.cos(thetas[0*n+1:1*n+1])**2.) )

    # W FAC elements
    dqW = qs[1*n+3] - qs[1*n+2]
    dl_para[1*n+2:2*n+2] = (dqW * rhos[1*n+2:2*n+2]**3. /
                            np.sqrt(1. + 3.*np.cos(thetas[1*n+2:2*n+2])**2.) )


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
      Iphi = Jphi * (rho_min * dtheta) # phi component

    # ionospheric element
    Iqs[0] = 0
    Ips[0] = 0
    Iphis[0] = Iphi

    # equatorial element
    Iqs[1*n+1] = 0
    Ips[1*n+1] = 0
    Iphis[1*n+1] = -Iphi

    # E FAC elements
    Iqs[0*n+1:1*n+1] = -Iphi
    Ips[0*n+1:1*n+1] = 0
    Iphis[0*n+1:1*n] = 0

    # W FAC elements
    Iqs[1*n+2:2*n+2] = Iphi
    Ips[1*n+2:2*n+2] = 0
    Iphis[1*n+2:2*n+2] = 0


    #
    # Next, convert currents in dipole coordinates to currents in spherical
    #
    (Irhos, Ithetas) = _dp2sp_dir(Iqs, Ips, thetas)


    #
    # Finally, divide current vectors by dl_perp to create current densities
    #
    Jrhos = Irhos / dl_perp
    Jthetas = Ithetas / dl_perp
    Jphis = Iphis / dl_perp


  return (phis, thetas, rhos, Jphis, Jthetas, Jrhos, dl_para, dl_perp)


def _bostromType2(phi_min, phi_max,
                  theta_min, theta_max,
                  rho_min, rho_max,
                  Jphi, Jtheta,
                  n, isI):
  """
  Discretize a Type 2 Bostrom current loop, and return results in 1D arrays.
  """

  #
  # Logic: create a discretized current loop using linear segments in dipole
  #        coordinates, then converting these into current (density) segments
  #        in spherical coordinates according to Swisdak (2006), "Notes on
  #        the Dipole Coordinate System", arXiv:physics/0606044v1.
  #

  if rho_min == rho_max:
    # bypass any dipole coordinate stuff if it's going to be trimmed later
    phis = np.atleast_1d((phi_min + phi_max) / 2.)
    thetas = np.atleast_1d((theta_min + theta_max) / 2.)
    rhos = np.atleast_1d(rho_min)
    Jphis = np.atleast_1d(0.)
    Jthetas = np.atleast_1d(Jtheta)
    Jrhos = np.atleast_1d(0.)
    dl_para = np.atleast_1d(rhos * (theta_max - theta_min)  )
    if isI:
      dl_perp = np.atleast_1d(1.)
    else:
      dl_perp = np.atleast_1d(rhos * np.sin(thetas) * (phi_max - phi_min))
      
  else:

    # initialize arrays to hold data for discrete Bostrom loop elements
    qs = np.zeros(2*n+2)
    ps = np.zeros(2*n+2)
    phis = np.zeros(2*n+2)
    dl_perp = np.zeros(2*n+2)
    dl_para = np.zeros(2*n+2)
    Iqs = np.zeros(2*n+2)
    Ips = np.zeros(2*n+2)
    Iphis = np.zeros(2*n+2)

    #
    # First, generate position vectors for centers of current (density) elements
    #
    # ionospheric element
    qs[0] = np.cos((theta_max+theta_min)/2.) / rho_min**2.
    ps[0] = rho_min / np.sin((theta_max+theta_min)/2.)**2.
    phis[0] = (phi_max+phi_min)/2.

    # equatorial element
    qs[1*n+1] = 0
    ps[1*n+1] = rho_min / np.sin((theta_max+theta_min)/2.)**2.
    phis[1*n+1] = (phi_max+phi_min)/2.

    # S FAC elements
    q_max = np.cos(theta_max) / rho_min**2.
    qs[0*n+1:1*n+1] = (np.linspace(q_max, 0, n+1)[:-1] +
                       np.linspace(q_max, 0, n+1)[1:]) / 2.

    ps [0*n+1:1*n+1] = rho_min / np.sin(theta_max)**2.
    phis [0*n+1:1*n+1] = (phi_max+phi_min)/2.

    # N FAC elements
    q_min = np.cos(theta_min) / rho_min**2.
    qs[1*n+2:2*n+2] = (np.linspace(0, q_min, n+1)[:-1] +
                       np.linspace(0, q_min, n+1)[1:]) / 2.

    ps[1*n+2:2*n+2] = rho_min / np.sin(theta_min)**2.
    phis[1*n+2:2*n+2] = (phi_max+phi_min)/2.



    # convert qs and ps to rhos and thetas...phis remain unchanged
    (rhos, thetas) = _dp2sp_pos(qs, ps)


    #
    # Next, calculate pathlengths perpendicular to current density element at
    # elements' positions;
  #

    if isI:

      # don't waste cpu cycles if current density is not requested
      dl_perp[:] = 1.

    else:

      dphi = (phi_max-phi_min)

      # ionospheric element
      dl_perp[0] = rho_min * np.sin((theta_max+theta_min)/2.) * dphi

      # equatorial element
      dl_perp[1*n+1] = dphi * rhos[1*n+1] * np.sin(thetas[1*n+1])

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

    dphi = (phi_max-phi_min)
    dtheta=(theta_max-theta_min)

    # ionospheric element
    dl_para[0] = rho_min * dtheta

    # equatorial element
    dp = rho_min / np.sin(theta_min)**2 - rho_min / np.sin(theta_max)**2.
    dl_para[1*n+1] = (dp * np.sin(thetas[1*n+1])**3. /
                      np.sqrt(1. + 3.*np.cos(thetas[1*n+1])**2.) )

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

    # convert ionospheric current density into simple current

    if isI:
      Itheta = Jtheta
      #Iphi = Jphi
    else:
      Itheta = Jtheta * (rho_min * np.sin((theta_max+theta_min)/2.) * dphi)
      #Iphi = Jphi * (rho_min * dtheta) # phi component



    # NOTE: in nature, ionospheric current is NOT thought to flow on constant
    #       q or p, but rather it has components in both "directions". However,
    #       FACs do flow along lines of constant p, and equatorial currents flow
    #       along a line of constant q (i.e., q=0), We force this onto our
    #       current loops, which should transform into a purely theta current.
    #       This calculation is not actually required. -EJR

    # ionospheric element
    Iqs[0] = -np.sin((theta_max+theta_min)/2.) / np.sqrt(1. + 3.*np.cos((theta_max+theta_min)/2.)**2.) * Itheta
    Ips[0] = -2*np.cos((theta_max+theta_min)/2.) / np.sqrt(1. + 3.*np.cos((theta_max+theta_min)/2.)**2.) * Itheta
    Iphis[0] = 0

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


    #
    # Next, convert currents in dipole coordinates to currents in spherical
    #
    (Irhos, Ithetas) = _dp2sp_dir(Iqs, Ips, thetas)


    #
    # Finally, divide current vectors by dl_perp to create current densities
    #
    Jrhos = Irhos / dl_perp
    Jthetas = Ithetas / dl_perp
    Jphis = Iphis / dl_perp


  return (phis, thetas, rhos, Jphis, Jthetas, Jrhos, dl_para, dl_perp)



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
  rhos[0*n+1:1*n+1] = (10**(np.linspace(0, 1, n+1)**2)[:-1] +
                       10**(np.linspace(0, 1, n+1)**2)[1:]) / 2. * rho_min
  thetas[0*n+1:1*n+1] = (theta_max+theta_min)/2.
  phis[0*n+1:1*n+1] = phi_max

  # W FAC elements (place more elements closer to Earth)
  rhos[1*n+2:2*n+2] = (10**(np.linspace(1, 0, n+1)**2)[:-1] +
                       10**(np.linspace(1, 0, n+1)**2)[1:]) / 2. * rho_min

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
  rhos[0*n+1:1*n+1] = (10**(np.linspace(0, 1, n+1)**2)[:-1] +
                       10**(np.linspace(0, 1, n+1)**2)[1:]) / 2. * rho_min

  thetas[0*n+1:1*n+1] = theta_max
  phis [0*n+1:1*n+1] = (phi_max+phi_min)/2.

  # N FAC elements (place more elements closer to Earth)
  rhos[1*n+2:2*n+2] = (10**(np.linspace(1, 0, n+1)**2)[:-1] +
                       10**(np.linspace(1, 0, n+1)**2)[1:]) / 2. * rho_min

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


def test_Type2_AllLongitude():
  """
  A Type 2 Bostrom loop whose boundaries extend 360 degrees (2*pi) should
  result in a surface perturbation of zero. This is stated by Bonnevier et
  al. (1970), although I don't fully follow their proof. Really, it just
  comes down to Ampere's Law for a toroidal solenoid.
  """
  import numpy as np
  import pyLTR.Physics.BS as bs

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
  (dBphi, dBtheta, dBrho) = bs.bs_sphere(r_B2, J_B2, d_B2,
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
