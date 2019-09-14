"""
Use Matplotlib's Basemap Toolkit to plot (typically) ionospheric data onto a
polar map projection. The interface is designed to mimic PolarPlot.py as closely
as possible, although it will necessarily have more options, meaning that code
written to use this module is not guaranteed to work if PolarPlot is dropped
in as a replacement (this module *is* intended as a drop-in replacement for
PolarPlot...i.e., it is backward compatible).
"""

# follow MPL community standards (devaiates from most existing pyLTR code)
import matplotlib.pyplot as plt
import numpy as np

# import Basemap toolkit and supporting module(s)
import mpl_toolkits
mpl_toolkits.__path__.append('/glade/work/phamkh/personal_python_clone3-6/lib/python3.6/site-packages/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap, shiftgrid, interp
#from basemap import Basemap, shiftgrid, interp

# import scipy's interpolate sub-package
from scipy import interpolate as sInterp

# import pyLTR
import pyLTR

# importe datetime
import datetime

# for warning messages
import warnings



## 
## This commented block was a failed attempt to "monkey-patch" Basemap such that
## it might transform map coordinates into an alternate coordinate system (e.g.,
## Solar Magnetic). Turns out Basemap is an unwieldy beast under the hood, and I
## couldn't not get this to work as intended. Still, there may be some useful
## tidbits below, so I'm keeping it around for now. Some specific issues that
## I was unable to adress:
##
## * Basemap class makes heavy use of objects defined within the basemap module,
##   meaning a monkey patch must somehow create/load those objects itself; this
##   would not be an issue if workable patches get incorporated into Basemap
##   package.
## * Basemap does some weird stuff using temporary stereographic projectsions
##   when working in, among others, an orthographic projection. This leads to
##   undesired clipping that I could never explain.
## * Basemap makes heavy use of _geoslib, which is itself not well documented,
##   to say the least.
##
## -EJR 10/2014
##
## 
## # this is a modified version of Basemap._readboundarydata(). After its end, it
## # gets monkeypatched into the Basemap class
## def _rbd(self,name,as_polygons=False, llXform=None, **kwargs):
##       """
##       read boundary data, clip to map projection region.
##       
##       MONKEYPATCH    MONKEYPATCH    MONKEYPATCH 
##       
##       New kwarg llXform: function reference that transforms geographic
##                          longitude-latitude pairs into a non-geographic
##                          coordinate system. For example, convert boundary
##                          coordinates into a geomagnetic reference frame.
##                          The simple recipe should be:
##                          
##                          lonNew, latNew = llXform(lonGeo, latGeo, **kwargs)
##                          
##                          If more info is needed (e.g., datetime), then
##       
##       MONKEYPATCH    MONKEYPATCH    MONKEYPATCH 
##       """
##               
##       print 'Calling Monkeypatched Basemap._readboundarydata()'
##       
##       
##       msg = dedent("""
##       Unable to open boundary dataset file. Only the 'crude', 'low',
##       'intermediate' and 'high' resolution datasets are installed by default.
##       If you are requesting a 'full' resolution dataset, you may need to
##       download and install those files separately
##       (see the basemap README for details).""")
##       # only gshhs coastlines can be polygons.
##       if name != 'gshhs': as_polygons=False
##       try:
##           bdatfile = open(os.path.join(basemap_datadir,name+'_'+self.resolution+'.dat'),'rb')
##           bdatmetafile = open(os.path.join(basemap_datadir,name+'meta_'+self.resolution+'.dat'),'r')
##       except:
##           raise IOError(msg)
##       polygons = []
##       polygon_types = []
##       # coastlines are polygons, other boundaries are line segments.
##       if name == 'gshhs':
##           Shape = _geoslib.Polygon
##       else:
##           Shape = _geoslib.LineString
##       # see if map projection region polygon contains a pole.
##       NPole = _geoslib.Point(self(0.,90.))
##       SPole = _geoslib.Point(self(0.,-90.))
##       boundarypolyxy = self._boundarypolyxy
##       boundarypolyll = self._boundarypolyll
##       hasNP = NPole.within(boundarypolyxy)
##       hasSP = SPole.within(boundarypolyxy)
##       containsPole = hasNP or hasSP
##       # these projections cannot cross pole.
##       if containsPole and\
##           self.projection in _cylproj + _pseudocyl + ['geos']:
##           raise ValueError('%s projection cannot cross pole'%(self.projection))
##       # make sure some projections have has containsPole=True
##       # we will compute the intersections in stereographic
##       # coordinates, then transform back. This is
##       # because these projections are only defined on a hemisphere, and
##       # some boundary features (like Eurasia) would be undefined otherwise.
##       tostere =\
##       ['omerc','ortho','gnom','nsper','nplaea','npaeqd','splaea','spaeqd']
##       if self.projection in tostere and name == 'gshhs':
##           containsPole = True
##           lon_0=self.projparams['lon_0']
##           lat_0=self.projparams['lat_0']
##           re = self.projparams['R']
##           # center of stereographic projection restricted to be
##           # nearest one of 6 points on the sphere (every 90 deg lat/lon).
##           lon0 = 90.*(np.around(lon_0/90.))
##           lat0 = 90.*(np.around(lat_0/90.))
##           if np.abs(int(lat0)) == 90: lon0=0.
##           maptran = pyproj.Proj(proj='stere',lon_0=lon0,lat_0=lat0,R=re)
##           # boundary polygon for ortho/gnom/nsper projection
##           # in stereographic coordinates.
##           b = self._boundarypolyll.boundary
##           blons = b[:,0]; blats = b[:,1]
##           b[:,0], b[:,1] = maptran(blons, blats)
##           boundarypolyxy = _geoslib.Polygon(b)
##       for line in bdatmetafile:
##           linesplit = line.split()
##           area = float(linesplit[1])
##           south = float(linesplit[3])
##           north = float(linesplit[4])
##           crossdatelineE=False; crossdatelineW=False
##           if name == 'gshhs':
##               id = linesplit[7]
##               if id.endswith('E'):
##                   crossdatelineE = True
##               elif id.endswith('W'):
##                   crossdatelineW = True
##           # make sure south/north limits of dateline crossing polygons
##           # (Eurasia) are the same, since they will be merged into one.
##           # (this avoids having one filtered out and not the other).
##           if crossdatelineE:
##               south_save=south
##               north_save=north
##           if crossdatelineW:
##               south=south_save
##               north=north_save
##           if area < 0.: area = 1.e30
##           useit = self.latmax>=south and self.latmin<=north and area>self.area_thresh
##           if useit:
##               typ = int(linesplit[0])
##               npts = int(linesplit[2])
##               offsetbytes = int(linesplit[5])
##               bytecount = int(linesplit[6])
##               bdatfile.seek(offsetbytes,0)
##               # read in binary string convert into an npts by 2
##               # numpy array (first column is lons, second is lats).
##               polystring = bdatfile.read(bytecount)
##               # binary data is little endian.
##               b = np.array(np.fromstring(polystring,dtype='<f4'),'f8')
##               b.shape = (npts,2)
##               
##               
##               
##               
##               #
##               # transform {lon,lat} pairs into alternate coordinates
##               #
##               for i in range(npts):
##                  llNew = llXform(b[i,0], b[i,1], **kwargs)
##                  b[i,0] = llNew[0]
##                  b[i,1] = llNew[i]
##                             
##               
##               
##               
##               b2 = b.copy()
##               # merge polygons that cross dateline.
##               poly = Shape(b)
##               # hack to try to avoid having Antartica filled polygon
##               # covering entire map (if skipAnart = False, this happens
##               # for ortho lon_0=-120, lat_0=60, for example).
##               skipAntart = self.projection in tostere and south < -89 and \
##                not hasSP
##               if crossdatelineE and not skipAntart:
##                   if not poly.is_valid(): poly=poly.fix()
##                   polyE = poly
##                   continue
##               elif crossdatelineW and not skipAntart:
##                   if not poly.is_valid(): poly=poly.fix()
##                   b = poly.boundary
##                   b[:,0] = b[:,0]+360.
##                   poly = Shape(b)
##                   poly = poly.union(polyE)
##                   if not poly.is_valid(): poly=poly.fix()
##                   b = poly.boundary
##                   b2 = b.copy()
##                   # fix Antartica.
##                   if name == 'gshhs' and south < -89:
##                       b = b[4:,:]
##                       b2 = b.copy()
##                       poly = Shape(b)
##               # if map boundary polygon is a valid one in lat/lon
##               # coordinates (i.e. it does not contain either pole),
##               # the intersections of the boundary geometries
##               # and the map projection region can be computed before
##               # transforming the boundary geometry to map projection
##               # coordinates (this saves time, especially for small map
##               # regions and high-resolution boundary geometries).
##               if not containsPole:
##                   # close Antarctica.
##                   if name == 'gshhs' and south < -89:
##                       lons2 = b[:,0]
##                       lats = b[:,1]
##                       lons1 = lons2 - 360.
##                       lons3 = lons2 + 360.
##                       lons = lons1.tolist()+lons2.tolist()+lons3.tolist()
##                       lats = lats.tolist()+lats.tolist()+lats.tolist()
##                       lonstart,latstart = lons[0], lats[0]
##                       lonend,latend = lons[-1], lats[-1]
##                       lons.insert(0,lonstart)
##                       lats.insert(0,-90.)
##                       lons.append(lonend)
##                       lats.append(-90.)
##                       b = np.empty((len(lons),2),np.float64)
##                       b[:,0] = lons; b[:,1] = lats
##                       poly = Shape(b)
##                       if not poly.is_valid(): poly=poly.fix()
##                       # if polygon instersects map projection
##                       # region, process it.
##                       if poly.intersects(boundarypolyll):
##                           if name != 'gshhs' or as_polygons:
##                               geoms = poly.intersection(boundarypolyll)
##                           else:
##                               # convert polygons to line segments
##                               poly = _geoslib.LineString(poly.boundary)
##                               geoms = poly.intersection(boundarypolyll)
##                           # iterate over geometries in intersection.
##                           for psub in geoms:
##                               b = psub.boundary
##                               blons = b[:,0]; blats = b[:,1]
##                               bx, by = self(blons, blats)
##                               polygons.append(list(zip(bx,by)))
##                               polygon_types.append(typ)
##                   else:
##                       # create duplicate polygons shifted by -360 and +360
##                       # (so as to properly treat polygons that cross
##                       # Greenwich meridian).
##                       b2[:,0] = b[:,0]-360
##                       poly1 = Shape(b2)
##                       b2[:,0] = b[:,0]+360
##                       poly2 = Shape(b2)
##                       polys = [poly1,poly,poly2]
##                       for poly in polys:
##                           # try to fix "non-noded intersection" errors.
##                           if not poly.is_valid(): poly=poly.fix()
##                           # if polygon instersects map projection
##                           # region, process it.
##                           if poly.intersects(boundarypolyll):
##                               if name != 'gshhs' or as_polygons:
##                                   geoms = poly.intersection(boundarypolyll)
##                               else:
##                                   # convert polygons to line segments
##                                   # note: use fix method here or Eurasia
##                                   # line segments sometimes disappear.
##                                   poly = _geoslib.LineString(poly.fix().boundary)
##                                   geoms = poly.intersection(boundarypolyll)
##                               # iterate over geometries in intersection.
##                               for psub in geoms:
##                                   b = psub.boundary
##                                   blons = b[:,0]; blats = b[:,1]
##                                   # transformation from lat/lon to
##                                   # map projection coordinates.
##                                   bx, by = self(blons, blats)
##                                   if not as_polygons or len(bx) > 4:
##                                       polygons.append(list(zip(bx,by)))
##                                       polygon_types.append(typ)
##               # if map boundary polygon is not valid in lat/lon
##               # coordinates, compute intersection between map
##               # projection region and boundary geometries in map
##               # projection coordinates.
##               else:
##                   # transform coordinates from lat/lon
##                   # to map projection coordinates.
##                   # special case for ortho/gnom/nsper, compute coastline polygon
##                   # vertices in stereographic coords.
##                   if name == 'gshhs' and as_polygons and self.projection in tostere:
##                       b[:,0], b[:,1] = maptran(b[:,0], b[:,1])
##                   else:
##                       b[:,0], b[:,1] = self(b[:,0], b[:,1])
##                   goodmask = np.logical_and(b[:,0]<1.e20,b[:,1]<1.e20)
##                   # if less than two points are valid in
##                   # map proj coords, skip this geometry.
##                   if np.sum(goodmask) <= 1: continue
##                   if name != 'gshhs' or (name == 'gshhs' and not as_polygons):
##                       # if not a polygon,
##                       # just remove parts of geometry that are undefined
##                       # in this map projection.
##                       bx = np.compress(goodmask, b[:,0])
##                       by = np.compress(goodmask, b[:,1])
##                       # split coastline segments that jump across entire plot.
##                       xd = (bx[1:]-bx[0:-1])**2
##                       yd = (by[1:]-by[0:-1])**2
##                       dist = np.sqrt(xd+yd)
##                       split = dist > 0.5*(self.xmax-self.xmin)
##                       if np.sum(split) and self.projection not in _cylproj:
##                           ind = (np.compress(split,np.squeeze(split*np.indices(xd.shape)))+1).tolist()
##                           iprev = 0
##                           ind.append(len(xd))
##                           for i in ind:
##                               # don't add empty lists.
##                               if len(list(range(iprev,i))):
##                                   polygons.append(list(zip(bx[iprev:i],by[iprev:i])))
##                               iprev = i
##                       else:
##                           polygons.append(list(zip(bx,by)))
##                       polygon_types.append(typ)
##                       continue
##                   # create a GEOS geometry object.
##                   if name == 'gshhs' and not as_polygons:
##                       # convert polygons to line segments
##                       poly = _geoslib.LineString(poly.boundary)
##                   else:
##                       poly = Shape(b)
##                   # this is a workaround to avoid
##                   # "GEOS_ERROR: TopologyException:
##                   # found non-noded intersection between ..."
##                   if not poly.is_valid():
##                      print type(poly)
##                      for coord in poly.get_coords():
##                         print coord
##                      poly=poly.fix()
##                   # if geometry instersects map projection
##                   # region, and doesn't have any invalid points, process it.
##                   if goodmask.all() and poly.intersects(boundarypolyxy):
##                       # if geometry intersection calculation fails,
##                       # just move on.
##                       try:
##                           geoms = poly.intersection(boundarypolyxy)
##                       except:
##                           continue
##                       # iterate over geometries in intersection.
##                       for psub in geoms:
##                           b = psub.boundary
##                           # if projection in ['ortho','gnom','nsper'],
##                           # transform polygon from stereographic
##                           # to ortho/gnom/nsper coordinates.
##                           if self.projection in tostere:
##                               # if coastline polygon covers more than 99%
##                               # of map region for fulldisk projection,
##                               # it's probably bogus, so skip it.
##                               #areafrac = psub.area()/boundarypolyxy.area()
##                               #if self.projection == ['ortho','nsper']:
##                               #    if name == 'gshhs' and\
##                               #       self._fulldisk and\
##                               #       areafrac > 0.99: continue
##                               # inverse transform from stereographic
##                               # to lat/lon.
##                               b[:,0], b[:,1] = maptran(b[:,0], b[:,1], inverse=True)
##                               # orthographic/gnomonic/nsper.
##                               b[:,0], b[:,1]= self(b[:,0], b[:,1])
##                           if not as_polygons or len(b) > 4:
##                               polygons.append(list(zip(b[:,0],b[:,1])))
##                               polygon_types.append(typ)
##       return polygons, polygon_types
##   
## 
## 
## # monkeypatch mostly for testing; changes should be submitted to the Basemap
## # developer if they actually work...he can figure out how to properly include
## # them.
## 
## # I don't really understand monkeypatching, but the Basemap class relies heavily
## # on global object definitions within the basemap module, which of course we do
## # not have access to without the following...
## from mpl_toolkits.basemap import *
## from mpl_toolkits.basemap import _geoslib
## from mpl_toolkits.basemap import _proj
## from mpl_toolkits.basemap import _cylproj
## from mpl_toolkits.basemap import _pseudocyl
## 
## # finally, substitute _rdb for _readboundarydata
## Basemap._readboundarydata = _rbd
## 
## 
## def geo2sm(lonGeo, latGeo, **kwargs):
##    """
##    Wrapper to Geopack libraries that transforms {lon,lat} pairs in Geographic
##    coordinates to {lon,lat} pairs in Solar Magnetic coordinates.
##    
##    INPUTS
##       - lonGeo          : longitude of point in geographic coordinates in degrees
##       - latGeo          : latitude of point in geographic coordinates in degrees
##    
##    
##    KEYWORDS
##       - dtUTC           : datetime.datetime object specifying the UTC time at
##                           which to perform the transform; necessary because
##                           SM coordinates are not static, but change as the
##                           geomagnetic pole migrates.
##    
##    OUTPUTS
##       - lonSM           : longitude in SM coordinates in degrees
##       - latSM           : latitude in SM coordinates in degrees
##    """
##    
##    # use **kwargs to get keyword arguments
##    dtUTC = kwargs.pop('dtUTC', None)
##    
##    
##    # set default dtUTC to noon today...I think this will place geographic pole
##    # noon-ward of magnetic pole, but in any case, this is consistent with other
##    # functions' default dtUTC in this module.
##    if dtUTC == None:
##       dtUTC = datetime.datetime.combine(datetime.date.today(), 
##                                         datetime.time(12))
##       warnings.warn('UTC assumed to be noon today')
##    
##    
##    # convert spherical coordinates to XYZ on unit-radius sphere
##    x,y,z = SPHtoCAR(lonGeo*np.pi/180, (90-latGeo)*np.pi/180, 1)
##    
##    
##    # transform to SM coordinates
##    x,y,z = GEOtoSM(x, y, z, dtUTC)
##    
##    
##    # convert XYZ into spherical coordinates on unit-radius sphere
##    lonSM, colatSM, radSM = pyLTR.transform.CARtoSPH(x,y,z)
##    
##    
##    # convert lonSM,latSM back into degrees before returning
##    
##    return lonSM*180/np.pi, 90-(colatSM*180/np.pi)
##    
   




def _normalize180(lon):
    """
    Normalize lon to range [180, 180)
    
    ...shamelessly stolen from test_rotpole.py in Basemap package
    """
    lower = -180.; upper = 180.
    if lon > upper or lon == lower:
        lon = lower + abs(lon + upper) % (abs(lower) + abs(upper))
    if lon < lower or lon == upper:
        lon = upper - abs(lon - lower) % (abs(lower) + abs(upper))
    return lower if lon == upper else lon   


   


def basePolar(dtUTC,
              coordSystem='Geographic',
              longTicks=np.arange(180,-180,-45),
              longLabel=False,  
              colatTicks=np.arange(0,181,30),
              colatLabel=False,
              coordLongTicks=None,
              coordLongLabel=False,  
              coordColatTicks=None,
              coordColatLabel=False,
              resolution='l',
              ax=None):
   """
   Generates a Basemap object and plot with no data on it. This is a wrapper
   for the Basemap class, but is quite restrictive, only allowing north/south
   polar orthographic projections, with local noon pointing up.
   
 
   Inputs
      - dtUTC           : datetime object in UTC (required for geographic to
                          geomagnetic transformations, and to ensure noon
                          local-time always points up).
   
   Keywords
      - coordSystem     : a string specifying the coordinate system of plotted
                          data; if not 'Geographic', all Basemap objects (e.g.,
                          meridians, parallels, boundaries), which are usually
                          assumed to be in geographic coordinates, will be 
                          transformed into this coordinate system.
                          (options: 'Geographic', ...need to add one or two...)
                          (default='Geographic'; i.e., no transform)
      - longTicks       : longitudes at which to draw meridians 
                          (default=np.arange(-180,181,30))
      - longLabel       : label the longitude meridians (default=False)
      - colatTicks      : colatitudes at which to draw parallels 
                          (default=np.arange(0,181,30))
      - colatLabel      : label the latitutde parallels (default=False)
      - coordLongTicks  : longitudes at which to draw coordSystem meridians 
                          (default=None)
      - coordLongLabel  : label coordSystem longitude meridians (default=False)
      - coordColatTicks : colatitudes at which to draw coordSystem parallels;
                          this keyword argument is used to specify which hemi-
                          sphere (N or S) of coordSystem is plotted...if more
                          parallels are less than 90, plot the northern hemi-
                          sphere, otherwise plot southern (default=0)
      - coordColatLabel : label coordSystem latitutde parallels (default=False)
      - resolution      : resolution of map features 
                          (default='l', or Basemap's "low" res)
      - ax              : axes instance to plot Basemap objects to
                          (default=None, or gca())
    
    Outputs
      Reference to a mpl_toolkits.basemap.Basemap object
   
   """
   
   """
   For now we just plot in geographic-pole-centered coordinates, which means
   that the magnetic pole will move around as the Earth rotates. This is not
   ideal for scientific analysis, but probably OK for simple visualizations.
   
   Unfortunately we cannot simply transform the objects generated at Basemap
   initialization, and by member functions like drawstates(), because all 
   related coordinates are already clipped to a non-transformed map boundary, 
   meaning any coordinate transformation will result in truncated geographic 
   features.
   
   Somehow Basemap boundaries, stored in geographic {lon,lat} coordinates, must
   be extracted from Basemap data files, transformed into geo-magnetic {lon,lat} 
   coordinates, transformed into map {x,y} coordinates, clipped to map boundary,
   and re-saved as object attributes that Basemap knows how to handle (e.g., a 
   list named "coastpolygons"). In short, we need to re-implement the Basemap
   member function _readboundarydata() with a coordinate transformation option.
   
   Unfortunately, _readboundarydata() is a fairly complicated support function,
   and as such, not part of Basemap's API. This means that if we monkey-patch
   (i.e., dynamically replace the attribute Basemap._readboundarydata with our 
   own function at runtime), it is not-at-all unlikely that things will break
   when a future version of Basemap is released.
      
   Perhaps we can monkey-patch _readboundarydata, and submit a patch to the
   Basemap developer(s), crossing fingers that it gets included in a future
   release. 
   
   NOTE: Early attempts at monkey-patching failed because the monkey-patched
   _readboundarydata could not access attributes available to the original 
   function. The only thing I can think of to try to get around this is import
   everything imported in Basemap.__init__ inside this module, but I have very 
   little confidence even this will work; there is something fundamental I 
   still do not understand about monkey-patching, and 'scope' in Python.
   
   -EJR 9/2014
   """   
   
   
   # always clear current axes instance before creating a new basemap
   plt.cla()
   
   # check for valid coordSystem before anything else
   if coordSystem not in ['Geographic']:
      print(('Unrecognized coordinate system '+coordSystem+' requested'))
      raise Exception
   
   # check if we are plotting northern hemisphere, or southern
   if ((np.array(coordColatTicks) < 90).sum() > 
       (np.array(coordColatTicks) > 90).sum()):
      isNorth = True
   elif ((np.array(coordColatTicks) < 90).sum() < 
         (np.array(coordColatTicks) > 90).sum()):
      isNorth = False
   else:
      print('Ambiguous hemisphere specified via coordColatTicks keyword argument')
      raise Exception
   
   
   # process known coordinate systems
   if coordSystem is 'Geographic':
      
      # lt0 is the local time of the 0 longitude meridian in degrees
      lt0 = (dtUTC.hour + dtUTC.minute/60. + dtUTC.second/3600.) * 360./24.
      
      # lon_0 should be -lt0 in northern hemisphere, and -lt0+180 in southern,
      # in order to ensure noon localtime always points up
      if isNorth:
         lon_0 = -lt0
         lat_0 = 90
      else:
         lon_0 = -lt0+180
         lat_0 = -90
      
      # initialize Basemap
      # NOTE: as I understand it, specifying kwarg ax here ensures that all
      #       subsequent drawing methods called from this Basemap object
      #       will use this axes unless they are called with their own
      #       kwarg ax
      m = Basemap(projection='ortho', lat_0=lat_0, lon_0=_normalize180(lon_0), 
                  resolution=resolution, ax=ax)
      
      # draw continents
      continents = m.fillcontinents(color='gray', lake_color=None)
      
      # draw geographic meridians and label them if requested
      meridiansGeo = m.drawmeridians(np.array(longTicks), latmax=90,
                                     labels=np.array([1,1,1,1])*longLabel)
      
      # draw geographic parallels
      parallelsGeo = m.drawparallels(90 - np.array(colatTicks), latmax=90)
      
      # Basemap only places labels where lines intersect a map boundary, so
      # we need to manually place parallel labels inside the pole-centered
      # orthographic map.
      if np.array(colatLabel).any():
         # FIXME: do something here
         pass
      
      
   else:
      # should never get here, since we checked coordSystem previously
      print(('Unrecognized coordinate system '+coordSystem+' requested'))
      raise Exception
   
   
   # return Basemap instance
   return m
   


def plotPolar(longitude, colatitude, *args, **kwargs):
   """
   plot lines or points on a mpl_toolkits.basemap.Basemap. This is a loose 
   wrapper for Basemap.plot(), allowing *args and **kwargs to be passed through 
   (many of which are subsequently passed through to plt.plot()).
   
   Inputs
      - longitude       : longitudes (degrees)
      - colatitude      : colatitudes (degrees)
      - *args           : non-keyword arguments to pass to Basemap.plot()
                          (e.g., a format specifier string)
                          
   
   Keywords
      - bm              : a mpl_toolkits.basemap.Basemap to use to plot data,
                          ideally generated with BasePolar() above; If None,
                          create mpl_toolkits.basemap.Basemap with BasePolar(),
                          directing 0 longitude up (i.e., as if UTC is noon).
                          (default=None)
      - zorder          : sets the stacking order for plot items; ideally this
                          would just be passed through in **kwargs, but Basemap 
                          reliably places contours below continents if the two 
                          have the same zorder, so we force a useful default
                          (default=2)
      - ax              : axes instance to plot Basemap objects to
                          (default=None, or gca())
      - **kwargs        : keyword arguments to pass to Basemap.plot()
                          NOTE: I don't fully understand **kwargs, but it seems
                                we can't actually specify our own keywords in 
                                this function definition, or they get treated
                                as positional arguments; they must be extracted 
                                from **kwargs - EJR 9/2014
    
    Outputs
      Reference to plot object; or
      a tuple with a plot object and a Basemap object if bm=None on input
   
   
   """
   
   
   # explicit keywords cannot be passed if **kwargs used (?)
   bm = kwargs.pop('bm',None)
   zorder = kwargs.pop('zorder',2)
   ax = kwargs.pop('ax',None)
   
   
   # check if a Basemap must be created first
   if bm==None:
      ret_bm=True
   else:
      ret_bm=False
   
   
   if ret_bm:
      
      # determine if northern or southern hemisphere should be plotted
      ucolats = np.unique(colatitude)
         # check if we are plotting northern hemisphere, or southern
      if (ucolats < 90).sum() > (ucolats > 90).sum():
         coordColatTicks=0
      elif (ucolats < 90).sum() < (ucolats > 90).sum():
         coordColatTicks=180
      else:
         warnings.warn('Hemisphere not obvious, assuming north')
         coordColatTicks=0
      
      # assume UTC is Noon, today
      dtUTC = datetime.datetime.combine(datetime.date.today(), 
                                        datetime.time(12))
      
      # generate a default Basemap
      warnings.warn('Geographic coordinate system assumed')
      warnings.warn('UTC assumed to be noon today')
      bm = basePolar(dtUTC, 
                     coordSystem='Geographic',
                     coordColatTicks=coordColatTicks,
                     ax=ax)
   
   
   # convert lons,colats into x,y projection coordinates
   x,y = bm(np.array(longitude), 90 - np.array(colatitude))
   
   
   # plot points/lines
   l = bm.plot(x, y, zorder=zorder, ax=ax, *args, **kwargs)
   
   
   # this is probably bad Python practice
   if ret_bm:
      return l, bm
   else:
      return l
 



def pcolorPolar(longGrid, colatGrid, scalarGrid,
                *args,
                **kwargs):
   """
   Generates pseudo-color plot on a mpl_toolkits.basemap.Basemap. This is a loose
   wrapper for Basemap.contour(), allowing most *args and **kwargs to be passed 
   through (many of which are subsequently passed through to plt.pcolor()).
   
 
   Inputs
      - longGrid        : meshgrid of longitudes (degrees)...unless tri==True
      - colatGrid       : meshgrid of colatitudes (degrees)...unless tri==True
      - scalarGrid      : scalar values corresponding to coordinates specified
                          by longGrid and colatGrid
      - *args           : non-keyword arguments to pass to Basemap.contour()
                          (e.g., a scalar N specifying the number of contours,
                                 or a vector V specifying the contour levels)
                          
   
   Keywords
      - bm              : a mpl_toolkits.basemap.Basemap to use to plot data,
                          ideally generated with BasePolar() above; If None,
                          create mpl_toolkits.basemap.Basemap with BasePolar(),
                          directing 0 longitude up (i.e., as if UTC is noon).
                          (default=None)
      - useMesh         : use Basemap.pcolormesh() instead of Basemap.pcolor();
                          it should be faster (default=False)
      - zorder          : sets the stacking order for conours; ideally this would
                          just be passed through in **kwargs, but Basemap seems 
                          to reliably place contours below continents if the two 
                          have the same zorder, so we force a useful default
                          (default=2)
      - ax              : axes instance to plot Basemap objects to
                          (default=None, or gca())
      - **kwargs        : keyword arguments to pass to Basemap.pcolor()
                          NOTE: I don't fully understand **kwargs, but it seems
                                we can't actually specify our own keywords in 
                                this function definition, or they get treated
                                as positional arguments; they must be extracted 
                                from **kwargs - EJR 9/2014
    
    Outputs
      Reference to pcolor object; or
      a tuple with a pcolor object and a Basemap object if bm=None on input

   """
   
   # explicit keywords cannot be passed if **kwargs used (?)
   bm = kwargs.pop('bm',None)
   useMesh = kwargs.pop('useMesh',False)
   zorder = kwargs.pop('zorder',2)
   ax = kwargs.pop('ax',None)
   
   
   # check if a Basemap must be created first
   if bm==None:
      ret_bm=True
   else:
      ret_bm=False
   
   
   if ret_bm:
      
      # determine if northern or southern hemisphere should be plotted
      ucolats = np.unique(colatGrid)
         # check if we are plotting northern hemisphere, or southern
      if (ucolats < 90).sum() > (ucolats > 90).sum():
         coordColatTicks=0
      elif (ucolats < 90).sum() < (ucolats > 90).sum():
         coordColatTicks=180
      else:
         warnings.warn('Hemisphere not obvious, assuming north')
         coordColatTicks=0
      
      # assume UTC is Noon, today
      dtUTC = datetime.datetime.combine(datetime.date.today(), 
                                        datetime.time(12))
      
      # generate a default Basemap
      warnings.warn('Geographic coordinate system assumed')
      warnings.warn('UTC assumed to be noon today')
      bm = basePolar(dtUTC, 
                     coordSystem='Geographic',
                     coordColatTicks=coordColatTicks,
                     ax=ax)
   
   
   # convert lons,colats into x,y projection coordinates
   x,y = bm(np.array(longGrid), 90 - np.array(colatGrid))
   
   
   # plot contours
   if useMesh:
      c = bm.pcolormesh(x, y, scalarGrid, zorder=zorder, ax=ax, *args, **kwargs)
   else:
      c = bm.pcolor(x, y, scalarGrid, zorder=zorder, ax=ax, *args, **kwargs)
   
   
   # this is probably bad Python practice
   if ret_bm:
      return c, bm
   else:
      return c



   
def contourPolar(longGrid, colatGrid, scalarGrid,
                 *args,
                 **kwargs):
   """
   Generates contours on a mpl_toolkits.basemap.Basemap. This is a loose wrapper
   for Basemap.contour(), allowing most *args and **kwargs to be passed through
   (many of which are subsequently passed through to plt.contour()).
   
 
   Inputs
      - longGrid        : meshgrid of longitudes (degrees)...unless tri==True
      - colatGrid       : meshgrid of colatitudes (degrees)...unless tri==True
      - scalarGrid      : scalar values corresponding to coordinates specified
                          by longGrid and colatGrid
      - *args           : non-keyword arguments to pass to Basemap.contour()
                          (e.g., a scalar N specifying the number of contours,
                                 or a vector V specifying the contour levels)
                          
   
   Keywords
      - bm              : a mpl_toolkits.basemap.Basemap to use to plot data,
                          ideally generated with BasePolar() above; If None,
                          create mpl_toolkits.basemap.Basemap with BasePolar(),
                          directing 0 longitude up (i.e., as if UTC is noon).
                          (default=None)
      - filled          : if True, call contourf(), else call contour()
                          (default=True)
      - zorder          : sets the stacking order for conours; ideally this would
                          just be passed through in **kwargs, but Basemap seems 
                          to reliably place contours below continents if the two 
                          have the same zorder, so we force a useful default
                          (default=2)
      - ax              : axes instance to plot Basemap objects to
                          (default=None, or gca())
      - **kwargs        : keyword arguments to pass to Basemap.contour()
                          (e.g., tri==True to force contours on an unstructured
                                 grid, in which case inputs to this function
                                 should be vectors, not grids)
                          NOTE: I don't fully understand **kwargs, but it seems
                                we can't actually specify our own keywords in 
                                this function definition, or they get treated
                                as positional arguments; they must be extracted 
                                from **kwargs - EJR 9/2014
    
    Outputs
      Reference to contour object; or
      a tuple with a contour object and a Basemap object if bm=None on input

   """
   
   # explicit keywords cannot be passed if **kwargs used (?)
   bm = kwargs.pop('bm',None)
   filled = kwargs.pop('filled',True)
   zorder = kwargs.pop('zorder',2)
   ax = kwargs.pop('ax',None)
   
   
   # check if a Basemap must be created first
   if bm==None:
      ret_bm=True
   else:
      ret_bm=False
   
   
   if ret_bm:
      
      # determine if northern or southern hemisphere should be plotted
      ucolats = np.unique(colatGrid)
         # check if we are plotting northern hemisphere, or southern
      if (ucolats < 90).sum() > (ucolats > 90).sum():
         coordColatTicks=0
      elif (ucolats < 90).sum() < (ucolats > 90).sum():
         coordColatTicks=180
      else:
         warnings.warn('Hemisphere not obvious, assuming north')
         coordColatTicks=0
      
      # assume UTC is Noon, today
      dtUTC = datetime.datetime.combine(datetime.date.today(), 
                                        datetime.time(12))
      
      # generate a default Basemap
      warnings.warn('Geographic coordinate system assumed')
      warnings.warn('UTC assumed to be noon today')
      bm = basePolar(dtUTC, 
                     coordSystem='Geographic',
                     coordColatTicks=coordColatTicks,
                     ax=ax)
   
   
   # convert lons,colats into x,y projection coordinates
   x,y = bm(np.array(longGrid), 90 - np.array(colatGrid))
   
   
   # plot contours
   if filled:
      c = bm.contourf(x, y, scalarGrid, zorder=zorder, ax=ax, *args, **kwargs)
   else:
      c = bm.contour(x, y, scalarGrid, zorder=zorder, ax=ax, *args, **kwargs)
   
   
   # this is probably bad Python practice
   if ret_bm:
      return c, bm
   else:
      return c


   
   
def quiverPolar(longGrid, colatGrid, vEastward, vSouthward,
                *args,
                **kwargs):
   """
   Generates 2d vector field on a mpl_toolkits.basemap.Basemap. This is a loose
   wrapper for Basemap.quiver(), allowing most *args and **kwargs to be passed 
   through (many of which are subsequently passed through to plt.quiver()).
   
 
   Inputs
      - longGrid        : meshgrid of longitudes (degrees)...unless tri==True
      - colatGrid       : meshgrid of colatitudes (degrees)...unless tri==True
      - vEastward       : locally eastward components of vector field
      - vSouthward      : locally southward components of vector field
      - *args           : non-keyword arguments to pass to Basemap.contour()
                          (e.g., a scalar N specifying the number of contours,
                                 or a vector V specifying the contour levels)
                          
   
   Keywords
      - bm              : a mpl_toolkits.basemap.Basemap to use to plot data,
                          ideally generated with BasePolar() above; If None,
                          create mpl_toolkits.basemap.Basemap with BasePolar(),
                          directing 0 longitude up (i.e., as if UTC is noon).
                          (default=None)
      - zorder          : sets the stacking order for arrows; ideally this would
                          just be passed through in **kwargs, but Basemap seems 
                          to reliably place arrows below continents if the two 
                          have the same zorder, so we force a useful default
                          (default=2)
      - nx              : number of grid points along map x-axis on which to 
                          place arrows
      - ny              : number of grid points along map y-axis on which to
                          place arrows
                          NOTE: if only one of nx or ny is passed, generate square grid;
                                if neither is passed, use original lon,lat grid (default)
      - ax              : axes instance to plot Basemap objects to
                          (default=None, or gca())
      - **kwargs        : keyword arguments to pass to Basemap.quiver()
                          NOTE: I don't fully understand **kwargs, but it seems
                                we can't actually specify our own keywords in 
                                this function definition, or they get treated
                                as positional arguments; they must be extracted 
                                from **kwargs - EJR 9/2014
    
    Outputs
      Reference to quiver object; or
      a tuple with a quiver object and a Basemap object if bm=None on input

   """
   
   # explicit keywords cannot be passed if **kwargs used (?)
   bm = kwargs.pop('bm',None)
   zorder = kwargs.pop('zorder', 2)
   ax = kwargs.pop('ax', None)
   nx = kwargs.pop('nx', None)
   ny = kwargs.pop('ny', None)
   if nx==None and ny!=None: nx=ny
   if ny==None and nx!=None: ny=nx
   
   
   # check if a Basemap must be created first
   if bm==None:
      ret_bm=True
   else:
      ret_bm=False
   
   
   if ret_bm:
      
      # determine if northern or southern hemisphere should be plotted
      ucolats = np.unique(colatGrid)
         # check if we are plotting northern hemisphere, or southern
      if (ucolats < 90).sum() > (ucolats > 90).sum():
         coordColatTicks=0
      elif (ucolats < 90).sum() < (ucolats > 90).sum():
         coordColatTicks=180
      else:
         warnings.warn('Hemisphere not obvious, assuming north')
         coordColatTicks=0

      # assume UTC is Noon, today
      dtUTC = datetime.datetime.combine(datetime.date.today(), 
                                        datetime.time(12))
      
      # generate a default Basemap
      warnings.warn('Geographic coordinate system assumed')
      warnings.warn('UTC assumed to be noon today')
      bm = basePolar(dtUTC, 
                     coordSystem='Geographic',
                     coordColatTicks=coordColatTicks,
                     ax=ax)
   
   
   # convert lons,colats and vectors into x,y,u,v projection coordinates
   if nx==None and ny==None:
      # vectors remain on input grid, presumably regular in lon,lat
      u,v,x,y = bm.rotate_vector(vEastward, -vSouthward,
                                 longGrid, 90 - np.array(colatGrid),
                                 returnxy=True)
   else:
      """
      #
      # Basemap.transform_vector() does not generate results consistent with 
      # Basemap.rotate_vector(). We re-implment transform_vector() here, but 
      # call a different 2d interpolator. -EJR 9/2014
      #
      
      # vectors placed on regular grid in map coordinates
      # FIXME: the interpolation algorithm used here requires that the longitudes
      #        monotonically increase from -180 to 180 on a regular grid, where
      #        longitude cooresponds to the right-most dimension. This is not
      #        how our MIX grids are set up, so we need to manipulate things a
      #        bit making non-general assumptions.
      longitudes = (longGrid.T)[0,:] # assumes MIX grid!
      latitudes = np.flipud((90-colatGrid.T)[:,0]) # assumes MIX grid!
      ugrid,newLongs = shiftgrid(180., np.array(vEastward).T, # assumes MIX grid
                                 longitudes, start=False)
      vgrid,newLongs = shiftgrid(180., np.flipud(-np.array(vSouthward).T), # assumes MIX grid
                                 longitudes, start=False)
      u,v,x,y = bm.transform_vector(ugrid, vgrid, newLongs, latitudes,
                                    nx, ny, returnxy=True, masked=True)      
      """
      
      uin,vin,xin,yin = bm.rotate_vector(vEastward, -vSouthward,
                                         longGrid, 90 - np.array(colatGrid),
                                         returnxy=True)
      longs, lats, x, y = bm.makegrid(nx,ny,returnxy=True)
      
      
      u = sInterp.griddata((xin.flatten(), yin.flatten()), uin.flatten(), 
                           (x, y), method='linear')
      v = sInterp.griddata((xin.flatten(), yin.flatten()), vin.flatten(), 
                           (x, y), method='linear')
      
      
   # plot vector field
   q = bm.quiver(x, y, u, v, zorder=zorder, ax=ax, *args, **kwargs)
   
   
   # this is probably bad Python practice
   if ret_bm:
      return q, bm
   else:
      return q



def QuiverPlotDict(longitude, colatitude, scalars, vectors,
                   plotOpts1=None, plotOpts2=None, 
                   longTicks=None, longLabels=None, 
                   colatTicks=None, colatLabels=None,
                   northPOV=None, coordSystem=None, dtUTC=None,
                   plotColorBar=True, plotQuiverKey=True,
                   useMesh=False,
                   points=[],
                   tri=False,
                   userAxes=None):
   """
   Produce a well-labeled polar plot of a vector field overlaid on contours of
   a scalar field, all overtop a polar-POV map of the Earth. This is a wrapper 
   for quiverPolar() and contourPolar() that is intended to be a drop-in 
   replacement for PolarPlot.QuiverPlotDict().
   
   Inputs
     - first is a dictionary holding a 2D meshgrid of longitudes
     - second is a dictionary holding a 2D meshgrid of colatitudes
     - third is dictionary holding a 2D array of a scalar field whose elements 
       are located at coordinates specified in the first and second arguments;
       as per MIX convention: dim1 = longitudes, dim2=colatitudes
     - fourth is a 2-tuple of 2D arrays of local {long,colat} direction vector
       components whose elements are located at coordinates specified in the
       first and second arguments; dim1 = longitudes, dim2=colatitudes
   
   Keywords
     - plotOpts1  - dictionary holding various optional parameters for tweaking
                    the scalar field appearance
                    'colormap': string specifying a colormap for surface or 
                                contourf plots
                                FIXME: this should be an actual colormap object,
                                       NOT just the name of a standard colormap
                    'min': minimum data value mapped to colormap
                    'max': maximum data value mapped to colormap
                    'format_str': format string for min/max labels
                    'numContours': number of contours between min and max
                    'numTicks': number of ticks in colorbar
                    
     - plotOpts2  - dictionary holding various optional parameters for tweaking
                    the vector field appearance
                    'scale': floating point number specifying how many data unit
                             vector arrows will fit across the width of the plot;
                             default is for max. vector amplitude to be 1/10th
                             plot width
                    'width': floating point number specifying the width of an
                             arrow shaft as a fraction of plot width
                    'pivot': should be one of:
                             'tail' - pivot about tail (default)
                             'middle' - pivot about middle
                             'tip' - pivot about tip
                    'color': color of arrow
     
     - longTicks  - where to place 'longitude' ticks in radians
     - longLabels - labels to place at longTicks
     - colatTicks - where to place 'colatiude' ticks in 'colatitude' units
     - colatLabels- labels to place at colatTicks
     
     - northPOV   - force a north-polar POV if True; a south polar-POV if False;
                    NOTE: unlike PolarPlot.QuiverPlotDict(), this keyword forces
                          an actual coordinate transformation, not adjustments
                          to the plot labels. So, if set to False, AND the bulk
                          of colatitudes appear to be in the north, these will
                          be rotated about the X axis to place them in the south,
                          and a southern hemisphere map will be generated. This
                          allows users to plot MIX southern hemisphere data
                          without having to rotate the data themselves. If set
                          to True, and colatitudes imply north-polar POV, no 
                          transformation will be performed.
                          (default=None; POV determined from colatitudes)
     
     - coordSystem- a string specifying the coordinate system of plotted data;
                    if not 'Geographic', all Basemap objects (e.g., meridians,
                    parallels, boundaries), which are usually assumed to be in
                    geographic coordinates, will be transformed into this
                    coordinate system.
                    (options: 'Geographic', ...need to add one or two...)
                    (default='Geographic'; i.e., no transform)
     
     - dtUTC      - datetime object in UTC (required for geographic to
                    geomagnetic transformations, and to ensure noon local-time
                    always points up).
                         
     - plotColorBar- if True, plot a colorbar
     - plotQuiverKey- if True, plot and label a scaled arrow outside the plot
     - useMesh    - if True, plot scalar field as surface plot, not filled contours
                    (should be quicker in theory, but seems to be much slower)
     - points     - a list of (lon,colat[,label]) sets that define and label 
                    points on the map; these should be in the same coordinate
                    system as longitude and colatitude.
     - trie       - used to interpolate irregular grids in contour plots.
     - userAxes   - Used to set ax kwarg in Basemap initialization.
     
   Outputs
     Reference to a Basemap object (?)
   
   """
   
   # extract coordinates from dictionaries
   longGrid = longitude['data'].copy() * 180/np.pi
   colatGrid = colatitude['data'].copy() * 180/np.pi
   
   if dtUTC==None:
      # if dtUTC keyword not passed, assume 0 longitude points to noon
      dtUTC = datetime.datetime.combine(datetime.date.today(), 
                                        datetime.time(12))
      warnings.warn('UTC assumed to be noon today')

   if coordSystem==None:
      # if coordSystem keyword not passed, assume geographic
      coordSystem='Geographic'
      warnings.warn('Geographic coordinate system assumed')
   
      

   # determine default hemisphere to plot
   # NOTE: this is a MIX-centric kludge, and may not generalize well
   ucolats = np.unique(colatGrid)
      # check if we are plotting northern hemisphere, or southern
   if (ucolats < 90).sum() > (ucolats > 90).sum():
      poleColat=0
   elif (ucolats < 90).sum() < (ucolats > 90).sum():
      poleColat=180
   else:
      warnings.warn('Hemisphere not obvious, assuming north')
      poleColat=0
   
   
   # reconcile northPOV and poleColat if they differ
   flipHemi=False
   if northPOV==None:
      pass
   elif northPOV and poleColat==0:
      pass
   elif not northPOV and poleColat==180:
      pass
   else:
      flipHemi = True # flip coordinates for, e.g., MIX southern hemisphere data
      longGrid = 360 - longGrid
      colatGrid = 180 - colatGrid
      poleColat = 180 - poleColat
   
   

   # ideally, coordSystem would simply get passed through to basePolar(), but
   # since basePolar() cannot yet handle anything but Geographic coordinates,
   # we must transform non-Geographic coordinates into Geographic coordinates,
   # then change coordSystem to 'Geographic', before calling basePolar().
   sm2geo=False
   if coordSystem == 'Geographic':
      pass
   elif coordSystem == 'Solar Magnetic':
      sm2geo=True
      
      # convert coordinates back to radians
      longGrid = longGrid * np.pi/180
      colatGrid = colatGrid * np.pi/180
      
      # rotate from SM to GEO coordinates;
      longGridOrig = longGrid.copy() # save for possible vector transform later
      colatGridOrig = colatGrid.copy() # save for possible vector transform later
      x,y,z = pyLTR.transform.SPHtoCAR(longGrid,colatGrid,1)
      x,y,z = pyLTR.transform.SMtoGEO(x,y,z,dtUTC)
      longGrid, colatGrid, _ = pyLTR.transform.CARtoSPH(x,y,z)
                     
      longGrid = longGrid * 180/np.pi
      colatGrid = colatGrid * 180/np.pi
      
      coordSystem = 'Geographic'
      
   else:
      print(('Unrecognized coordinate system '+coordSystem+' specified via kwarg'))
      raise Exception
   
   
   # generate Basemap
   bm = basePolar(dtUTC, coordSystem=coordSystem, coordColatTicks=poleColat,
                  ax=userAxes)
   
   
   
   
   # if scalars is False (or None, or empty, etc.), skip and proceed to quiver plot
   if scalars:
      
      # PROCESS PLOT OPTIONS FOR FIRST (SCALAR) INPUT
      
      if plotOpts1 == None:
         plotOpts1 = {}
      
      # if colormap is supplied use it
      if 'colormap' in plotOpts1:
         cmap1=eval('plt.cm.'+plotOpts1['colormap'])
      else:
         cmap1=None # default is used

      # if limits are given use them, if not use the variables min/max values
      if 'min' in plotOpts1:
         lower1 = plotOpts1['min']
      else:
         lower1 = scalars['data'].min()
      if 'max' in plotOpts1:
         upper1 = plotOpts1['max']
      else:
         upper1 = scalars['data'].max()
      # if format string for max/min is given use otherwise do default
      if 'format_str' in plotOpts1:
         format_str1 = plotOpts1['format_str']
      else:
         format_str1='%.2f'

      # if number of contours is given use it otherwise do 51
      if 'numContours' in plotOpts1:
         ncontours1 = plotOpts1['numContours']
      else:
         ncontours1 = 31

      # if number of ticks is given use it otherwise do 51
      if 'numTicks' in plotOpts1:
         nticks1 = plotOpts1['numTicks']
      else:
         nticks1 = 11
      
      
      contours1 = np.linspace(lower1,upper1,ncontours1)
      ticks1 = np.linspace(lower1,upper1,nticks1)
      
      scalarGrid = scalars['data']
      
      
      if (useMesh):
         # should probably change keyword useMesh to usePcolor, since there is
         # pcolormesh() as an option
         c = pcolorPolar(longGrid, colatGrid, scalarGrid, bm=bm, 
                         cmap=cmap1, vmin=lower1, vmax=upper1, alpha=0.7, 
                         edgecolors='face',tri=tri)
      else:
         # plot filled contours
         c = contourPolar(longGrid, colatGrid, scalarGrid, contours1, bm=bm,
                          filled=True, extend='both', cmap=cmap1, alpha=0.7,
                          tri=tri)
      
      
      if (plotColorBar):
         # NOTE: it *should not* be necessary to pass the mappable c to Basemap's
         #       colorbar function, but something is wrong with plt.gci(), such
         #       that it returns None when called from bm.colorbar(), causing
         #       an attribute error...this kluge short-circuits the whole issue,
         #       but matplotlib needs this to be fixed.
         cb = bm.colorbar(c, ticks=ticks1)
         cb.set_label(scalars['name']+' ['+scalars['units']+']')
      
      
      plt.annotate(('min: '+format_str1+'\nmax: ' + format_str1) % 
                 (scalarGrid.min() ,scalarGrid.max()), 
                 (0.65, -.05),  xycoords='axes fraction',
                 annotation_clip=False)
   
   


    # if vectors is False (or None, or empty, etc.), skip quiver plot
   if vectors:
      
      # copy 'longitudinal' component of directional vector to polar theta component
      vEastward = vectors[0]['data'].copy()
      
      # copy 'colatitudinal' component of directional vector to polar r component
      vSouthward = vectors[1]['data'].copy()
      
      # if a flipped coordinate system (e.g., MIX southern hemisphere), change
      # signs of vector components
      if flipHemi:
         vEastward = -vEastward
         vSouthward = -vSouthward
      
      
      if sm2geo:
         x,y,z,dx,dy,dz = pyLTR.transform.SPHtoCAR(longGridOrig,colatGridOrig,1,
                                                   vEastward,vSouthward,0)
         x,y,z = pyLTR.transform.SMtoGEO(x,y,z,dtUTC)
         dx,dy,dz = pyLTR.transform.SMtoGEO(dx,dy,dz,dtUTC)
         _, _, _, vEastward, vSouthward, _ = pyLTR.transform.CARtoSPH(x,y,z,dx,dy,dz)
      
      
      
      # PROCESS PLOT OPTIONS FOR SECOND (VECTOR) INPUT
      if plotOpts2 == None:
         plotOpts2 = {}
      
      # use scale if given, otherwise max. magnitude generates arrow 1/10 plot-width
      if 'scale' in plotOpts2:
         scale = plotOpts2['scale']
      else:
         scale = np.sqrt(vEastward**2 + vSouthward**2).max() / 0.1
      
      # use width if given, otherwise default is .0025 plot-width
      if 'width' in plotOpts2:
         width = plotOpts2['width']
      else:
         width = .0025
      
      # use pivot if given, otherwise default is 'tail'
      if 'pivot' in plotOpts2:
         pivot = plotOpts2['pivot']
      else:
         pivot = 'tail'
      
      # use color if given, otherwise default is black
      if 'color' in plotOpts2:
         color2 = plotOpts2['color']
      else:
         color2 = 'k'
         
      
      units='width' # use plot width as basic unit for all quiver attributes
      q = quiverPolar(longGrid, colatGrid, vEastward, vSouthward, bm=bm, nx=40,
                      units=units, scale=scale, width=width, pivot=pivot, 
                      color=color2)         
      
      
      if plotQuiverKey:
         # Since we forced units to be 'width', a scale keyword equal to 1 implies 
         # that an arrow whose length is the full width of the plot will be equal 
         # to 1 data unit, a scale keyword of 2 implies that an arrow whose length 
         # is the full width of the plot will be equal to two data units, etc.
         # Here we draw a "Quiver Key" that is 1/10th the width of the plot, and 
         # properly scale its value...much pained thought went into convincing my-
         # self that this is correct, but feel free to verify -EJR 12/2013
         plt.quiverkey(q, .3, 0, .1*scale, 
                       ('%3.1e '+'%s') % (.1*scale,vectors[0]['units']), 
                       coordinates='axes',labelpos='S')

      
      # plot and label points on the map
      for point in points:
         # flip coordinates if necessary
         if flipHemi:
            longP = 360 - point[0] * 180/np.pi
            colatP = 180 - point[1] * 180/np.pi
         else:
            longP = point[0] * 180/np.pi
            colatP = point[1] * 180/np.pi
         
         
         if sm2geo:
            # convert coordinates back to radians
            longP = longP * np.pi/180
            colatP = colatP * np.pi/180
            
            # rotate from SM to GEO coordinates;
            x,y,z = pyLTR.transform.SPHtoCAR(longP,colatP,1)
            x,y,z = pyLTR.transform.SMtoGEO(x,y,z,dtUTC)
            longP, colatP, _ = pyLTR.transform.CARtoSPH(x,y,z)
                           
            longP = longP * 180/np.pi
            colatP = colatP * 180/np.pi
         
         
         # plot point (filled circle, with dot in middle)
         p1 = plotPolar(longP, colatP, bm=bm, alpha=1,
                        marker='o', ms=4, mec='black', mfc='white', mew=1,)
         p2 = plotPolar(longP, colatP, bm=bm, alpha=1,
                        marker='.', ms=1, mec='black', mfc='black', mew=1)
        
         # label point if string passed
         if len(point) > 2:
            labelP = point[2]
            x,y = bm(longP, 90-colatP)
            plt.text(x, y, labelP)

   
   # return bm
   return bm



def OverPlotDict(longitude, colatitude, scalars1, scalars2,
                 plotOpts1=None, plotOpts2=None, 
                 longTicks=None, longLabels=None, 
                 colatTicks=None, colatLabels=None,
                 northPOV=None, coordSystem=None, dtUTC=None,
                 plotColorBar=True, plotContourLabels=True,
                 useMesh=False,
                 points=[],
                 userAxes=None):
   """
   Produce a well-labeled polar plot of line contours of scalars2 field over top
   a filled contour plot of scalars1 field, all overtop a polar-POV map of the 
   Earth. This is a wrapper contourPolar() that is intended to be a drop-in 
   replacement for PolarPlot.OverPlotDict().
   
   Inputs
     - first is a dictionary holding a 2D meshgrid of longitudes
     - second is a dictionary holding a 2D meshgrid of colatitudes
     - third is dictionary holding a 2D array of a scalar field whose elements 
       are located at coordinates specified in the first and second arguments;
       as per MIX convention: dim1 = longitudes, dim2=colatitudes
     - fourth is dictionary holding a 2D array of a scalar field whose elements 
       are located at coordinates specified in the first and second arguments;
       as per MIX convention: dim1 = longitudes, dim2=colatitudes
          
   Keywords
     - plotOpts1  - dictionary holding various optional parameters for tweaking
                    the filled contour scalar field appearance
                    'colormap': string specifying a colormap for surface or 
                                contourf plots
                                FIXME: this should be an actual colormap object,
                                       NOT just the name of a standard colormap
                    'min': minimum data value mapped to colormap
                    'max': maximum data value mapped to colormap
                    'format_str': format string for min/max labels
                    'numContours': number of contours between min and max
                    'numTicks': number of ticks in colorbar
                    
     - plotOpts2  - dictionary holding various optional parameters for tweaking
                    the line contour scalar field appearance
                    'colormap': string specifying a colormap for surface or 
                                contourf plots
                                FIXME: this should be an actual colormap object,
                                       NOT just the name of a standard colormap
                    'colors': a recognized color, or list of colors
                    'min': minimum data value mapped to colormap
                    'max': maximum data value mapped to colormap
                    'format_str': format string for min/max labels
                    'numContours': number of contours between min and max
                    'numTicks': number of ticks in colorbar
     
     - longTicks  - where to place 'longitude' ticks in radians
     - longLabels - labels to place at longTicks
     - colatTicks - where to place 'colatiude' ticks in 'colatitude' units
     - colatLabels- labels to place at colatTicks
     
     - northPOV   - force a north-polar POV if True; a south polar-POV if False;
                    NOTE: unlike PolarPlot.QuiverPlotDict(), this keyword forces
                          an actual coordinate transformation, not adjustments
                          to the plot labels. So, if set to False, AND the bulk
                          of colatitudes appear to be in the north, these will
                          be rotated about the X axis to place them in the south,
                          and a southern hemisphere map will be generated. This
                          allows users to plot MIX southern hemisphere data
                          without having to rotate the data themselves. If set
                          to True, and colatitudes imply north-polar POV, no 
                          transformation will be performed.
                          (default=None; POV determined from colatitudes)
     
     - coordSystem- a string specifying the coordinate system of plotted data;
                    if not 'Geographic', all Basemap objects (e.g., meridians,
                    parallels, boundaries), which are usually assumed to be in
                    geographic coordinates, will be transformed into this
                    coordinate system.
                    (options: 'Geographic', ...need to add one or two...)
                    (default='Geographic'; i.e., no transform)
     
     - dtUTC      - datetime object in UTC (required for geographic to
                    geomagnetic transformations, and to ensure noon local-time
                    always points up).
                         
     - plotColorBar- if True, plot a colorbar for filled contours/mesh
     - plotContourLabels- if True, label line contours
     - useMesh    - if True, plot scalar field as surface plot, not filled contours
                    (should be quicker in theory, but seems to be much slower)
     - points     - a list of (lon,colat[,label]) sets that define and label 
                    points on the map; these should be in the same coordinate
                    system as longitude and colatitude.
     - userAxes   - Used to set ax kwarg in Basemap initialization.
     
   Outputs
     Reference to a Basemap object (?)
   
   """
   
   # extract coordinates from dictionaries
   longGrid = longitude['data'].copy() * 180/np.pi
   colatGrid = colatitude['data'].copy() * 180/np.pi
   
   if dtUTC==None:
      # if dtUTC keyword not passed, assume 0 longitude points to noon
      dtUTC = datetime.datetime.combine(datetime.date.today(), 
                                        datetime.time(12))
      warnings.warn('UTC assumed to be noon today')

   if coordSystem==None:
      # if coordSystem keyword not passed, assume geographic
      coordSystem='Geographic'
      warnings.warn('Geographic coordinate system assumed')
   
      

   # determine default hemisphere to plot
   # NOTE: this is a MIX-centric kludge, and may not generalize well
   ucolats = np.unique(colatGrid)
      # check if we are plotting northern hemisphere, or southern
   if (ucolats < 90).sum() > (ucolats > 90).sum():
      poleColat=0
   elif (ucolats < 90).sum() < (ucolats > 90).sum():
      poleColat=180
   else:
      warnings.warn('Hemisphere not obvious, assuming north')
      poleColat=0
   
   
   # reconcile northPOV and poleColat if they differ
   flipHemi=False
   if northPOV==None:
      pass
   elif northPOV and poleColat==0:
      pass
   elif not northPOV and poleColat==180:
      pass
   else:
      flipHemi = True # flip coordinates for, e.g., MIX southern hemisphere data
      longGrid = 360 - longGrid
      colatGrid = 180 - colatGrid
      poleColat = 180 - poleColat
   
   

   # ideally, coordSystem would simply get passed through to basePolar(), but
   # since basePolar() cannot yet handle anything but Geographic coordinates,
   # we must transform non-Geographic coordinates into Geographic coordinates,
   # then change coordSystem to 'Geographic', before calling basePolar().
   sm2geo=False
   if coordSystem == 'Geographic':
      pass
   elif coordSystem == 'Solar Magnetic':
      sm2geo=True
      # convert coordinates back to radians
      longGrid = longGrid * np.pi/180
      colatGrid = colatGrid * np.pi/180
      
      # rotate ionospheric contribution from SM to GEO coordinates; leave
      # position vectors unchanged for subsequent rotations
      x,y,z = pyLTR.transform.SPHtoCAR(longGrid,colatGrid,1)
      x,y,z = pyLTR.transform.SMtoGEO(x,y,z,dtUTC)
      longGrid, colatGrid, _ = pyLTR.transform.CARtoSPH(x,y,z)
      
      longGrid = longGrid * 180/np.pi
      colatGrid = colatGrid * 180/np.pi
      
      coordSystem = 'Geographic'
      
   else:
      print(('Unrecognized coordinate system '+coordSystem+' specified via kwarg'))
      raise Exception
   
   
   # generate Basemap
   bm = basePolar(dtUTC, coordSystem=coordSystem, coordColatTicks=poleColat,
                  ax=userAxes)
   
   
   
   
   # if scalars1 is False (or None, or empty, etc.), skip and proceed to quiver plot
   if scalars1:
      
      # PROCESS PLOT OPTIONS FOR FIRST (SCALAR) INPUT
      
      if plotOpts1 == None:
         plotOpts1 = {}
      
      # if colormap is supplied use it
      if 'colormap' in plotOpts1:
         cmap1=eval('plt.cm.'+plotOpts1['colormap'])
      else:
         cmap1=None # default is used

      # if limits are given use them, if not use the variables min/max values
      if 'min' in plotOpts1:
         lower1 = plotOpts1['min']
      else:
         lower1 = scalars1['data'].min()
      if 'max' in plotOpts1:
         upper1 = plotOpts1['max']
      else:
         upper1 = scalars1['data'].max()
      # if format string for max/min is given use otherwise do default
      if 'format_str' in plotOpts1:
         format_str1 = plotOpts1['format_str']
      else:
         format_str1='%.2f'

      # if number of contours is given use it otherwise do 51
      if 'numContours' in plotOpts1:
         ncontours1 = plotOpts1['numContours']
      else:
         ncontours1 = 31

      # if number of ticks is given use it otherwise do 51
      if 'numTicks' in plotOpts1:
         nticks1 = plotOpts1['numTicks']
      else:
         nticks1 = 11
      
      
      contours1 = np.linspace(lower1,upper1,ncontours1)
      ticks1 = np.linspace(lower1,upper1,nticks1)
      
      scalar1Grid = scalars1['data']
      
      
      if (useMesh):
         # should probably change keyword useMesh to usePcolor, since there is
         # pcolormesh() as an option
         c1 = pcolorPolar(longGrid, colatGrid, scalarGrid, bm=bm, 
                         cmap=cmap1, vmin=lower1, vmax=upper1, alpha=0.7,
                         edgecolors='face')
      else:
         # plot filled contours
         c1 = contourPolar(longGrid, colatGrid, scalar1Grid, contours1, bm=bm,
                           filled=True, extend='both', cmap=cmap1, alpha=0.7)
      
      
      if (plotColorBar):
         # NOTE: it *should not* be necessary to pass the mappable c1 to Basemap's
         #       colorbar function, but something is wrong with plt.gci(), such
         #       that it returns None when called from bm.colorbar(), causing
         #       an attribute error...this kluge short-circuits the whole issue,
         #       but matplotlib needs this to be fixed.
         cb = bm.colorbar(c1, ticks=ticks1)
         cb.set_label(scalars1['name']+' ['+scalars1['units']+']')
      
      
      plt.annotate(('min: '+format_str1+'\nmax: ' + format_str1) % 
                 (scalar1Grid.min() ,scalar1Grid.max()), 
                 (0.65, -.05),  xycoords='axes fraction',
                 annotation_clip=False)
      




    # if scalars2 is False (or None, or empty, etc.), skip quiver plot
   if scalars2:
      
      
      # PROCESS PLOT OPTIONS FOR FIRST (SCALAR) INPUT
      
      if plotOpts2 == None:
         plotOpts2 = {}
      
      # if colormap is supplied use it
      if 'colormap' in plotOpts2:
         cmap2=eval('plt.cm.'+plotOpts2['colormap'])
      else:
         cmap2=None # default is used
    
      # if colors is supplied use it
      if 'colors' in plotOpts2:
         colors2=plotOpts2['colors']
      else:
         colors2=None # default is used
      
      # can only have cmap or colors so if both use cmap
      if cmap2!=None and colors2!=None:
         colors2=None
      
      # if neither cmap or colors is supplied, default to black contours on overlay
      if cmap2==None and colors2==None:
         colors2='k'
      
      # if limits are given use them, if not use the variables min/max values
      if 'min' in plotOpts2:
         lower2 = plotOpts2['min']
      else:
         lower2 = scalars2['data'].min()
      if 'max' in plotOpts2:
         upper2 = plotOpts2['max']
      else:
         upper2 = scalars2['data'].max()
      # if format string for max/min is given use otherwise do default
      if 'format_str' in plotOpts2:
         format_str2 = plotOpts2['format_str']
      else:
         format_str2='%.2f'

      # if number of contours is given use it otherwise do 51
      if 'numContours' in plotOpts2:
         ncontours2 = plotOpts2['numContours']
      else:
         ncontours2 = 31

      # if number of ticks is given use it otherwise do 51
      if 'numTicks' in plotOpts2:
         nticks2 = plotOpts2['numTicks']
      else:
         nticks2 = 11
      
      
      contours2 = np.linspace(lower2,upper2,ncontours2)
      ticks2 = np.linspace(lower2,upper2,nticks2)
      
      scalar2Grid = scalars2['data']
      
      
      # plot line contours
      if colors2:
         c2 = contourPolar(longGrid, colatGrid, scalar2Grid, contours2, bm=bm,
                           filled=False, extend='both', colors=colors2)
      else:
         c2 = contourPolar(longGrid, colatGrid, scalar2Grid, contours2, bm=bm,
                           filled=False, extend='both', cmap=cmap2)
      
      if (plotContourLabels):
         plt.clabel(c2, inline=1, fontsize=10)
      
      
      plt.annotate(('min: '+format_str2+'\nmax: ' + format_str2) % 
                 (scalar2Grid.min() ,scalar2Grid.max()), 
                 (0.35, -.05),  xycoords='axes fraction',
                 annotation_clip=False, ha='right')

      # plot and label points on the map
      for point in points:
         # flip coordinates if necessary
         if flipHemi:
            longP = 360 - point[0] * 180/np.pi
            colatP = 180 - point[1] * 180/np.pi
         else:
            longP = point[0] * 180/np.pi
            colatP = point[1] * 180/np.pi
         
         if sm2geo:
            # convert coordinates back to radians
            longP = longP * np.pi/180
            colatP = colatP * np.pi/180
            
            # rotate from SM to GEO coordinates;
            x,y,z = pyLTR.transform.SPHtoCAR(longP,colatP,1)
            x,y,z = pyLTR.transform.SMtoGEO(x,y,z,dtUTC)
            longP, colatP, _ = pyLTR.transform.CARtoSPH(x,y,z)
                           
            longP = longP * 180/np.pi
            colatP = colatP * 180/np.pi
            
                  
         # plot point (filled circle, with dot in middle)
         p1 = plotPolar(longP, colatP, bm=bm, alpha=1,
                        marker='o', ms=4, mec='black', mfc='white', mew=1,)
         p2 = plotPolar(longP, colatP, bm=bm, alpha=1,
                        marker='.', ms=1, mec='black', mfc='black', mew=1)
        
         # label point if string passed
         if len(point) > 2:
            labelP = point[2]
            x,y = bm(longP, 90-colatP)
            plt.text(x, y, labelP)
   
   
   # return bm
   return bm




def BasicPlotDict(longitude, colatitude, scalars,
                  plotOpts=None,
                  longTicks=None, longLabels=None, 
                  colatTicks=None, colatLabels=None,
                  northPOV=None, coordSystem=None, dtUTC=None,
                  plotColorBar=True,
                  useMesh=False,
                  points=[],
                  userAxes=None):
   """
   Produce a well-labeled filled contour plot of scalar field all overtop a 
   polar-POV map of the Earth. This is a wrapper contourPolar() that is intended 
   to be a drop-in replacement for PolarPlot.BasicPlotDict().
   
   Inputs
     - first is a dictionary holding a 2D meshgrid of longitudes
     - second is a dictionary holding a 2D meshgrid of colatitudes
     - third is dictionary holding a 2D array of a scalar field whose elements 
       are located at coordinates specified in the first and second arguments;
       as per MIX convention: dim1 = longitudes, dim2=colatitudes
          
   Keywords
     - plotOpts   - dictionary holding various optional parameters for tweaking
                    the filled contour scalar field appearance
                    'colormap': string specifying a colormap for surface or 
                                contourf plots
                                FIXME: this should be an actual colormap object,
                                       NOT just the name of a standard colormap
                    'min': minimum data value mapped to colormap
                    'max': maximum data value mapped to colormap
                    'format_str': format string for min/max labels
                    'numContours': number of contours between min and max
                    'numTicks': number of ticks in colorbar
                    
     
     - longTicks  - where to place 'longitude' ticks in radians
     - longLabels - labels to place at longTicks
     - colatTicks - where to place 'colatiude' ticks in 'colatitude' units
     - colatLabels- labels to place at colatTicks
     
     - northPOV   - force a north-polar POV if True; a south polar-POV if False;
                    NOTE: unlike PolarPlot.QuiverPlotDict(), this keyword forces
                          an actual coordinate transformation, not adjustments
                          to the plot labels. So, if set to False, AND the bulk
                          of colatitudes appear to be in the north, these will
                          be rotated about the X axis to place them in the south,
                          and a southern hemisphere map will be generated. This
                          allows users to plot MIX southern hemisphere data
                          without having to rotate the data themselves. If set
                          to True, and colatitudes imply north-polar POV, no 
                          transformation will be performed.
                          (default=None; POV determined from colatitudes)
     
     - coordSystem- a string specifying the coordinate system of plotted data;
                    if not 'Geographic', all Basemap objects (e.g., meridians,
                    parallels, boundaries), which are usually assumed to be in
                    geographic coordinates, will be transformed into this
                    coordinate system.
                    (options: 'Geographic', ...need to add one or two...)
                    (default='Geographic'; i.e., no transform)
     
     - dtUTC      - datetime object in UTC (required for geographic to
                    geomagnetic transformations, and to ensure noon local-time
                    always points up).
                         
     - plotColorBar- if True, plot a colorbar for filled contours/mesh
     - useMesh    - if True, plot scalar field as surface plot, not filled contours
                    (should be quicker in theory, but seems to be much slower)
     - points     - a list of (lon,colat[,label]) sets that define and label 
                    points on the map; these should be in the same coordinate
                    system as longitude and colatitude.
     - userAxes   - Used to set ax kwarg in Basemap initialization.
     
   Outputs
     Reference to a Basemap object (?)
   
   """
   
   # extract coordinates from dictionaries
   longGrid = longitude['data'].copy() * 180/np.pi
   colatGrid = colatitude['data'].copy() * 180/np.pi
   
   
   if dtUTC==None:
      # if dtUTC keyword not passed, assume 0 longitude points to noon
      dtUTC = datetime.datetime.combine(datetime.date.today(), 
                                        datetime.time(12))
      warnings.warn('UTC assumed to be noon today')

   if coordSystem==None:
      # if coordSystem keyword not passed, assume geographic
      coordSystem='Geographic'
      warnings.warn('Geographic coordinate system assumed')
   
      

   # determine default hemisphere to plot
   # NOTE: this is a MIX-centric kludge, and may not generalize well
   ucolats = np.unique(colatGrid)
      # check if we are plotting northern hemisphere, or southern
   if (ucolats < 90).sum() > (ucolats > 90).sum():
      poleColat=0
   elif (ucolats < 90).sum() < (ucolats > 90).sum():
      poleColat=180
   else:
      warnings.warn('Hemisphere not obvious, assuming north')
      poleColat=0
   
   
   # reconcile northPOV and poleColat if they differ
   flipHemi=False
   if northPOV==None:
      pass
   elif northPOV and poleColat==0:
      pass
   elif not northPOV and poleColat==180:
      pass
   else:
      flipHemi = True # flip coordinates for, e.g., MIX southern hemisphere data
      longGrid = 360 - longGrid
      colatGrid = 180 - colatGrid
      poleColat = 180 - poleColat
   
   

   # ideally, coordSystem would simply get passed through to basePolar(), but
   # since basePolar() cannot yet handle anything but Geographic coordinates,
   # we must transform non-Geographic coordinates into Geographic coordinates,
   # then change coordSystem to 'Geographic', before calling basePolar().
   sm2geo=False
   if coordSystem == 'Geographic':
      pass
   elif coordSystem == 'Solar Magnetic':
      sm2geo=True
      
      # convert coordinates back to radians
      longGrid = longGrid * np.pi/180
      colatGrid = colatGrid * np.pi/180
      
      # rotate ionospheric contribution from SM to GEO coordinates; leave
      # position vectors unchanged for subsequent rotations
      x,y,z = pyLTR.transform.SPHtoCAR(longGrid,colatGrid,1)
      x,y,z = pyLTR.transform.SMtoGEO(x,y,z,dtUTC)
      longGrid, colatGrid, _ = pyLTR.transform.CARtoSPH(x,y,z)
      
      longGrid = longGrid * 180/np.pi
      colatGrid = colatGrid * 180/np.pi
      
      coordSystem = 'Geographic'
      
   else:
      print(('Unrecognized coordinate system '+coordSystem+' specified via kwarg'))
      raise Exception
   
   
   
   # generate Basemap
   bm = basePolar(dtUTC, coordSystem=coordSystem, coordColatTicks=poleColat,
                  ax=userAxes)
   
   
   
   
   # if scalars is False (or None, or empty, etc.), skip and proceed to quiver plot
   if scalars:
      
      # PROCESS PLOT OPTIONS FOR FIRST (SCALAR) INPUT
      
      if plotOpts == None:
         plotOpts = {}
      
      # if colormap is supplied use it
      if 'colormap' in plotOpts:
         cmap=eval('plt.cm.'+plotOpts['colormap'])
      else:
         cmap=None # default is used

      # if limits are given use them, if not use the variables min/max values
      if 'min' in plotOpts:
         lower = plotOpts['min']
      else:
         lower = scalars['data'].min()
      if 'max' in plotOpts:
         upper = plotOpts['max']
      else:
         upper = scalars['data'].max()
      # if format string for max/min is given use otherwise do default
      if 'format_str' in plotOpts:
         format_str = plotOpts['format_str']
      else:
         format_str='%.2f'

      # if number of contours is given use it otherwise do 51
      if 'numContours' in plotOpts:
         ncontours = plotOpts['numContours']
      else:
         ncontours = 31

      # if number of ticks is given use it otherwise do 51
      if 'numTicks' in plotOpts:
         nticks = plotOpts['numTicks']
      else:
         nticks = 11
      
      
      contours = np.linspace(lower,upper,ncontours)
      ticks = np.linspace(lower,upper,nticks)
      
      scalarGrid = scalars['data']
      
      
      if (useMesh):
         # should probably change keyword useMesh to usePcolor, since there is
         # pcolormesh() as an option
         c = pcolorPolar(longGrid, colatGrid, scalarGrid, bm=bm, 
                         cmap=cmap, vmin=lower, vmax=upper, alpha=0.7,
                         edgecolors='face')
      else:
         # plot filled contours
         c = contourPolar(longGrid, colatGrid, scalarGrid, contours, bm=bm,
                          filled=True, extend='both', cmap=cmap, alpha=0.7)
      
      
      if (plotColorBar):
         # NOTE: it *should not* be necessary to pass the mappable c to Basemap's
         #       colorbar function, but something is wrong with plt.gci(), such
         #       that it returns None when called from bm.colorbar(), causing
         #       an attribute error...this kluge short-circuits the whole issue,
         #       but matplotlib needs this to be fixed.
         cb = bm.colorbar(c, ticks=ticks)
         cb.set_label(scalars['name']+' ['+scalars['units']+']')
      
      
      plt.annotate(('min: '+format_str+'\nmax: ' + format_str) % 
                 (scalarGrid.min() ,scalarGrid.max()), 
                 (0.65, -.05),  xycoords='axes fraction',
                 annotation_clip=False)
      
      
      # plot and label points on the map
      for point in points:
         # flip coordinates if necessary
         if flipHemi:
            longP = 360 - point[0] * 180/np.pi
            colatP = 180 - point[1] * 180/np.pi
         else:
            longP = point[0] * 180/np.pi
            colatP = point[1] * 180/np.pi
         
         if sm2geo:
            # convert coordinates back to radians
            longP = longP * np.pi/180
            colatP = colatP * np.pi/180
            
            # rotate from SM to GEO coordinates;
            x,y,z = pyLTR.transform.SPHtoCAR(longP,colatP,1)
            x,y,z = pyLTR.transform.SMtoGEO(x,y,z,dtUTC)
            longP, colatP, _ = pyLTR.transform.CARtoSPH(x,y,z)
                           
            longP = longP * 180/np.pi
            colatP = colatP * 180/np.pi
            
                  
         # plot point (filled circle, with dot in middle)
         p1 = plotPolar(longP, colatP, bm=bm, alpha=1,
                        marker='o', ms=4, mec='black', mfc='white', mew=1,)
         p2 = plotPolar(longP, colatP, bm=bm, alpha=1,
                        marker='.', ms=1, mec='black', mfc='black', mew=1)
        
         # label point if string passed
         if len(point) > 2:
            labelP = point[2]
            x,y = bm(longP, 90-colatP)
            plt.text(x, y, labelP)
         
         
   # return bm
   return bm




