import numpy as n
import pylab as p
import pyhdf.SD as hdfsd

fname='/glade/scratch/wiltbemj/grids/Oct_mhd_2000-01-01T00-00-00Z.hdf'
hdf=hdfsd.SD(fname,hdfsd.SDC.READ)
gname='/glade/scratch/wiltbemj/grids/GRIDOUT-212x192x256.hdf'
ghdf=hdfsd.SD(gname,hdfsd.SDC.WRITE|hdfsd.SDC.CREATE)
rearth = 6.38e8
xgrid=hdf.select('X_grid').get()/rearth
ygrid=hdf.select('Y_grid').get()/rearth
zgrid=hdf.select('Z_grid').get()/rearth
d1=ghdf.create('X_grid',hdfsd.SDC.FLOAT32,xgrid.shape)
d1[:]=xgrid
d1.endaccess()
d1=ghdf.create('Y_grid',hdfsd.SDC.FLOAT32,xgrid.shape)
d1[:]=ygrid
d1.endaccess()
d1=ghdf.create('Z_grid',hdfsd.SDC.FLOAT32,xgrid.shape)
d1[:]=zgrid
d1.endaccess()
ghdf.end()



