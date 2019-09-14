import pyLTR
import numpy as n
import configparser
import os

config = configparser.ConfigParser()
config.read('/Users/wiltbemj/Dropbox/Python/LFM-Startup/startup.config')

ni = config.getint('Dimensions','NI')
nj = config.getint('Dimensions','NJ')
nk = config.getint('Dimensions','NK')
fileName = config.get('OutputFileName','Prefix')+'_%dx%dx%d_mhd_0000000.hdf'%(ni,nj,nk)

xscale = config.getfloat('Normalization','Xscale')
rmin = config.getfloat('GridSpecs','RMIN')*xscale
rmax = config.getfloat('GridSpecs','RMAX')*xscale
thetamin = config.getfloat('GridSpecs','THETAMIN')

monopole = config.getboolean('Magnetic polarity','Monopole')

sg = pyLTR.Grids.SphereGrid((53,48,64))
(p,t,r) = sg.ptrEdge(rmin=rmin,rmax=rmax,thetamin=thetamin)
(pc,tc,rc) = sg.ptrCenter()
(x,y,z) = sg.xyzEdge()

path = '/Users/wiltbemj/mhd_data'
lfmh = pyLTR.Tools.lfmstartup.lfmstartup(os.path.join(path,fileName),(ni,nj,nk))
lfmh.open(tzero=0.0)
lfmh.writeVar('X_grid',x)
lfmh.writeVar('Y_grid',y)
lfmh.writeVar('Z_grid',z)
lfmh.writeVar('rho_',400.*1.67e-24*n.ones((nk+1,nj+1,ni+1)))
lfmh.writeVar('c_',  5.e6*n.ones((nk+1,nj+1,ni+1)))
lfmh.writeVar('vx_',4.e7*n.sin(tc)*n.cos(pc))
lfmh.writeVar('vy_',4.e7*n.sin(pc)*n.sin(pc))
lfmh.writeVar('vz_',4.e7*n.cos(tc))
dtheta = t[0,1,0]-t[0,0,0]
dphi   = p[1,0,0]-p[0,0,0]
# note the fancy dstack to fill in the last i-face with correct bi values
bi = 0.002*rmin**2*n.sin(n.dstack((tc,tc[:,:,0])))*dtheta*dphi
lfmh.writeVar('bi_',bi)
lfmh.close()

