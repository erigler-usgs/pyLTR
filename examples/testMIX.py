import pyLTR
import numpy as n
import pylab as p
#path='/hao/aim/wiltbemj/mhd_data/Thanasis/Event7/'
path='./'
#runname='Thanasis-Event7'
runname=''
pyLTR.Models.MIX(path,runname)
data=pyLTR.Models.MIX(path,runname)
data.getVarNames()
data.getTimeRange()
timeRange=data.getTimeRange()
index0=0
x=data.read('Grid X', timeRange[index0])
#xdict={'data':x,'name':'X','units':r'm'} # MIX grid is in units of ri
xdict={'data':x*6500e3,'name':'X','units':r'm'}
y=data.read('Grid Y', timeRange[index0])
#ydict={'data':y,'name':'Y','units':r'm'} # MIX grid is in units of ri
ydict={'data':y*6500e3,'name':'Y','units':r'm'}
theta=n.arctan2(y,x)
theta[theta<0]=theta[theta<0]+2*n.pi
## plotting routines now rotate local noon to point up
#theta=theta+n.pi/2 # to put noon up
r=n.sqrt(x**2+y**2)
## plotting routines now expect longitude and colatitude, in radians, stored in dictionaries
longitude = {'data':theta,'name':r'\phi','units':r'rad'}
colatitude = {'data':n.arcsin(r),'name':r'\theta','units':r'rad'}
vals=data.read('Potential North [V]',timeRange[index0])/1000.0
psi_n={'data':vals,'name':r'$\Phi_N$','units':r'kV'}
vals=data.read('Pedersen conductance South [S]',timeRange[index0])
sigmap_n={'data':vals,'name':r'$\Sigma_P$','units':'S'}
## efield function now calculates gradient using central differences,
## thus returning values on the original grid; also, it expects x,y to
## have units consistent with ri; finally, it relies on the user to pass
## inputs that will give the desired output units (e.g., psi should be in
## mV if the user wants mV/m returned)
#angles_mid,efield=pyLTR.Physics.MIXCalcs.efield(x,y,psi_n['data'],6500.e3)
angles,efield=pyLTR.Physics.MIXCalcs.efield(x*6500.e3,y*6500.e3,psi_n['data']*1e6,6500.e3)
ephi_n={'data':efield[0],'name':r'$E_\phi$','units':r'mV/m'}
etheta_n={'data':efield[1],'name':r'$E_\theta$','units':r'mV/m'}
## efieldDict function expects psi inputs in kV, then scales the output generate mV/m
## (this is now generally true that all *Dict functions will check and rescale ins/outs)
#anglesdict,efielddict=pyLTR.Physics.MIXCalcs.efieldDict(xdict,ydict,psi_n,6500.e3)
anglesdict,efielddict=pyLTR.Physics.MIXCalcs.efieldDict(xdict,ydict,psi_n,6500.e3)
jhdict=pyLTR.Physics.MIXCalcs.jouleDict(efielddict,sigmap_n)
jpeddict=pyLTR.Physics.MIXCalcs.jpedDict(efielddict,sigmap_n)
p.figure()
## plotting routines now expect longitude,colatitude input coordinates as dictionaries
#pyLTR.Graphics.PolarPlot.BasicPlotDict(theta,r,efielddict[0])
pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,efielddict[0])
p.figure()
#pyLTR.Graphics.PolarPlot.BasicPlotDict(theta,r,jhdict)
pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,jhdict)
p.figure()
#pyLTR.Graphics.PolarPlot.BasicPlotDict(theta,r,jpeddict)
pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,jpeddict)
p.show()


