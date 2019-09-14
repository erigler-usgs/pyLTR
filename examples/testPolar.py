import pyLTR
import numpy as n
import pylab as p
if __name__ == '__main__':
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
  y=data.read('Grid Y', timeRange[index0])
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
  vals=data.read('Potential South [V]',timeRange[index0])/1000.0
  psi_s={'data':vals,'name':r'$\Phi_S$','units':r'kV'}
  p.figure()
  #pyLTR.Graphics.PolarPlot.BasicPlotDict(theta,r,psi_n)
  pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,psi_n)
  p.figure(figsize=(28,5))
  plotOpts={'min':-50,'max':50}
  ax=p.subplot(121,polar=True)
  #pyLTR.Graphics.PolarPlot.BasicPlotDict(theta,r,psi_n,plotOpts,userAxes=ax)
  pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,psi_n,plotOpts,userAxes=ax)
  ax=p.subplot(122,polar=True)
  #pyLTR.Graphics.PolarPlot.BasicPlotDict(theta,r,psi_s,plotOpts,userAxes=ax)
  pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,psi_s,plotOpts,userAxes=ax)
  p.figure()
  plotOpts={'min':-50,'max':50,'colormap':'RdBu_r','numContours':11,'numTicks':5}
  #pyLTR.Graphics.PolarPlot.BasicPlotDict(theta,r,psi_n,plotOpts)
  pyLTR.Graphics.PolarPlot.BasicPlotDict(longitude,colatitude,psi_n,plotOpts)
  p.show()
