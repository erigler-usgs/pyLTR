import pyLTR
import numpy as n
import pylab as p



# convert MIX ionospheric currents into an SSECS current system, then plot
# deltaB on Earth's surface


# create list of data objects for MIX output in current directory
try:
   dMIX = pyLTR.Models.MIX('./','')
except IndexError:
   # most likely cause of this error...just ignore
   print("No MIX output files in current directory")


# get time ranges for MIX data objects
try:
   tMIX = dMIX.getTimeRange()
   print(len(tMIX),"MIX outputs available between",min(tMIX),"and",max(tMIX))
except:
   print("Cannot determine MIX output time range")


# read in some northern hemisphere data from first file, create dictionaries
# with data values rescaled to supposedly typical units
ri=6500e3
x_MIX = dMIX.read('Grid X', tMIX[0])[::4,::3]
x_MIX_dict = {'data':x_MIX*ri,'name':r'X','units':r'm'}

y_MIX = dMIX.read('Grid Y', tMIX[0])[::4,::3]
y_MIX_dict = {'data':y_MIX*ri,'name':r'Y','units':r'm'}

psi_MIX = dMIX.read('Potential North [V]', tMIX[0])[::4,::3]
psi_MIX_dict = {'data':psi_MIX*1e-3,'name':r'$\psi$','units':r'kV'}

sigmap_MIX = dMIX.read('Pedersen conductance North [S]', tMIX[0])[::4,::3]
sigmap_MIX_dict = {'data':sigmap_MIX,'name':r'$\Sigma_P$','units':r'S'}

sigmah_MIX = dMIX.read('Hall conductance North [S]', tMIX[0])[::4,::3]
sigmah_MIX_dict = {'data':sigmah_MIX,'name':r'$\Sigma_H$','units':r'S'}


# compute the electric field vectors
((phi_MIX_dict,theta_MIX_dict),
 (ephi_MIX_dict,etheta_MIX_dict)) = pyLTR.Physics.MIXCalcs.efieldDict(
                                     x_MIX_dict,y_MIX_dict,psi_MIX_dict,ri=6500e3)

# compute 2D ionospheric current dentsity vectors
((Jphi_MIX_dict,Jtheta_MIX_dict),
 (Jpedphi_MIX_dict,Jpedtheta_MIX_dict),
 (Jhallphi_MIX_dict,Jhalltheta_MIX_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                                             (ephi_MIX_dict,etheta_MIX_dict),
                                             sigmap_MIX_dict, sigmah_MIX_dict)

phi_MIX = phi_MIX_dict['data']
theta_MIX=theta_MIX_dict['data']

# caclulate MIX grid cell boundaries in phi and theta
rion_min_MIX = [None] * 3 # initialize empty 3 list
rion_min_MIX[0] = p.zeros(phi_MIX.shape)
rion_min_MIX[0][1:,:] = phi_MIX[1:,:] - p.diff(phi_MIX, axis=0)/2.
rion_min_MIX[0][0,:] = phi_MIX[0,:] - p.diff(phi_MIX[0:2,:], axis=0).squeeze()/2.

rion_min_MIX[1] = p.zeros(theta_MIX.shape)
rion_min_MIX[1][:,1:] = theta_MIX[:,1:] - p.diff(theta_MIX, axis=1)/2.
rion_min_MIX[1][:,0] = theta_MIX[:,0] - p.diff(theta_MIX[:,0:2], axis=1).squeeze()/2.

rion_min_MIX[2] = p.zeros(theta_MIX.shape)
rion_min_MIX[2][:,:] = 6500.e3


rion_max_MIX = [None] * 3 # initialize empty 3 list
rion_max_MIX[0] = p.zeros(phi_MIX.shape)
rion_max_MIX[0][:-1,:] = phi_MIX[:-1,:] + p.diff(phi_MIX, axis=0)/2.
rion_max_MIX[0][-1,:] = phi_MIX[-1,:] + p.diff(phi_MIX[-2:,:], axis=0).squeeze()/2.

rion_max_MIX[1] = p.zeros(theta_MIX.shape)
rion_max_MIX[1][:,:-1] = theta_MIX[:,:-1] + p.diff(theta_MIX, axis=1)/2.
rion_max_MIX[1][:,-1] = theta_MIX[:,-1] + p.diff(theta_MIX[:,-2:], axis=1).squeeze()/2.

rion_max_MIX[2] = p.zeros(theta_MIX.shape)
rion_max_MIX[2][:,:] = p.Inf





# calculate total SSECS from MIX data
(rv_MIX_total, 
 Jv_MIX_total, 
 dv_MIX_total) = pyLTR.Physics.SSECS.ssecs_sphere(rion_min_MIX, 
                                                  rion_max_MIX, 
                                                  (Jphi_MIX_dict['data']/1e6, 
                                                   Jtheta_MIX_dict['data']/1e6), 
                                                  10, False)
# calculate ionospheric components of SSECS from MIX data
(rv_MIX_iono, 
 Jv_MIX_iono, 
 dv_MIX_iono) = pyLTR.Physics.SSECS.ssecs_sphere([rion_min_MIX[0],rion_min_MIX[1],rion_min_MIX[2]], 
                                                 [rion_max_MIX[0],rion_max_MIX[1],rion_min_MIX[2]], 
                                                 (Jphi_MIX_dict['data']/1e6, 
                                                  Jtheta_MIX_dict['data']/1e6), 
                                                 10, False)
# calculate non-ionospheric component of SSECS from MIX data
(rv_MIX_faceq, 
 Jv_MIX_faceq, 
 dv_MIX_faceq) = pyLTR.Physics.SSECS.ssecs_sphere([rion_min_MIX[0],rion_min_MIX[1],rion_min_MIX[2]+1], 
                                                  [rion_max_MIX[0],rion_max_MIX[1],rion_min_MIX[2]],
                                                  (Jphi_MIX_dict['data']/1e6, 
                                                   Jtheta_MIX_dict['data']/1e6), 
                                                  10, False)



# calculate deltaB for total SSECS 
(dBphi_MIX_total, 
 dBtheta_MIX_total, 
 dBrho_MIX_total) = pyLTR.Physics.BS.bs_sphere(rv_MIX_total, 
                                               Jv_MIX_total, 
                                               dv_MIX_total, 
                                               (phi_MIX,theta_MIX,p.tile(6378e3,phi_MIX.shape) ))
# convert into nanoTeslas
dBphi_MIX_total_dict = {'data':dBphi_MIX_total*1e9,'name':r'$dB_\phi$','units':r'nT'}
dBtheta_MIX_total_dict = {'data':dBtheta_MIX_total*1e9,'name':r'$dB_\theta$','units':r'nT'}
dBrho_MIX_total_dict = {'data':dBrho_MIX_total*1e9,'name':r'$dB_\rho$','units':r'nT'}


# calculate deltaB for ionospheric SSECS 
(dBphi_MIX_iono, 
 dBtheta_MIX_iono, 
 dBrho_MIX_iono) = pyLTR.Physics.BS.bs_sphere(rv_MIX_iono, 
                                              Jv_MIX_iono, 
                                              dv_MIX_iono, 
                                              (phi_MIX,theta_MIX,p.tile(6378e3,phi_MIX.shape) ))
# convert into nanoTeslas
dBphi_MIX_iono_dict = {'data':dBphi_MIX_iono*1e9,'name':r'$dB_\phi$','units':r'nT'}
dBtheta_MIX_iono_dict = {'data':dBtheta_MIX_iono*1e9,'name':r'$dB_\theta$','units':r'nT'}
dBrho_MIX_iono_dict = {'data':dBrho_MIX_iono*1e9,'name':r'$dB_\rho$','units':r'nT'}


# calculate deltaB for FACEQ SSECS 
(dBphi_MIX_faceq, 
 dBtheta_MIX_faceq, 
 dBrho_MIX_faceq) = pyLTR.Physics.BS.bs_sphere(rv_MIX_faceq, 
                                               Jv_MIX_faceq, 
                                               dv_MIX_faceq, 
                                               (phi_MIX,theta_MIX,p.tile(6378e3,phi_MIX.shape) ))
# convert into nanoTeslas
dBphi_MIX_faceq_dict = {'data':dBphi_MIX_faceq*1e9,'name':r'$dB_\phi$','units':r'nT'}
dBtheta_MIX_faceq_dict = {'data':dBtheta_MIX_faceq*1e9,'name':r'$dB_\theta$','units':r'nT'}
dBrho_MIX_faceq_dict = {'data':dBrho_MIX_faceq*1e9,'name':r'$dB_\rho$','units':r'nT'}



# Plot deltaBs on the Earth's surface
p.figure(figsize=(24,6))

# plot total deltaB on Earth's surface
ax1 = p.subplot(131,polar=True)
ax1=pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                            dBrho_MIX_total_dict, 
                                            (dBphi_MIX_total_dict,dBtheta_MIX_total_dict), 
                                            plotOpts1={'min':-120,'max':120}, plotOpts2={'scale':1.2e3},
                                            userAxes=ax1)


# plot ionospheric-induced deltaB on Earth's surface
ax2 = p.subplot(132,polar=True)
ax2=pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                            dBrho_MIX_iono_dict, 
                                            (dBphi_MIX_iono_dict,dBtheta_MIX_iono_dict), 
                                            plotOpts1={'min':-120,'max':120}, plotOpts2={'scale':1.2e3},
                                            userAxes=ax2)


# plot (mostly) field aligned current-induced deltaB on Earth's surface
ax3 = p.subplot(133,polar=True)
ax3=pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                            dBrho_MIX_faceq_dict, 
                                            (dBphi_MIX_faceq_dict,dBtheta_MIX_faceq_dict), 
                                            plotOpts1={'min':-30,'max':30}, plotOpts2={'scale':1.2e3},
                                            userAxes=ax3)

p.show()
