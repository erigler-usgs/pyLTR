import pyLTR
import numpy as n
import pylab as p
import time



# Generate Bostrom type 1 current sheet loop similar to Kisabeth and 
# Rostoker (1977); discretize to half-degree resolution grid
(phi_B1, 
 theta_B1, 
 rho_B1) = list(map(p.squeeze,p.meshgrid(p.linspace(170*p.pi/180.,190*p.pi/180.,41),
                                    p.linspace(20*p.pi/180.,25*p.pi/180,11),
                                    6500000, indexing='ij') ))
Iphi_B1 = -1e6 / 10 * p.ones((phi_B1.shape[0], phi_B1.shape[1]))
Itheta_B1 = 0 / 40 * p.ones((theta_B1.shape[0], theta_B1.shape[1]))


# mimimum positional boundaries
rion_min_B1 = [None] * 3 # initialize empty 3 list
rion_min_B1[0] = p.zeros(phi_B1.shape)
rion_min_B1[0][1:,:] = phi_B1[1:,:] - p.diff(phi_B1, axis=0)/2.
rion_min_B1[0][0,:] = phi_B1[0,:] - p.diff(phi_B1[0:2,:], axis=0).squeeze()/2.

rion_min_B1[1] = p.zeros(theta_B1.shape)
rion_min_B1[1][:,1:] = theta_B1[:,1:] - p.diff(theta_B1, axis=1)/2.
rion_min_B1[1][:,0] = theta_B1[:,0] - p.diff(theta_B1[:,0:2], axis=1).squeeze()/2.

rion_min_B1[2] = p.zeros(theta_B1.shape)
rion_min_B1[2][:,:] = 6500.e3

# maximum positional boundaries
rion_max_B1 = [None] * 3 # initialize empty 3 list
rion_max_B1[0] = p.zeros(phi_B1.shape)
rion_max_B1[0][:-1,:] = phi_B1[:-1,:] + p.diff(phi_B1, axis=0)/2.
rion_max_B1[0][-1,:] = phi_B1[-1,:] + p.diff(phi_B1[-2:,:], axis=0).squeeze()/2.

rion_max_B1[1] = p.zeros(theta_B1.shape)
rion_max_B1[1][:,:-1] = theta_B1[:,:-1] + p.diff(theta_B1, axis=1)/2.
rion_max_B1[1][:,-1] = theta_B1[:,-1] + p.diff(theta_B1[:,-2:], axis=1).squeeze()/2.

rion_max_B1[2] = p.zeros(theta_B1.shape)
rion_max_B1[2][:,:] = p.Inf


# calculate total SSECS from longitudinal ionospheric current
(rv_B1_total, 
 Jv_B1_total, 
 dv_B1_total) = pyLTR.Physics.SSECS.ssecs_sphere(rion_min_B1, 
                                                 rion_max_B1, 
                                                 (Iphi_B1, Itheta_B1), 
                                                 10, True)
# calculate ionospheric SSECS from longitudinal ionospheric current
(rv_B1_iono, 
 Jv_B1_iono, 
 dv_B1_iono) = pyLTR.Physics.SSECS.ssecs_sphere(rion_min_B1, 
                                                (rion_max_B1[0],
                                                 rion_max_B1[1],
                                                 rion_min_B1[2]), 
                                                (Iphi_B1, Itheta_B1), 
                                                10, True)
# calculate non-ionospheric SSECS from longitudinal ionospheric current
(rv_B1_faceq, 
 Jv_B1_faceq, 
 dv_B1_faceq) = pyLTR.Physics.SSECS.ssecs_sphere((rion_max_B1[0],
                                                  rion_max_B1[1],
                                                  rion_min_B1[2] + 1),
                                                 rion_min_B1, 
                                                 (Iphi_B1, Itheta_B1), 
                                                 10, True)


# calculate deltaBs for local grid; convert from natural spherical to HDZ coordinates
phi_obs, theta_obs, rho_obs = p.meshgrid(p.linspace(160,200,21) * p.pi/180,
                                         p.linspace(5,35,16) * p.pi/180, 
                                         6378000, indexing='ij')
tic=time.time()
(dBphi_B1_total,
 dBtheta_B1_total, 
 dBrho_B1_total) = pyLTR.Physics.BS.bs_sphere(rv_B1_total, 
                                              Jv_B1_total, 
                                              dv_B1_total, 
                                              (phi_obs, theta_obs, rho_obs) )

# calculate deltaB for ionospheric Bostrom type 1 SSECS 
(dBphi_B1_iono,
 dBtheta_B1_iono, 
 dBrho_B1_iono) = pyLTR.Physics.BS.bs_sphere(rv_B1_iono, 
                                             Jv_B1_iono, 
                                             dv_B1_iono, 
                                             (phi_obs, theta_obs, rho_obs) )

# calculate deltaB for non-ionospheric Bostrom type 1 SSECS 
(dBphi_B1_faceq,
 dBtheta_B1_faceq, 
 dBrho_B1_faceq) = pyLTR.Physics.BS.bs_sphere(rv_B1_faceq, 
                                              Jv_B1_faceq, 
                                              dv_B1_faceq, 
                                              (phi_obs, theta_obs, rho_obs) )
toc=time.time()
print(toc-tic,' seconds to process total, ionospheric, and non-ionospheric deltaB')


print(dBphi_B1_total.min(),dBphi_B1_total.max())
print(dBtheta_B1_total.min(),dBphi_B1_total.max())
print(dBrho_B1_total.min(),dBphi_B1_total.max())




# plots...
p.figure(figsize=(8,18))

ax1=p.subplot(311)
contours=p.linspace(-480,480,25)
#contours=p.append(p.linspace(-480,0,13), p.linspace(10,120,12)) #doesn't work as expected
p.contourf(phi_obs.squeeze() * 180/p.pi, 90. - theta_obs.squeeze() * 180/p.pi, -dBtheta_B1_total.squeeze() * 1e9,
           contours,extend='both')
p.title('H')
cb=p.colorbar()
cb.set_label('nT')


ax2=p.subplot(312)
contours=p.linspace(-240,240,25)
p.contourf(phi_obs.squeeze() * 180/p.pi, 90. - theta_obs.squeeze() * 180/p.pi, dBphi_B1_total.squeeze() * 1e9,
           contours,extend='both')
p.title('D')
cb=p.colorbar()
cb.set_label('nT')

ax3=p.subplot(313)
contours=p.linspace(-480,480,25)
p.contourf(phi_obs.squeeze() * 180/p.pi, 90. - theta_obs.squeeze() * 180/p.pi, -dBrho_B1_total.squeeze() * 1e9,
           contours,extend='both')
p.title('Z')
cb=p.colorbar()
cb.set_label('nT')




p.figure(figsize=(16,6))

ax1=p.subplot(121,polar=False)
contours=p.linspace(-480,480,25)
p.contourf(phi_obs.squeeze() * 180/p.pi, 90. - theta_obs.squeeze() * 180/p.pi, dBrho_B1_total.squeeze() * 1e9,
           contours,extend='both')
p.colorbar()

qh=p.quiver(phi_obs.squeeze()[::4,::3] * 180/p.pi, 90. - theta_obs.squeeze()[::4,::3] * 180/p.pi,
            dBphi_B1_total.squeeze()[::4,::3] * 1e9,
            -dBtheta_B1_total.squeeze()[::4,::3] * 1e9)

ax2=p.subplot(122,polar=True)
ax2=pyLTR.Graphics.PolarPlot.QuiverPlotDict({'data':phi_obs.squeeze()},
                                        {'data':theta_obs.squeeze()},
                                        {'data':dBrho_B1_total.squeeze() * 1e9,'name':r'','units':'nT'},
                                        None,
                                        plotOpts1={'min':-480,'max':480,'numContours':25,'numTicks':9},
                                        plotOpts2={'width':.001,'scale':5e3},
                                        userAxes=ax2)
ax2=pyLTR.Graphics.PolarPlot.QuiverPlotDict({'data':phi_obs.squeeze()[::4,::3]},
                                        {'data':theta_obs.squeeze()[::4,::3]},
                                        None,
                                        ({'data':dBphi_B1_total.squeeze()[::4,::3] * 1e9,'name':r'','units':'nT'},
                                         {'data':dBtheta_B1_total.squeeze()[::4,::3] * 1e9,'name':r'','units':'nT'}),
                                        plotOpts1={'min':-480,'max':480,'numContours':25,'numTicks':9},
                                        plotOpts2={'width':.001,'scale':5e3},
                                        userAxes=ax2)








# Generate Bostrom type 2 current sheet loop similar to Kisabeth and 
# Rostoker (1977); discretize to half-degree resolution grid
(phi_B2, 
 theta_B2, 
 rho_B2) = list(map(p.squeeze,p.meshgrid(p.linspace(178*p.pi/180.,182*p.pi/180.,9),
                                    p.linspace(20.5*p.pi/180.,24.5*p.pi/180,9),
                                    6500000, indexing='ij') ))
Iphi_B2 = 0 / 8 * p.ones((phi_B2.shape[0], phi_B2.shape[1]))
Itheta_B2 = 1e6 / 8 * p.ones((theta_B2.shape[0], theta_B2.shape[1]))



# mimimum positional boundaries
rion_min_B2 = [None] * 3 # initialize empty 3 list
rion_min_B2[0] = p.zeros(phi_B2.shape)
rion_min_B2[0][1:,:] = phi_B2[1:,:] - p.diff(phi_B2, axis=0)/2.
rion_min_B2[0][0,:] = phi_B2[0,:] - p.diff(phi_B2[0:2,:], axis=0).squeeze()/2.

rion_min_B2[1] = p.zeros(theta_B2.shape)
rion_min_B2[1][:,1:] = theta_B2[:,1:] - p.diff(theta_B2, axis=1)/2.
rion_min_B2[1][:,0] = theta_B2[:,0] - p.diff(theta_B2[:,0:2], axis=1).squeeze()/2.

rion_min_B2[2] = p.zeros(theta_B2.shape)
rion_min_B2[2][:,:] = 6500.e3

# maximum positional boundaries
rion_max_B2 = [None] * 3 # initialize empty 3 list
rion_max_B2[0] = p.zeros(phi_B2.shape)
rion_max_B2[0][:-1,:] = phi_B2[:-1,:] + p.diff(phi_B2, axis=0)/2.
rion_max_B2[0][-1,:] = phi_B2[-1,:] + p.diff(phi_B2[-2:,:], axis=0).squeeze()/2.

rion_max_B2[1] = p.zeros(theta_B2.shape)
rion_max_B2[1][:,:-1] = theta_B2[:,:-1] + p.diff(theta_B2, axis=1)/2.
rion_max_B2[1][:,-1] = theta_B2[:,-1] + p.diff(theta_B2[:,-2:], axis=1).squeeze()/2.

rion_max_B2[2] = p.zeros(theta_B2.shape)
rion_max_B2[2][:,:] = p.Inf


# calculate total SSECS from longitudinal ionospheric current
(rv_B2_total, 
 Jv_B2_total, 
 dv_B2_total) = pyLTR.Physics.SSECS.ssecs_sphere(rion_min_B2, 
                                                 rion_max_B2, 
                                                 (Iphi_B2, Itheta_B2), 
                                                 10, True)
# calculate ionospheric SSECS from longitudinal ionospheric current
(rv_B2_iono, 
 Jv_B2_iono, 
 dv_B2_iono) = pyLTR.Physics.SSECS.ssecs_sphere(rion_min_B2, 
                                                (rion_max_B2[0],
                                                 rion_max_B2[1],
                                                 rion_min_B2[2]), 
                                                (Iphi_B2, Itheta_B2), 
                                                10, True)
# calculate non-ionospheric SSECS from longitudinal ionospheric current
(rv_B2_faceq, 
 Jv_B2_faceq, 
 dv_B2_faceq) = pyLTR.Physics.SSECS.ssecs_sphere((rion_max_B2[0],
                                                  rion_max_B2[1],
                                                  rion_min_B2[2] + 1),
                                                 rion_min_B2, 
                                                 (Iphi_B2, Itheta_B2), 
                                                 10, True)


# calculate deltaBs for local grid; convert from natural spherical to HDZ coordinates
phi_obs, theta_obs, rho_obs = p.meshgrid(p.linspace(160,200,21) * p.pi/180,
                                         p.linspace(5,35,16) * p.pi/180, 
                                         6378000, indexing='ij')

(dBphi_B2_total,
 dBtheta_B2_total, 
 dBrho_B2_total) = pyLTR.Physics.BS.bs_sphere(rv_B2_total, 
                                              Jv_B2_total, 
                                              dv_B2_total, 
                                              (phi_obs, theta_obs, rho_obs) )

# calculate deltaB for ionospheric Bostrom type 1 SSECS 
(dBphi_B2_iono,
 dBtheta_B2_iono, 
 dBrho_B2_iono) = pyLTR.Physics.BS.bs_sphere(rv_B2_iono, 
                                             Jv_B2_iono, 
                                             dv_B2_iono, 
                                             (phi_obs, theta_obs, rho_obs) )

# calculate deltaB for non-ionospheric Bostrom type 1 SSECS 
(dBphi_B2_faceq,
 dBtheta_B2_faceq, 
 dBrho_B2_faceq) = pyLTR.Physics.BS.bs_sphere(rv_B2_faceq, 
                                              Jv_B2_faceq, 
                                              dv_B2_faceq, 
                                              (phi_obs, theta_obs, rho_obs) )


print(dBphi_B2_total.min(),dBphi_B2_total.max())
print(dBtheta_B2_total.min(),dBphi_B2_total.max())
print(dBrho_B2_total.min(),dBphi_B2_total.max())


# plots...
p.figure(figsize=(8,18))

ax1=p.subplot(311)
contours=p.linspace(-240,240,25)
#contours=p.append(p.linspace(-480,0,13), p.linspace(10,120,12)) #doesn't work as expected
p.contourf(phi_obs.squeeze() * 180/p.pi, 90. - theta_obs.squeeze() * 180/p.pi, -dBtheta_B2_total.squeeze() * 1e9,
           contours,extend='both')
p.title('H')
cb=p.colorbar()
cb.set_label('nT')


ax2=p.subplot(312)
contours=p.linspace(-600,600,25)
p.contourf(phi_obs.squeeze() * 180/p.pi, 90. - theta_obs.squeeze() * 180/p.pi, dBphi_B2_total.squeeze() * 1e9,
           contours,extend='both')
p.title('D')
cb=p.colorbar()
cb.set_label('nT')

ax3=p.subplot(313)
contours=p.linspace(-600,600,25)
p.contourf(phi_obs.squeeze() * 180/p.pi, 90. - theta_obs.squeeze() * 180/p.pi, -dBrho_B2_total.squeeze() * 1e9,
           contours,extend='both')
p.title('Z')
cb=p.colorbar()
cb.set_label('nT')




p.figure(figsize=(16,6))

ax1=p.subplot(121,polar=False)
contours=p.linspace(-600,600,25)
p.contourf(phi_obs.squeeze() * 180/p.pi, 90. - theta_obs.squeeze() * 180/p.pi, dBrho_B2_total.squeeze() * 1e9,
           contours,extend='both')
p.colorbar()

qh=p.quiver(phi_obs.squeeze()[::4,::3] * 180/p.pi, 90. - theta_obs.squeeze()[::4,::3] * 180/p.pi,
            dBphi_B2_total.squeeze()[::4,::3] * 1e9,
            -dBtheta_B2_total.squeeze()[::4,::3] * 1e9)

ax2=p.subplot(122,polar=True)
ax2=pyLTR.Graphics.PolarPlot.QuiverPlotDict({'data':phi_obs.squeeze()},
                                        {'data':theta_obs.squeeze()},
                                        {'data':dBrho_B2_total.squeeze() * 1e9,'name':r'','units':'nT'},
                                        None,
                                        plotOpts1={'min':-600,'max':600,'numContours':25,'numTicks':9},
                                        plotOpts2={'width':.001,'scale':5e3},
                                        userAxes=ax2)
ax2=pyLTR.Graphics.PolarPlot.QuiverPlotDict({'data':phi_obs.squeeze()[::4,::3]},
                                        {'data':theta_obs.squeeze()[::4,::3]},
                                        None,
                                        ({'data':dBphi_B2_total.squeeze()[::4,::3] * 1e9,'name':r'','units':'nT'},
                                         {'data':dBtheta_B2_total.squeeze()[::4,::3] * 1e9,'name':r'','units':'nT'}),
                                        plotOpts1={'min':-480,'max':480,'numContours':25,'numTicks':9},
                                        plotOpts2={'width':.001,'scale':5e3},
                                        userAxes=ax2)
p.show()





