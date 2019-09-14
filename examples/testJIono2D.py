import pyLTR
import numpy as n
import pylab as p

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
x_MIX = dMIX.read('Grid X', tMIX[0])
x_MIX_dict = {'data':x_MIX*ri,'name':r'X','units':r'm'}

y_MIX = dMIX.read('Grid Y', tMIX[0])
y_MIX_dict = {'data':y_MIX*ri,'name':r'Y','units':r'm'}

psi_MIX = dMIX.read('Potential North [V]', tMIX[0])
psi_MIX_dict = {'data':psi_MIX*1e-3,'name':r'$\psi$','units':r'kV'}

sigmap_MIX = dMIX.read('Pedersen conductance North [S]', tMIX[0])
sigmap_MIX_dict = {'data':sigmap_MIX,'name':r'$\Sigma_P$','units':r'S'}

sigmah_MIX = dMIX.read('Hall conductance North [S]', tMIX[0])
sigmah_MIX_dict = {'data':sigmah_MIX,'name':r'$\Sigma_H$','units':r'S'}

fac_MIX = dMIX.read('FAC North [A/m^2]', tMIX[0])
fac_MIX_dict = {'data':fac_MIX*1e6,'name':r'$J_\parallel$','units':r'$\mu A/m^2$'}


# compute the electric field vectors from a northern POV
((phi_MIX_dict,theta_MIX_dict),
 (ephi_MIX_dict,etheta_MIX_dict)) = pyLTR.Physics.MIXCalcs.efieldDict(
                                     x_MIX_dict,y_MIX_dict,psi_MIX_dict,ri=6500e3,sm=True)

# plot electric field vectors over the potential field
p.figure(figsize=(8,6))
ax = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                             psi_MIX_dict, (ephi_MIX_dict, etheta_MIX_dict))


# compute current density vectors
((Jphi_MIX_dict,Jtheta_MIX_dict),
 (Jpedphi_MIX_dict,Jpedtheta_MIX_dict),
 (Jhallphi_MIX_dict,Jhalltheta_MIX_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                                             (ephi_MIX_dict,etheta_MIX_dict),
                                             sigmap_MIX_dict, sigmah_MIX_dict)

# plot Pedersen and Hall horizontal current density vectors over potential field
p.figure(figsize=(16,6))

ax1 = p.subplot(121,polar=True)
ax1 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                              psi_MIX_dict, (Jpedphi_MIX_dict, Jpedtheta_MIX_dict),
                                              plotOpts1={'min':-100,'max':100},
                                              plotOpts2={'scale':5e6},
                                              userAxes=ax1)

ax2 = p.subplot(122,polar=True)
ax2 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                              psi_MIX_dict, (Jhallphi_MIX_dict, Jhalltheta_MIX_dict),
                                              plotOpts1={'min':-100,'max':100},
                                              plotOpts2={'scale':5e6},
                                              userAxes=ax2)

# plot total horizontal current density vectors over vertical current density 
# (i.e., -FAC; note the difference in magnitude...this is expected because the
#  horizontal vectors are partially integrated densities, while FACs are not)
# NOTE: we zoom in here to show how the vectors are not actually drawn at the
#       locations specified in the input arrays, but are instead interpolated
#       to a new grid that is less dense toward the center.
p.figure(figsize=(24,6))

ax1 = p.subplot(131,polar=True)
ax1 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                             fac_MIX_dict, (Jphi_MIX_dict, Jtheta_MIX_dict),
                                             plotOpts1={'min':-0.5,'max':0.5},
                                             plotOpts2={'width':.005,
                                                        'scale':5e6,
                                                        'pivot':'middle',
                                                        'color':'blue'},
                                             userAxes=ax1)

ax2 = p.subplot(132,polar=True)
ax2 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                             fac_MIX_dict, (Jphi_MIX_dict, Jtheta_MIX_dict),
                                             plotOpts1={'min':-0.5,'max':0.5},
                                             plotOpts2={'width':.005,
                                                        'scale':5e6,
                                                        'pivot':'middle',
                                                        'color':'blue'},
                                             userAxes=ax2)
ax2.set_rlim((0,.2))

ax3 = p.subplot(133,polar=True)
ax3 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                             fac_MIX_dict, (Jphi_MIX_dict, Jtheta_MIX_dict),
                                             plotOpts1={'min':-0.5,'max':0.5},
                                             plotOpts2={'width':.005,
                                                        'scale':5e6,
                                                        'pivot':'middle',
                                                        'color':'blue'},
                                             userAxes=ax3)
ax3.set_rlim((0,.02))





# read in some southern hemisphere data from first file, create dictionaries
# with data values rescaled to supposedly typical units
ri=6500e3
x_MIX = dMIX.read('Grid X', tMIX[0])
x_MIX_dict = {'data':x_MIX*ri,'name':r'X','units':r'm'}

y_MIX = dMIX.read('Grid Y', tMIX[0])
y_MIX_dict = {'data':y_MIX*ri,'name':r'Y','units':r'm'}

psi_MIX = dMIX.read('Potential South [V]', tMIX[0])
psi_MIX_dict = {'data':psi_MIX*1e-3,'name':r'$\psi$','units':r'kV'}

sigmap_MIX = dMIX.read('Pedersen conductance South [S]', tMIX[0])
sigmap_MIX_dict = {'data':sigmap_MIX,'name':r'$\Sigma_P$','units':r'S'}

sigmah_MIX = dMIX.read('Hall conductance South [S]', tMIX[0])
sigmah_MIX_dict = {'data':sigmah_MIX,'name':r'$\Sigma_H$','units':r'S'}

fac_MIX = dMIX.read('FAC South [A/m^2]', tMIX[0])
fac_MIX_dict = {'data':fac_MIX*1e6,'name':r'$J_\parallel$','units':r'$\mu A/m^2$'}


# compute the electric field vectors from a northern POV
((phi_MIX_dict,theta_MIX_dict),
 (ephi_MIX_dict,etheta_MIX_dict)) = pyLTR.Physics.MIXCalcs.efieldDict(
                                     x_MIX_dict,y_MIX_dict,psi_MIX_dict,ri=6500e3,sm=False)

# plot electric field vectors over the potential field
p.figure(figsize=(8,6))
ax = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                             psi_MIX_dict, (ephi_MIX_dict, etheta_MIX_dict))


# compute current dentsity vectors
((Jphi_MIX_dict,Jtheta_MIX_dict),
 (Jpedphi_MIX_dict,Jpedtheta_MIX_dict),
 (Jhallphi_MIX_dict,Jhalltheta_MIX_dict)) = pyLTR.Physics.MIXCalcs.jphithetaDict(
                                             (ephi_MIX_dict,etheta_MIX_dict),
                                             sigmap_MIX_dict, sigmah_MIX_dict)

# plot Pedersen and Hall horizontal current density vectors over potential field
p.figure(figsize=(16,6))

ax1 = p.subplot(121,polar=True)
ax1 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                              psi_MIX_dict, (Jpedphi_MIX_dict, Jpedtheta_MIX_dict),
                                              plotOpts1={'min':-100,'max':100},
                                              plotOpts2={'scale':5e6},
                                              userAxes=ax1)

ax2 = p.subplot(122,polar=True)
ax2 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                              psi_MIX_dict, (Jhallphi_MIX_dict, Jhalltheta_MIX_dict),
                                              plotOpts1={'min':-100,'max':100},
                                              plotOpts2={'scale':5e6},
                                              userAxes=ax2)

# plot total horizontal current density vectors over vertical current density 
# (i.e., -FAC; note the difference in magnitude...this is expected because the
#  horizontal vectors are partially integrated densities, while FACs are not)
# NOTE: we zoom in here to show how the vectors are not actually drawn at the
#       locations specified in the input arrays, but are instead interpolated
#       to a new grid that is less dense toward the center.
p.figure(figsize=(24,6))

ax1 = p.subplot(131,polar=True)
ax1 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                             fac_MIX_dict, (Jphi_MIX_dict, Jtheta_MIX_dict),
                                             plotOpts1={'min':-0.5,'max':0.5},
                                             plotOpts2={'width':.005,
                                                        'scale':5e6,
                                                        'pivot':'middle',
                                                        'color':'blue'},
                                             userAxes=ax1)

ax2 = p.subplot(132,polar=True)
ax2 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                             fac_MIX_dict, (Jphi_MIX_dict, Jtheta_MIX_dict),
                                             plotOpts1={'min':-0.5,'max':0.5},
                                             plotOpts2={'width':.005,
                                                        'scale':5e6,
                                                        'pivot':'middle',
                                                        'color':'blue'},
                                             userAxes=ax2)
ax2.set_rlim((0,.2))

ax3 = p.subplot(133,polar=True)
ax3 = pyLTR.Graphics.PolarPlot.QuiverPlotDict(phi_MIX_dict, theta_MIX_dict, 
                                             fac_MIX_dict, (Jphi_MIX_dict, Jtheta_MIX_dict),
                                             plotOpts1={'min':-0.5,'max':0.5},
                                             plotOpts2={'width':.005,
                                                        'scale':5e6,
                                                        'pivot':'middle',
                                                        'color':'blue'},
                                             userAxes=ax3)
ax3.set_rlim((0,.02))

p.show()

