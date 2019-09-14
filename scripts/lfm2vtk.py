from numpy import array,vstack,hstack,ascontiguousarray,zeros
import vtk
from vtk.util import numpy_support
import time
start = time.clock()

############################################################################################################
#lfmfile = '../PSI/results/mas2lfm_g105_scaled_correctcs_mhd_0030000.dmp'
lfmfile = '/Users/merkivg1/work/LFM-OCT/SNS/S10/SNS-Bz5-Vx200-N5-S10/SNS-Bz5-Vx200-N5-S10_mhd_0456000.dmp'
parallel = False
############################################################################################################


from pyhdf.SD import SD, SDC
f = SD(lfmfile,mode=SDC.READ)

vx = f.select('vx_').get()[:-1,:-1,:-1]
vy = f.select('vy_').get()[:-1,:-1,:-1]
vz = f.select('vz_').get()[:-1,:-1,:-1]
bx = f.select('bx_').get()[:-1,:-1,:-1]
by = f.select('by_').get()[:-1,:-1,:-1]
bz = f.select('bz_').get()[:-1,:-1,:-1]
rho = f.select('rho_').get()[:-1,:-1,:-1]
c = f.select('c_').get()[:-1,:-1,:-1]
x = f.select('X_grid').get()[:]
y = f.select('Y_grid').get()[:]
z = f.select('Z_grid').get()[:]

f.end()

nk,nj,ni = c.shape


# x=zeros((400,400,400))
# y=zeros((400,400,400))
# z=zeros((400,400,400))
# vx = zeros((399,399,399))
# vy = zeros((399,399,399))
# vz = zeros((399,399,399))
# bx = zeros((399,399,399))
# by = zeros((399,399,399))
# bz = zeros((399,399,399))
# rho = zeros((399,399,399))
# c = zeros((399,399,399))
# nk,nj,ni = c.shape

# The 400x400x400 grid takes ~90s total to convert, of which 80s are spent writing the file.

# VTK Stuff

# grid
print(('Before zip1',time.clock()-start))
coords = ascontiguousarray(vstack((x.ravel(),y.ravel(),z.ravel())).T)   # This is way faster than zip
print(('After zip1',time.clock()-start))
#array(zip(x.ravel(),y.ravel(),z.ravel()))
#vstack((x.ravel(),y.ravel(),z.ravel())).T   # This does not work with numpy_to_vtk that requires contiguous arrays
vtkCoords=numpy_support.numpy_to_vtk(coords)
grid=vtk.vtkStructuredGrid()
points=vtk.vtkPoints()
points.SetData(vtkCoords)
grid.SetDimensions(ni+1,nj+1,nk+1)
grid.SetPoints(points)
print(('After grid',time.clock()-start))

V = ascontiguousarray(vstack((vx.ravel(),vy.ravel(),vz.ravel())).T)  
#array(zip(vx.ravel(),vy.ravel(),vz.ravel()))
vtkV = numpy_support.numpy_to_vtk(V)
vtkV.SetName('Velocity')
grid.GetCellData().AddArray(vtkV)

B = ascontiguousarray(vstack((bx.ravel(),by.ravel(),bz.ravel())).T)  
#array(zip(bx.ravel(),by.ravel(),bz.ravel()))
vtkB = numpy_support.numpy_to_vtk(B)
vtkB.SetName('Magnetic field')
grid.GetCellData().AddArray(vtkB)

rho_ = rho.ravel()
vtkRho = numpy_support.numpy_to_vtk(rho_)
vtkRho.SetName('Density')
grid.GetCellData().AddArray(vtkRho)

c_ = c.ravel()
vtkC = numpy_support.numpy_to_vtk(c_)
vtkC.SetName('Density')
grid.GetCellData().AddArray(vtkC)

print(('Before sg init',time.clock()-start))

if parallel:
    sg=vtk.vtkXMLPStructuredGridWriter()
    sg.SetFileName(lfmfile+'.pvts')
    sg.SetNumberOfPieces(4)
    sg.SetStartPiece(0)
    sg.SetEndPiece(3)
else:
    sg=vtk.vtkXMLStructuredGridWriter()
    sg.SetFileName(lfmfile+'.vts')
sg.SetInput(grid)
sg.SetDataModeToBinary()
print(('Before write',time.clock()-start))
sg.Write()

print((time.clock()-start))
