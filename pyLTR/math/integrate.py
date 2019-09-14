import numpy

def distance(p0, p1):
  """ Calculate the distance between p0 & p1--two points in R**3 """
  return( numpy.sqrt( (p0[0]-p1[0])**2 + 
                      (p0[1]-p1[1])**2 + 
                      (p0[2]-p1[2])**2 ) )

def calcFaceAreas(x,y,z):
  """ 
  Calculate the area of each face in a quad mesh.
 
  >>> calcFaceAreas(numpy.array([[ 0., 1.],[ 1., 0.]]), numpy.array([[ 0.,1.],[ 1.,0.]]), numpy.array([[ 0., 0.],[ 0.,0.]]))
  array([[ 2.]])
  """
  (nLonP1, nLatP1) = x.shape
  (nLon, nLat) = (nLonP1-1, nLatP1-1)

  area = numpy.zeros((nLon, nLat))

  for i in range(nLon):
    for j in range(nLat):
      left  = distance( (x[i,j],   y[i,j],   z[i,j]),   (x[i,j+1],   y[i,j+1],   z[i,j+1]) )
      right = distance( (x[i+1,j], y[i+1,j], z[i+1,j]), (x[i+1,j+1], y[i+1,j+1], z[i+1,j+1]) )
      top =   distance( (x[i,j+1], y[i,j+1], z[i,j+1]), (x[i+1,j+1], y[i+1,j+1], z[i+1,j+1]) )
      bot =   distance( (x[i,j],   y[i,j],   z[i,j]),   (x[i+1,j],   y[i+1,j],   z[i+1,j]) )
      
      area[i,j] = 0.5*(left+right) * 0.5*(top+bot)

  return area
