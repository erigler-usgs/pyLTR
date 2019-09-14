import numpy as n
from . import HexahedralGrid

import unittest

class TestHexahedralGrid(unittest.TestCase):

    def setUp(self):
        x=n.empty((3,3,3))
        y=n.empty((3,3,3))
        z=n.empty((3,3,3))
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    x[i,j,k] = i
                    y[i,j,k] = j
                    z[i,j,k] = k
  
        self.grid = HexahedralGrid.HexahedralGrid(x,y,z)
              
    def test_volume(self):        
        volume = self.grid.cellVolume()
        self.assertEqual(volume.shape, (2,2,2))
        self.assertEqual(volume.sum(), 8)

        volume2 = self.grid.cellVolume()
        self.assertEqual(volume2.sum(),volume.sum())

    def test_cellCenters(self):
        (xc,yc,zc) = self.grid.cellCenters()
        self.assertEqual(xc.shape, (2,2,2))
        self.assertEqual(yc.sum(), 8)
        self.assertEqual(zc[1,1,1], 1.5)
                
    def test_faceVectors(self):
        (dxI,dyI,dzI,dxJ,dyJ,dzJ,dxK,dyK,dzK) = self.grid.faceVectors()
        self.assertEqual(dxI.shape,(2,2,2))

    def test_edgeLengths(self):
        (di,dj,dk) = self.grid.edgeLengths()
        self.assertEqual(di.shape,(2,3,3))
        self.assertEqual(dj.shape,(3,2,3))
        self.assertEqual(dk.shape,(3,3,2))               
        
if __name__ == '__main__':
    unittest.main()
