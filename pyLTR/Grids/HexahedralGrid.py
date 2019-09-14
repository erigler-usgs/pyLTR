import numpy as n

class HexahedralGrid(object):
    """
    Convenience methods for a hexahedral grid like the LFM.  Houses
    calculations for cell center locations, cell volume calculation,
    etc.
    """
    def __init__(self, x,y,z):
        """
        Parameters: 
          x,y,z: X,Y,Z grid positions. Must be logically orthogonal
            3-d arrays. Use Numpy arrays for best performance.
        """
        self.x = x
        self.y = y
        self.z = z

        # prefixing members & methods with double underscore "__"
        # denotes private member/method in Python.  See Python Style
        # Guide PEP 8 for more details:
        # http://www.python.org/dev/peps/pep-0008/
        
        # Cell Volume
        self.__isVolumeCalculated = False
        self.__volume = None
        
        # Cell Centers
        self.__isCellCentersCalculated = False
        self.__xCenter = None
        self.__yCenter = None
        self.__zCenter = None
        
        # Face Centers
        self.__isFaceCentersCalculated = False
        self.__xCenterFaceI = None
        self.__yCenterFaceI = None
        self.__zCenterFaceI = None
        self.__xCenterFaceJ = None
        self.__yCenterFaceJ = None
        self.__zCenterFaceJ = None
        self.__xCenterFaceK = None
        self.__yCenterFaceK = None
        self.__zCenterFaceK = None
        
        #Vector between Cell Faces
        self.__isFaceVectorsCalculated = False
        self.__dxFaceI = None
        self.__dyFaceI = None
        self.__dzFaceI = None
        self.__dxFaceJ = None
        self.__dyFaceJ = None
        self.__dzFaceJ = None
        self.__dxFaceK = None
        self.__dyFaceK = None
        self.__dzFaceK = None
        
        #Edge Lengths
        self.__isEdgeLengthsCalculated = False
        self.__dEdgeI = None
        self.__dEdgeJ = None
        self.__dEdgeK = None
        
        
        
    def cellVolume(self):
        """
        Returns 3d array storing the volume of each cell.
        """
        if self.__isVolumeCalculated:
            return self.__volume
            
        self.faceVectors()

        # Volume of parallelpiped is a triple product
        # http://en.wikipedia.org/wiki/Parallelepiped
        self.__volume = abs(
            self.__dxFaceI*(
                self.__dyFaceJ*self.__dzFaceK-self.__dyFaceK*self.__dzFaceJ)+
            self.__dyFaceI*(
                self.__dzFaceJ*self.__dxFaceK-self.__dzFaceK*self.__dxFaceJ)+
            self.__dzFaceI*(
                self.__dxFaceJ*self.__dyFaceK-self.__dxFaceK*self.__dyFaceJ))
                          
              
        self.__isVolumeCalculated = True
        return self.__volume
    
    def faceCenters(self):
        """
        Returns x,y,z location of i,j,k face centers
        """
        
        if self.__isFaceCentersCalculated:
            return (self.__xCenterFaceI,self.__yCenterFaceI,self.__zCenterFaceI,
                    self.__xCenterFaceJ,self.__yCenterFaceJ,self.__zCenterFaceJ,
                    self.__xCenterFaceK,self.__yCenterFaceK,self.__zCenterFaceK)
                    
    
        # the :-1 and 1: is used to accomplish j and j+1 addition
        self.__xCenterFaceI=0.25*(self.x[:,:-1,:-1]+self.x[:,1:,:-1]+
                              self.x[:,:-1,1:]+self.x[:,1:,1:])
        self.__xCenterFaceJ=0.25*(self.x[:-1,:,:-1]+self.x[1:,:,:-1]+
                              self.x[:-1,:,1:]+self.x[1:,:,1:])
        self.__xCenterFaceK=0.25*(self.x[:-1,:-1,:]+self.x[:-1,1:,:]+
                              self.x[1:,:-1,:]+self.x[1:,1:,:])

        self.__yCenterFaceI=0.25*(self.y[:,:-1,:-1]+self.y[:,1:,:-1]+
                              self.y[:,:-1,1:]+self.y[:,1:,1:])
        self.__yCenterFaceJ=0.25*(self.y[:-1,:,:-1]+self.y[1:,:,:-1]+
                              self.y[:-1,:,1:]+self.y[1:,:,1:])
        self.__yCenterFaceK=0.25*(self.y[:-1,:-1,:]+self.y[:-1,1:,:]+
                              self.y[1:,:-1,:]+self.y[1:,1:,:])

        self.__zCenterFaceI=0.25*(self.z[:,:-1,:-1]+self.z[:,1:,:-1]+
                              self.z[:,:-1,1:]+self.z[:,1:,1:])
        self.__zCenterFaceJ=0.25*(self.z[:-1,:,:-1]+self.z[1:,:,:-1]+
                              self.z[:-1,:,1:]+self.z[1:,:,1:])
        self.__zCenterFaceK=0.25*(self.z[:-1,:-1,:]+self.z[:-1,1:,:]+
                              self.z[1:,:-1,:]+self.z[1:,1:,:])
                              
        self.__isFaceCentersCalculated = True
        return (self.__xCenterFaceI,self.__yCenterFaceI,self.__zCenterFaceI,
                self.__xCenterFaceJ,self.__yCenterFaceJ,self.__zCenterFaceJ,
                self.__xCenterFaceK,self.__yCenterFaceK,self.__zCenterFaceK)
    
    def faceVectors(self):
        """
        Returns dx,dy,dz 3d vectors between the centers of i,j,k faces
        """
        
        if self.__isFaceVectorsCalculated:
            return(self.__dxFaceI,self.__dyFaceI,self.__dzFaceI,
                   self.__dxFaceJ,self.__dyFaceJ,self.__dzFaceJ,
                   self.__dxFaceK,self.__dyFaceK,self.__dzFaceK)
                   
        self.faceCenters()
        self.__dxFaceI=self.__xCenterFaceI[1:,:,:]-self.__xCenterFaceI[:-1,:,:]
        self.__dxFaceJ=self.__xCenterFaceJ[:,1:,:]-self.__xCenterFaceJ[:,:-1,:]
        self.__dxFaceK=self.__xCenterFaceK[:,:,1:]-self.__xCenterFaceK[:,:,:-1]

        self.__dyFaceI=self.__yCenterFaceI[1:,:,:]-self.__yCenterFaceI[:-1,:,:]
        self.__dyFaceJ=self.__yCenterFaceJ[:,1:,:]-self.__yCenterFaceJ[:,:-1,:]
        self.__dyFaceK=self.__yCenterFaceK[:,:,1:]-self.__yCenterFaceK[:,:,:-1]

        self.__dzFaceI=self.__zCenterFaceI[1:,:,:]-self.__zCenterFaceI[:-1,:,:]
        self.__dzFaceJ=self.__zCenterFaceJ[:,1:,:]-self.__zCenterFaceJ[:,:-1,:]
        self.__dzFaceK=self.__zCenterFaceK[:,:,1:]-self.__zCenterFaceK[:,:,:-1]
        
        self.__isFaceVectorsCalculated = True
        
        return(self.__dxFaceI,self.__dyFaceI,self.__dzFaceI,
               self.__dxFaceJ,self.__dyFaceJ,self.__dzFaceJ,
               self.__dxFaceK,self.__dyFaceK,self.__dzFaceK)
                  
                   
    def edgeLengths(self):
        """
        Computes the lengths of the i,j,k edges.
        """
        
        if self.__isEdgeLengthsCalculated:
            return(self.__dEdgeI,self.__dEdgeJ,self.__dEdgeK)
            
        self.__dEdgeI=n.sqrt((self.x[1:,:,:]-self.x[:-1,:,:])**2+
                             (self.y[1:,:,:]-self.y[:-1,:,:])**2+
                             (self.z[1:,:,:]-self.z[:-1,:,:])**2)
        
        self.__dEdgeJ=n.sqrt((self.x[:,1:,:]-self.x[:,:-1,:])**2+
                              (self.y[:,1:,:]-self.y[:,:-1,:])**2+
                              (self.z[:,1:,:]-self.z[:,:-1,:])**2)
                      
        self.__dEdgeK=n.sqrt((self.x[:,:,1:]-self.x[:,:,:-1])**2+
                              (self.y[:,:,1:]-self.y[:,:,:-1])**2+
                              (self.z[:,:,1:]-self.z[:,:,:-1])**2)
                      
        self.__isEdgeLengthsCalculated = True
        return(self.__dEdgeI,self.__dEdgeJ,self.__dEdgeK)         

    def cellCenters(self):
        """
        Returns x,y,z location of cell centers
        """
        
        if self.__isCellCentersCalculated:
            return (self.__xCenter,self.__yCenter,self.__zCenter)
        
        # Center location is just average of 8 corners    
        self.__xCenter=0.125*(self.x[:-1,:-1,:-1]+self.x[1:,:-1,:-1]+
                              self.x[:-1,1:,:-1]+self.x[:-1,:-1,1:]+
                              self.x[1:,1:,:-1]+self.x[1:,:-1,1:]+
                              self.x[:-1,1:,1:]+self.x[1:,1:,1:] )
        self.__yCenter=0.125*(self.y[:-1,:-1,:-1]+self.y[1:,:-1,:-1]+
                              self.y[:-1,1:,:-1]+self.y[:-1,:-1,1:]+
                              self.y[1:,1:,:-1]+self.y[1:,:-1,1:]+
                              self.y[:-1,1:,1:]+self.y[1:,1:,1:] )
        self.__zCenter=0.125*(self.z[:-1,:-1,:-1]+self.z[1:,:-1,:-1]+
                              self.z[:-1,1:,:-1]+self.z[:-1,:-1,1:]+
                              self.z[1:,1:,:-1]+self.z[1:,:-1,1:]+
                              self.z[:-1,1:,1:]+self.z[1:,1:,1:] )
                              
        self.__isCellCentersCalculated = True
        return (self.__xCenter,self.__yCenter,self.__zCenter)
