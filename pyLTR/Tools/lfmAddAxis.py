import numpy as n

def lfmAddAxis(x):
    """
    Add axis to LFM Cell Centered Variable
    
    Inputs
        x - Variable to add axis - typical size (ni,nj,nk)
        
    Outputs
        xA - Variable with axis added new size (ni,nj+2,nk)
    """
    (ni,nj,nk) = x.shape
    xA = n.zeros((ni,nj+2,nk))
    xA[:,1:-1,:] = x
    # Average around k for each and replicate in ni,nk and then add to output
    xA[:,0,:] = n.reshape(n.tile(n.mean(x[:,0,:],axis=1),nk),(ni,nk),
                            order='F')
    xA[:,nj+1,:] = n.reshape(n.tile(n.mean(x[:,nj-1,:],axis=1),nk),(ni,nk),
                            order='F')
    return xA
    