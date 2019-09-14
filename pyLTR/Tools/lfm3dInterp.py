import numpy as n

def lfm3dInterp(xc,yc,zc,data,x0,y0,z0):
    """
    Computes value of data at interpolation point using LFM cell centered grid
    xc,yc,zc - Cell Center locations of LFM grid
    data - value to be interpolated
    x0,y0,z0 - location of interpolation point
    
    return val of data at point
    """
    thetac = n.pi - n.sign(zc[5,5,:])*n.arccos(-1.0*yc[5,5,:]/n.sqrt(yc[5,5,:]**2+zc[5,5,:]**2))
    if (z0 == 0):
        theta0 = n.pi/2.0-n.pi/2.0*n.sign(y0)
    else:
        theta0 = n.pi-n.sign(z0)*n.arccos(-1.0*y0/n.sqrt(y0**2+z0**2))
        
    kp1 = n.argmax(thetac>theta0)
    rho0 = n.sqrt(y0**2+z0**2)
    thetak = thetac[kp1-1]
    thetakp1 = thetac[kp1]
    yk = rho0*n.cos(thetak)
    zk = rho0*n.sin(thetak)
    ykp1 = rho0*n.cos(thetakp1)
    zkp1 = rho0*n.sin(thetakp1)
    valkp1 = lfmKshellTriInterp(xc,yc,zc,data,x0,ykp1,zkp1,kp1)
    valk = lfmKshellTriInterp(xc,yc,zc,data,x0,yk,zk,kp1-1)
    
    if (kp1 == 0): thetak=thetak-2.0*n.pi
    val = (valkp1-valk)*(theta0-thetak)/(thetakp1-thetak)+valk
    return val
    
def lfmKshellTriInterp(xc,yc,zc,data,x0,y0,z0,KK):
    """
    Computes value of data at interpolation point using LFM cell centered 
    k-plane
    Inputs - 
        xc,yc,zc - Cell Center locations of LFM grid with axis added
        data - value to be interpolated with axis added
        x0,y0,z0 - location of interpolation point on the k-plane
        KK - k-shell on which to perform interpolation
    Outputs
        value of data at point
    """
    # First determine array size and get copies of plane we need
    (ni,nj,nk) = xc.shape
    p = xc[:,:,KK]
    q = n.sqrt(yc[:,:,KK]**2+zc[:,:,KK]**2)
    data1 = data[:,:,KK]
    p1 = x0
    q1 = n.sqrt(y0**2+z0**2)
    # While breaking out of for loop is genearlly a bad programing pratice we 
    # do it here for speed.  Once we have found the location we don't need to 
    # any more computaiton.  NB - The iLoopBreak logical is needed becuase the 
    # break in the j-loop only aborts that loop and not the outer i-loop
    
    # find out which triangle is (p1.q1) in
    # search through (i,j) pairs, each cell is divided to two trangles:
    # 1 (i,j) (i+1,j),(i,j+1)
    # 2 (i+1,j+1) (i+1,j),(i,j+1)
    
    iLoopBreak = False
    for i in range(ni-1):
        for j in range(nj-1):
            s1 = [p[i,j]-p1,q[i,j]-q1]
            s2 = [p[i+1,j]-p1,q[i+1,j]-q1]
            s3 = [p[i+1,j+1]-p1,q[i+1,j+1]-q1]
            s4 = [p[i,j+1]-p1,q[i,j+1]-q1]
            # triangle 1, ANG(12)+ANG(24)+ang(41)=2*pi
            theta12=n.arccos((s1[0]*s2[0]+s1[1]*s2[1])/n.sqrt((s1[0]**2+s1[1]**2)*(s2[0]**2+s2[1]**2)))
            theta24=n.arccos((s2[0]*s4[0]+s2[1]*s4[1])/n.sqrt((s2[0]**2+s2[1]**2)*(s4[0]**2+s4[1]**2)))        
            theta41=n.arccos((s4[0]*s1[0]+s4[1]*s1[1])/n.sqrt((s4[0]**2+s4[1]**2)*(s1[0]**2+s1[1]**2)))
            if(n.abs(theta12+theta24+theta41-2.0*n.pi)< 0.001):
                xx1 = p[i,j]
                yy1 = q[i,j]
                ff1 = data1[i,j]
                xx2 = p[i+1,j]
                yy2 = q[i+1,j]
                ff2 = data1[i+1,j]
                xx3 = p[i,j+1]
                yy3 = q[i,j+1]
                ff3 = data1[i,j+1]
                iLoopBreak = True
                break
            # triangle 2, ANG(23)+ANG(34)+ang(42)=2*pi
            theta23=n.arccos((s2[0]*s3[0]+s2[1]*s3[1])/n.sqrt((s2[0]**2+s2[1]**2)*(s3[0]**2+s3[1]**2)))
            theta34=n.arccos((s3[0]*s4[0]+s3[1]*s4[1])/n.sqrt((s3[0]**2+s3[1]**2)*(s4[0]**2+s4[1]**2)))        
            theta42=n.arccos((s4[0]*s2[0]+s4[1]*s2[1])/n.sqrt((s4[0]**2+s4[1]**2)*(s2[0]**2+s2[1]**2)))
            if(n.abs(theta23+theta34+theta42-2.0*n.pi)< 0.001):
                xx1 = p[i+1,j+1]
                yy1 = q[i+1,j+1]
                ff1 = data1[i+1,j+1]
                xx2 = p[i+1,j]
                yy2 = q[i+1,j]
                ff2 = data1[i+1,j]
                xx3 = p[i,j+1]
                yy3 = q[i,j+1]
                ff3 = data1[i,j+1]
                iLoopBreak = True
                break
            if iLoopBreak: break
    try:
        arr1 = n.array([xx1,yy1,1])
        arr2 = n.array([xx2,yy2,1])
        arr3 = n.array([xx3,yy3,1])
        arr = n.array([p1,q1,1])
        return ((ff1*n.linalg.det(n.array([arr,arr2,arr3]))-
                ff2*n.linalg.det(n.array([arr,arr1,arr3]))+
                ff3*n.linalg.det(n.array([arr,arr1,arr2])))/
                n.linalg.det(n.array([arr1,arr2,arr3])))
    except UnboundLocalError:
        return n.nan

