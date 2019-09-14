def LFMEfield(grid,ei,ej,ek,rion=6.5e8):
    """
    Computes the Cell Center Electric field from the E(ijk) components in the HDF file
    
    grid - LFM Grid object in Rion unites
    ei - Ei electric field at edges in grid
    ej - Ej electric field at edges in grid
    ek - Ek electric field at edges in grid
    
    rion - ion radius defaults to 6.5e8 cm set equal to 1 if you pass raw lfm grid
    
    Returns 
    
    ex - X Electric field in V/m
    ey - Y Electric field in V/m
    ez - Z Electric field in V/m
    
    """
    
    e_i = 0.25*(ei[:,:-1,:-1]+ei[:,:-1,1:]+ei[:,1:,:-1]+ei[:,1:,1:])
    e_j = 0.25*(ej[:-1,:,:-1]+ej[:-1,:,1:]+ej[1:,:,:-1]+ej[1:,:,1:])
    e_k = 0.25*(ek[:-1,:-1,:]+ek[1:,:-1,:]+ek[:-1,1:,:]+ek[1:,1:,:])
    (face_ix,face_iy,face_iz,
    face_jx,face_jy,face_jz,
    face_kx,face_ky,face_kz)=grid.faceVectors()
    m_11 = face_kz*face_jy - face_ky*face_jz
    m_21 = face_kx*face_jz - face_kz*face_jx
    m_31 = face_ky*face_jx - face_kx*face_jy
    
    m_12 = face_ky*face_iz - face_kz*face_iy
    m_22 = face_kz*face_ix - face_kx*face_iz
    m_32 = face_kx*face_iy - face_ky*face_ix
    
    m_13 = face_jz*face_iy - face_jy*face_iz
    m_23 = face_jx*face_iz - face_jz*face_ix
    m_33 = face_jy*face_ix - face_jx*face_iy        
    
    determ = 1.e6*(face_ix*m_11+face_jx*m_12+face_kx*m_13)
    
    ex = 1.0*(m_11*e_i+m_12*e_j+m_13*e_k)/determ
    ey = 1.0*(m_21*e_i+m_22*e_j+m_23*e_k)/determ
    ez = 1.0*(m_31*e_i+m_32*e_j+m_33*e_k)/determ
    
    return (ex,ey,ez)   
    