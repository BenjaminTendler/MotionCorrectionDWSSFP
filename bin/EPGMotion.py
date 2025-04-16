##
#Load Modules
import numpy as np
from numba import njit
import pdb

@njit()
def EPGMotion(opt, MotionScale = None, Sequence = 'DW-SSFP'):

    #Account for DW-SE Sequence
    if Sequence == 'DW-SE':
        opt['nTR'] = np.asarray([2], dtype='f8')
    
    #Initialise some values to prevent repeat calculations
    nTR = opt['nTR'].astype(np.int32)[0]
    kOrder=opt['kOrder'].astype(np.int32)[0]
    
    # Set no motion parameter
    if MotionScale is None:
        MotionScale = np.zeros((nTR))
    
    # Define exponential relaxation coefficients
    e1 = np.exp(-opt['TR'] / opt['T1'])
    e2 = np.exp(-opt['TR'] / opt['T2'])
    
    # Define b-values
    gamma_tau_G = opt['gamma'] * opt['tau'] * opt['G']
    bTrans_Grad = (gamma_tau_G**2) * opt['tau'] * 10**-21
    bTrans_noGrad = (gamma_tau_G**2) * (opt['TR'] - opt['tau']) * 10**-21
    bLong = (gamma_tau_G**2) * opt['TR'] * 10**-21
    
    # Define Velocity-unit
    vTrans_Grad = gamma_tau_G * opt['tau'] * MotionScale * 10**-12
    vTrans_noGrad = gamma_tau_G * (opt['TR'] - opt['tau']) * MotionScale * 10**-12
    vLong = gamma_tau_G * opt['TR'] * MotionScale * 10**-12

    # Create empty matrices
    pfx, pfy, pzx, pzy, fx, fy, zx, zy, F = MatrixInit(kOrder,nTR)
    #Define RF Matrix components
    a, b, c, d, e, f, g, h, hb, gb, ec, fc = RFArray(opt.copy())
    
    # Set initial conditions
    zx[0] = 1
    
    # Run EPG
    for j in range(nTR):
        # Perform RF Operation
        for k in range(min(j + 1,kOrder)):
            # Define indices
            kk = k
            nn = kOrder + k
            mm = kOrder - k

            #Account for DW-SE Sequence
            if Sequence == 'DW-SE':
                if j == 0:
                    opt['alpha'] = np.asarray([90], dtype='f8')
                    a, b, c, d, e, f, g, h, hb, gb, ec, fc = RFArray(opt.copy())
                if j == 1:
                    opt['alpha'] = np.asarray([180], dtype='f8')
                    a, b, c, d, e, f, g, h, hb, gb, ec, fc = RFArray(opt.copy())
            
            pfx[nn] = (a * fx[nn] + hb * fx[mm] + gb * fy[mm] + ec * zx[kk] + fc * zy[kk])
            pfy[nn] = (a * fy[nn] - hb * fy[mm] + gb * fx[mm] - fc * zx[kk] + ec * zy[kk])
            pfx[mm] = (hb * fx[nn] + gb * fy[nn] + a * fx[mm] + ec * zx[kk] - fc * zy[kk])
            pfy[mm] = (gb * fx[nn] - hb * fy[nn] + a * fy[mm] - fc * zx[kk] - ec * zy[kk])
            pzx[kk] = (-ec * fx[nn] + fc * fy[nn] - ec * fx[mm] + fc * fy[mm] + 2.0 * d * zx[kk]) / 2.0
            pzy[kk] = (-fc * fx[nn] - ec * fy[nn] + fc * fx[mm] + ec * fy[mm] + 2.0 * d * zy[kk]) / 2.0
        
        # Relaxation and diffusion encoding
        for k in range(-kOrder, kOrder):
            kk = k
            nn = kOrder + kk
            
            # Transverse
            f = (pfx[nn] + 1j * pfy[nn]) * e2 * np.exp(-bTrans_Grad[0] * ((2 * k + 1)**2 / 4 + 1 / 12) * opt['D'][0] +
                -bTrans_noGrad[0] * (k + 1)**2 * opt['D'][0] \
                -1j * (vTrans_Grad[j] * (2 * k + 1) / 2 \
                + vTrans_noGrad[j] * (k + 1)))
            fx[nn + 1] = np.real(f[0])
            fy[nn + 1] = np.imag(f[0])
            
            # Longitudinal
            exp_bLong=np.exp(-bLong[0] * k**2 * opt['D'][0])
            if k > 0:
                zx[kk] = pzx[kk] * e1[0] * exp_bLong
            if k == 0:
                zx[kk] = pzx[kk] * e1[0] + 1.0 - e1[0]
            if k >= 0:
                zy[kk] = pzy[kk] * e1[0] * exp_bLong
            if k >= 0:
                z = (zx[kk] + 1j * zy[kk])*np.exp(-1j * vLong[j] * k)
            if k > 0:
                zx[kk] = np.real(z)
            if k >= 0:
                zy[kk] = np.imag(z)
        
        # Calculate DW-SSFP component
        F[j] = (fx[(fx.shape[0] - 1) // 2] + 1j * fy[(fy.shape[0] - 1) // 2])

    return F

@njit()
def EPGMotionWholeImage(opt, MotionScale = None, Sequence = 'DW-SSFP'):
    
    ##
    #Account for DW-SE Sequence
    if Sequence == 'DW-SE':
        opt['nTR'] = np.asarray([2], dtype='f8')

    ##
    #Copy dictionary
    optSlice = opt.copy()

    ##
    #Initialise Array
    F = np.zeros((len(opt['Mask']),opt['nTR'].astype(np.int32)[0]),dtype = 'c8')

    ##
    #Define Motion Parameter
    if MotionScale is None:
        MotionScale = np.zeros((opt['T1'][:].shape[0],opt['nTR'].astype(np.int32)[0]))

    #Loop and allocate
    Spacing = int(20000)
    ArrLength = opt['D'].shape[0]

    for k in range(0,ArrLength,Spacing):
        optSlice['D'] = opt['D'][k:min(k+Spacing,ArrLength)]
        optSlice['T1'] = opt['T1'][k:min(k+Spacing,ArrLength)]
        optSlice['T2'] = opt['T2'][k:min(k+Spacing,ArrLength)]
        optSlice['B1'] = opt['B1'][k:min(k+Spacing,ArrLength)]
        optSlice['Mask'] = opt['Mask'][k:min(k+Spacing,ArrLength)]
        F[k:min(k+Spacing,ArrLength),:] = EPGMotionWholeImageSlice(optSlice, MotionScale = MotionScale[k:min(k+Spacing,ArrLength),:], Sequence = Sequence)
    
    return F

@njit()
def EPGMotionWholeImageSlice(opt, MotionScale=None, Sequence = 'DW-SSFP'):

    #Initialise some values to prevent repeat calculations
    nTR = opt['nTR'].astype(np.int32)[0]
    kOrder=opt['kOrder'].astype(np.int32)[0]
    
    # Set no motion parameter
    if MotionScale is None:
        MotionScale = np.zeros((opt['T1'][:].shape[0],nTR))
    
    # Define exponential relaxation coefficients
    e1 = np.exp(-opt['TR'] / opt['T1'][:])
    e2 = np.exp(-opt['TR'] / opt['T2'][:])
    
    # Define b-values
    gamma_tau_G = opt['gamma'] * opt['tau'] * opt['G']
    bTrans_Grad = (gamma_tau_G**2) * opt['tau'] * 10**-21
    bTrans_noGrad = (gamma_tau_G**2) * (opt['TR'] - opt['tau']) * 10**-21
    bLong = (gamma_tau_G**2) * opt['TR'] * 10**-21
    
    # Define Velocity-unit
    vTrans_Grad = gamma_tau_G * opt['tau'] * MotionScale.transpose() * 10**-12
    vTrans_noGrad = gamma_tau_G * (opt['TR'] - opt['tau']) * MotionScale.transpose() * 10**-12
    vLong = gamma_tau_G * opt['TR'] * MotionScale.transpose() * 10**-12

    # Create empty matrices
    pfx, pfy, pzx, pzy, fx, fy, zx, zy, F = MatrixInitWholeImage(opt.copy(),kOrder,nTR)
    #Define RF Matrix components
    a, b, c, d, e, f, g, h, hb, gb, ec, fc = RFArrayWholeImage(opt.copy())
    
    # Set initial conditions
    zx[0,:] = 1
    # Run EPG
    for j in range(nTR):
        # Perform RF Operation
        for k in range(min(j + 1,kOrder)):
            # Define indices
            kk = k
            nn = kOrder + k
            mm = kOrder - k
            
            #Account for DW-SE Sequence
            if Sequence == 'DW-SE':
                if j == 0:
                    opt['alpha'] = np.asarray([90], dtype='f8')
                    a, b, c, d, e, f, g, h, hb, gb, ec, fc = RFArrayWholeImage(opt.copy())
                if j == 1:
                    opt['alpha'] = np.asarray([180], dtype='f8')
                    a, b, c, d, e, f, g, h, hb, gb, ec, fc = RFArrayWholeImage(opt.copy())
            pfx[nn,:] = (a * fx[nn,:] + hb * fx[mm,:] + gb * fy[mm,:] + ec * zx[kk,:] + fc * zy[kk,:])
            pfy[nn,:] = (a * fy[nn,:] - hb * fy[mm,:] + gb * fx[mm,:] - fc * zx[kk,:] + ec * zy[kk,:])
            pfx[mm,:] = (hb * fx[nn,:] + gb * fy[nn,:] + a * fx[mm,:] + ec * zx[kk,:] - fc * zy[kk,:])
            pfy[mm,:] = (gb * fx[nn,:] - hb * fy[nn,:] + a * fy[mm,:] - fc * zx[kk,:] - ec * zy[kk,:])
            pzx[kk,:] = (-ec * fx[nn,:] + fc * fy[nn,:] - ec * fx[mm,:] + fc * fy[mm,:] + 2.0 * d * zx[kk,:]) / 2.0
            pzy[kk,:] = (-fc * fx[nn,:] - ec * fy[nn,:] + fc * fx[mm,:] + ec * fy[mm,:] + 2.0 * d * zy[kk,:]) / 2.0
        
        # Relaxation and diffusion encoding
        for k in range(-kOrder, kOrder):
            kk = k
            nn = kOrder + kk
            
            # Transverse
            f = (pfx[nn,:] + 1j * pfy[nn,:]) * e2[:] * np.exp(-bTrans_Grad[0] * ((2 * k + 1)**2 / 4 + 1 / 12) * opt['D'][:] +
                -bTrans_noGrad[0] * (k + 1)**2 * opt['D'][:] \
                -1j * (vTrans_Grad[j,:] * (2 * k + 1) / 2 \
                + vTrans_noGrad[j,:] * (k + 1)))
            fx[nn + 1,:] = np.real(f[:])
            fy[nn + 1,:] = np.imag(f[:])
            
            # Longitudinal
            exp_bLong=np.exp(-bLong[0] * k**2 * opt['D'][:])
            if k > 0:
                zx[kk,:] = pzx[kk,:] * e1[:] * exp_bLong[:]
            if k == 0:
                zx[kk,:] = pzx[kk,:] * e1[:] + 1.0 - e1[:]
            if k >= 0:
                zy[kk,:] = pzy[kk,:] * e1[:] * exp_bLong[:]
            if k >= 0:
                z = (zx[kk,:] + 1j * zy[kk,:])*np.exp(-1j * vLong[j,:] * k)
            if k > 0:
                zx[kk,:] = np.real(z[:])
            if k >= 0:
                zy[kk,:] = np.imag(z[:])
        
        # Calculate DW-SSFP component
        F[j,:] = (fx[(fx.shape[0] - 1) // 2,:] + 1j * fy[(fy.shape[0] - 1) // 2,:])
    return np.swapaxes(F,1,0)


@njit()
def MatrixInitWholeImage(opt,kOrder,nTR):
    """
    Initialize the matrices as needed based on the dimensions specified in opt.
    """
    #Identify number of parameters
    nParameters=len(opt['Mask'])
    
    # Generate matrices
    pfx = np.zeros((2 * kOrder + 1,nParameters))
    pfy = np.zeros((2 * kOrder + 1,nParameters))
    pzx = np.zeros((kOrder,nParameters))
    pzy = np.zeros((kOrder,nParameters))
    fx = np.zeros((2 * kOrder + 1,nParameters))
    fy = np.zeros((2 * kOrder + 1,nParameters))
    zx = np.zeros((kOrder,nParameters))
    zy = np.zeros((kOrder,nParameters))
    F = np.zeros((nTR,nParameters),dtype = 'c8')

    return pfx, pfy, pzx, pzy, fx, fy, zx, zy, F

@njit()
def MatrixInit(kOrder,nTR):
    """
    Initialize the matrices as needed based on the dimensions specified in opt.
    """

    # Generate matrices
    pfx = np.zeros((2 * kOrder + 1))
    pfy = np.zeros((2 * kOrder + 1))
    pzx = np.zeros((kOrder))
    pzy = np.zeros((kOrder))
    fx = np.zeros((2 * kOrder + 1))
    fy = np.zeros((2 * kOrder + 1))
    zx = np.zeros((kOrder))
    zy = np.zeros((kOrder))
    F = np.zeros((nTR),dtype = 'c8')

    return pfx, pfy, pzx, pzy, fx, fy, zx, zy, F

@njit()
def RFArray(opt):
    """
    Calculate RF matrix components based on the given flip angle and phase.
    """
    # Define angles
    alpha = opt['alpha'] * opt['B1'] * np.pi / 180
    phi = opt['phi'] * np.pi / 180
    
    # Define matrix components
    a = np.cos(alpha / 2)**2
    b = np.sin(alpha / 2)**2
    c = np.sin(alpha)
    d = np.cos(alpha)
    e = np.sin(phi)
    f = np.cos(phi)
    g = np.sin(2.0 * phi)
    h = np.cos(2.0 * phi)
    
    hb = h * b
    gb = g * b
    ec = e * c
    fc = f * c
    
    return a[0], b[0], c[0], d[0], e[0], f[0], g[0], h[0], hb[0], gb[0], ec[0], fc[0]

@njit()
def RFArrayWholeImage(opt):
    """
    Calculate RF matrix components based on the given flip angle and phase.
    """
    # Define angles
    alpha = opt['alpha'] * opt['B1'] * np.pi / 180
    phi = opt['phi'] * np.pi / 180
    
    # Define matrix components
    a = np.cos(alpha / 2)**2
    b = np.sin(alpha / 2)**2
    c = np.sin(alpha)
    d = np.cos(alpha)
    e = np.sin(phi)
    f = np.cos(phi)
    g = np.sin(2.0 * phi)
    h = np.cos(2.0 * phi)
    
    hb = h * b
    gb = g * b
    ec = e * c
    fc = f * c
    
    return a[:], b[:], c[:], d[:], e[:], f[:], g[:], h[:], hb[:], gb[:], ec[:], fc[:]

def kOrder(opt):
    """
    Identify kOrder without considering diffusion effects.
    """
    # Define Reference
    opt['kOrder'] = opt['nTR']
    
    # Identify kOrder without considering diffusion effects
    opt['D'] = np.asarray([0], dtype='f8')
    FRef = EPGMotion(opt)
    
    F = np.zeros((len(FRef), opt['nTR'].astype(np.int32)[0]) ,dtype = 'c8')
    
    for k in range(1, opt['nTR'].astype(np.int32)[0] + 1):
        opt['kOrder'] = np.asarray([k], dtype='f8')
        F[:, k - 1] = EPGMotion(opt)
        
        if np.sum(np.abs(F[1:, k - 1] - FRef[1:]) / np.abs(FRef[1:])) < 1E-8:
            return np.asarray([k], dtype='f8') 
    
    return opt['nTR']  # In case no valid kValue is found
