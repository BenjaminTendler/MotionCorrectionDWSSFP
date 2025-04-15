##
#Load Modules
import numpy as np
from numba import njit
import pdb
from EPGMotion import *

@njit()
def Jacobian_MotionCorrection(theta, opt):
    # Cache frequently used values
    nTR = opt["nTR"].astype(np.int32)[0]
    steadyStateTR = opt["SteadyStateTR"].astype(np.int32)[0]
    nDummy = opt['nDummy'].astype(np.int32)[0]
    MotionSize = nTR - steadyStateTR

    # Convert inputs to numpy arrays
    opt['D'] = np.asarray([theta[0]], dtype='f8')
    S0 = theta[1]
    phi = theta[2:2+opt['GArr'].shape[0]]

    # Preallocate results
    FDerivative = np.zeros((nTR, opt['GArr'].shape[0],len(theta)), dtype='c8')

    # Preallocate MotionScale array 
    MotionScale = np.zeros(nTR)

    # Loop over k
    for k in range(opt['GArr'].shape[0]):
        opt['G'] = np.asarray([opt['GArr'][k]], dtype='f8')
        
        # Update MotionScale array
        MotionScale[steadyStateTR:] = theta[2+opt['GArr'].shape[0] + MotionSize * k: 2+opt['GArr'].shape[0] + MotionSize * (k + 1)]
        # Diffusivity Derivative - Numerical
        FDerivative[:, k, 0] = S0 * EPGMotionDiffusionDerivative(opt, MotionScale) * np.exp(1j * phi[k])
        opt['D'] = np.asarray([theta[0]], dtype='f8')
        # S0 Derivative - Analytical
        FDerivative[:, k, 1] = EPGMotion(opt, MotionScale) * np.exp(1j * phi[k])
        # Phase Derivative - Analytical
        FDerivative[:, k, k+2] = 1j * S0 * EPGMotion(opt, MotionScale) * np.exp(1j * phi[k])
        # Motion Derivative - Analytical
        FDerivativeMotion = EPGMotionMotionDerivative(opt, MotionScale)
        FDerivative[:, k, 2+opt['GArr'].shape[0]+MotionSize*k:2+opt['GArr'].shape[0]+(k+1)*MotionSize] = S0 * FDerivativeMotion[:,steadyStateTR:] * np.exp(1j * phi[k])
    
    # Extract real and imaginary parts and flatten the result
    FOutReal = np.real(FDerivative[nDummy:,...]).copy()
    FOutImag = np.imag(FDerivative[nDummy:,...]).copy()
    ReshapeSize=FOutReal.shape[0]*FOutReal.shape[1]
    Output = np.concatenate((FOutReal.reshape(ReshapeSize,FOutReal.shape[2]),FOutImag.reshape(ReshapeSize,FOutReal.shape[2])))

    return Output

@njit()
def Jacobian_noMotionCorrection(theta, opt, MotionScale):
    # Cache frequently used values
    nTR = opt["nTR"].astype(np.int32)[0]
    steadyStateTR = opt["SteadyStateTR"].astype(np.int32)[0]
    nDummy = opt['nDummy'].astype(np.int32)[0]
    MotionSize = nTR - steadyStateTR

    # Convert inputs to numpy arrays
    opt['D'] = np.asarray([theta[0]], dtype='f8')
    S0 = theta[1]
    phi = theta[2:opt['GArr'].shape[0]]

    # Preallocate results
    FDerivative = np.zeros((nTR, opt['GArr'].shape[0],len(theta)), dtype='c8')

    # Loop over k
    for k in range(opt['GArr'].shape[0]):
        opt['G'] = np.asarray([opt['GArr'][k]], dtype='f8')
        # Diffusivity Derivative - Numerical
        FDerivative[:, k, 0] = S0 * EPGMotionDiffusionDerivative(opt, MotionScale[:,k]) * np.exp(1j * phi[k])
        opt['D'] = np.asarray([theta[0]], dtype='f8')
        # S0 Derivative - Analytical
        FDerivative[:, k, 1] = EPGMotion(opt, MotionScale[:,k]) * np.exp(1j * phi[k])
        # Phase Derivative - Analytical
        FDerivative[:, k, k+2] = 1j * S0 * EPGMotion(opt, MotionScale[:,k]) * np.exp(1j * phi[k])

    # Extract real and imaginary parts and flatten the result
    FOutReal = np.real(FDerivative[nDummy:,...]).copy()
    FOutImag = np.imag(FDerivative[nDummy:,...]).copy()
    ReshapeSize=FOutReal.shape[0]*FOutReal.shape[1]
    Output = np.concatenate((FOutReal.reshape(ReshapeSize,FOutReal.shape[2]),FOutImag.reshape(ReshapeSize,FOutReal.shape[2])))

    return Output


@njit()
def Jacobian_noMotionCorrectionWholeImage(theta, opt):
    # Cache frequently used values
    nTR = opt["nTR"].astype(np.int32)[0]
    steadyStateTR = opt["SteadyStateTR"].astype(np.int32)[0]
    nDummy = opt['nDummy'].astype(np.int32)[0]
    MotionSize = nTR - steadyStateTR
    nParameters=len(opt['Mask'])

    # Convert inputs to numpy arrays
    opt['D'] = theta[0:nParameters]
    S0 = theta[nParameters:2*nParameters]
    phi = theta[2*nParameters:]

    # Preallocate results
    FDerivative = np.zeros((nTR, nParameters, 2,len(theta)), dtype='c8')
    # Preallocate MotionScale array 
    MotionScale = np.zeros(nTR)

    # Loop over k
    for k in range(2):
        opt['G'] = np.asarray([opt['GArr'][k]], dtype='f8')
        # Default Simulation
        F=S0 * EPGMotionWholeImage(opt) * np.exp(1j * phi[(k)*nParameters:(k+1)*nParameters])
        FDerivative[:, :, k, :]=np.expand_dims(F,2).repeat(FDerivative.shape[-1]).reshape(FDerivative[:,:,k,:].shape)
        # Diffusivity Derivative - Numerical
        DDerivative = S0 * EPGMotionDiffusionDerivativeWholeImage(opt, MotionScale) * np.exp(1j * phi[(k)*nParameters:(k+1)*nParameters])
        for l in range(nParameters):
            FDerivative[:, l, k, l]=DDerivative[:,l]
        opt['D'] = theta[0:nParameters]
        # S0 Derivative - Analytical
        SDerivative = EPGMotionWholeImage(opt) * np.exp(1j * phi[(k)*nParameters:(k+1)*nParameters])
        for l in range(nParameters):
            FDerivative[:, l, k, l+nParameters]=SDerivative[:,l]        # Phase Derivative - Analytical
        if k==0:
            FDerivative[:, k, 3*nParameters:4*nParameters] = 1j*FDerivative[:, k, 3*nParameters:4*nParameters]
            FDerivative[:, k, 4*nParameters:]=0
        else:
            FDerivative[:, k, 3*nParameters:4*nParameters] = 0
            FDerivative[:, k, 4*nParameters:]=1j*FDerivative[:, k, 4*nParameters:]      
    # Extract real and imaginary parts and flatten the result
    Concat = np.swapaxes(np.concatenate((
        np.real(FDerivative[nDummy:, :, 0, :]),
        np.real(FDerivative[nDummy:, :, 1, :]),
        np.imag(FDerivative[nDummy:, :, 0, :]),
        np.imag(FDerivative[nDummy:, :, 1, :])
    )),0,1)
    Output=np.ascontiguousarray(Concat).reshape(-1,Concat.shape[-1])
    return Output



@njit()
def EPGMotionDiffusionDerivative(opt, MotionScale=None):
    Step=1E-7
    Centre=EPGMotion(opt,MotionScale)
    opt['D'] = np.asarray([opt['D'][0]+Step], dtype='f8')
    Right=EPGMotion(opt,MotionScale)
    opt['D'] = np.asarray([opt['D'][0]-2*Step], dtype='f8')
    Left=EPGMotion(opt,MotionScale)
    opt['D'] = np.asarray([opt['D'][0]-Step], dtype='f8')
    Left2=EPGMotion(opt,MotionScale)
    opt['D'] = np.asarray([opt['D'][0]+4*Step], dtype='f8')
    Right2=EPGMotion(opt,MotionScale)
    Grad=(1/12*Left2-2/3*Left+2/3*Right-1/12*Right2)/Step
    return Grad

@njit()
def EPGMotionDiffusionDerivativeWholeImage(opt, MotionScale=None):
    Step=opt['D']*0+1E-7
    Centre=EPGMotionWholeImage(opt,MotionScale)
    opt['D'] = opt['D']+Step
    Right=EPGMotionWholeImage(opt,MotionScale)
    opt['D'] = opt['D']-2*Step
    Left=EPGMotionWholeImage(opt,MotionScale)
    opt['D'] = opt['D']-Step
    Left2=EPGMotionWholeImage(opt,MotionScale)
    opt['D'] = opt['D']+4*Step
    Right2=EPGMotionWholeImage(opt,MotionScale)
    Grad=(1/12*Left2-2/3*Left+2/3*Right-1/12*Right2)/Step[0]
    return Grad

@njit()
def EPGMotionMotionDerivative(opt, MotionScale=None):
    #
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
    pfx, pfy, pzx, pzy, fx, fy, zx, zy, F = MatrixInitMotionDerivative(kOrder,nTR)
    #Define RF Matrix components
    a, b, c, d, e, f, g, h, hb, gb, ec, fc = RFArray(opt)
    
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
            f = (pfx[nn,:] + 1j * pfy[nn,:]) * e2 * np.exp(-bTrans_Grad[0] * ((2 * k + 1)**2 / 4 + 1 / 12) * opt['D'][0] +
                -bTrans_noGrad[0] * (k + 1)**2 * opt['D'][0] \
                -1j * (vTrans_Grad[j] * (2 * k + 1) / 2 \
                + vTrans_noGrad[j] * (k + 1)))
            
            # Perform Gradient operation
            scale_f=-1j * gamma_tau_G * 10**-12 * (opt['tau'] * (2 * k + 1) / 2 \
                       + (opt['TR'] - opt['tau']) * (k + 1))
            f[j]=f[j] * scale_f[0]            

            fx[nn + 1,:] = np.real(f)
            fy[nn + 1,:] = np.imag(f)
                
            # Longitudinal
            exp_bLong=np.exp(-bLong[0] * k**2 * opt['D'][0])
            if k > 0:
                zx[kk,:] = pzx[kk,:] * e1[0] * exp_bLong
            if k == 0:
                zx[kk,:] = pzx[kk,:] * e1[0] + 1.0 - e1[0]
                zx[kk,j] = 0 
                zx[kk,0:j] = pzx[kk,0:j] * e1[0]
                zy[kk,:] = pzy[kk,:] * e1[0] * exp_bLong
                zy[kk,j] = 0
            if k > 0:
                zy[kk,:] = pzy[kk,:] * e1[0] * exp_bLong
            if k >= 0:
                z = (zx[kk,:] + 1j * zy[kk,:])*np.exp(-1j * vLong[j] * k)
                scale_z=-1j * k * gamma_tau_G * opt['TR'] * 10**-12 
                z[j] = z[j] * scale_z[0]
            if k > 0:
                zx[kk,:] = np.real(z)
            if k >= 0:
                zy[kk,:] = np.imag(z)
        
        # Calculate DW-SSFP component
        F[j,:] = (fx[(fx.shape[0] - 1) // 2,:] + 1j * fy[(fy.shape[0] - 1) // 2,:])
        F[0:j,j] = 0

    return F

@njit()
def MatrixInitMotionDerivative(kOrder,nTR):
    """
    Initialize the matrices as needed based on the dimensions specified in opt.
    """

    # Generate matrices
    pfx = np.zeros((2 * kOrder + 1,nTR))
    pfy = np.zeros((2 * kOrder + 1,nTR))
    pzx = np.zeros((kOrder,nTR))
    pzy = np.zeros((kOrder,nTR))
    fx = np.zeros((2 * kOrder + 1,nTR))
    fy = np.zeros((2 * kOrder + 1,nTR))
    zx = np.zeros((kOrder,nTR))
    zy = np.zeros((kOrder,nTR))
    F = np.zeros((nTR,nTR),dtype = 'c8')

    return pfx, pfy, pzx, pzy, fx, fy, zx, zy, F
