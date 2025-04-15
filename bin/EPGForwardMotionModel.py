import numpy as np
from numba import njit
from EPGMotion import *
from Jacobian_MotionCorrection import *

@njit()
def EPGForwardMotionModel(theta, opt):
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
    F = np.zeros((nTR, opt['GArr'].shape[0]), dtype='c8')

    # Preallocate MotionScale array 
    MotionScale = np.zeros(nTR)

    # Loop over k
    for k in range(opt['GArr'].shape[0]):
        opt['G'] = np.asarray([opt['GArr'][k]], dtype='f8')
        
        # Update MotionScale array
        MotionScale[steadyStateTR:] = theta[2+opt['GArr'].shape[0] + MotionSize * k: 2+opt['GArr'].shape[0] + MotionSize * (k + 1)]
        
        # Perform the EPGMotion calculation
        F[:, k] = S0 * EPGMotion(opt, MotionScale) * np.exp(1j * phi[k])

    # Extract real and imaginary parts and flatten the result
    FOutReal = np.real(F[nDummy:,...]).copy()
    FOutImag = np.imag(F[nDummy:,...]).copy()
    ReshapeSize=FOutReal.shape[0]*FOutReal.shape[1]
    Output = np.concatenate((FOutReal.reshape(ReshapeSize),FOutImag.reshape(ReshapeSize)))

    return Output

@njit()
def EPGForwardModel(theta, opt, MotionScale):
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
    F = np.zeros((nTR, opt['GArr'].shape[0]), dtype='c8')

    # Loop over k
    for k in range(opt['GArr'].shape[0]):
        opt['G'] = np.asarray([opt['GArr'][k]], dtype='f8')
        # Perform the EPGMotion calculation
        F[:, k] = S0 * EPGMotion(opt, MotionScale[:,k]) * np.exp(1j * phi[k])

    # Extract real and imaginary parts and flatten the result
    FOutReal = np.real(F[nDummy:,...]).copy()
    FOutImag = np.imag(F[nDummy:,...]).copy()
    ReshapeSize=FOutReal.shape[0]*FOutReal.shape[1]
    Output = np.concatenate((FOutReal.reshape(ReshapeSize),FOutImag.reshape(ReshapeSize)))

    return Output

def EPGForwardModelFitting(x,theta,opt,MotionScale=None):
    theta=np.array([*theta])
    if MotionScale is None:
        MotionScale = np.zeros((int(opt["nTR"]),opt['GArr'].shape[0]))
    return EPGForwardModel(theta,opt,MotionScale)

def EPGForwardMotionModelFitting(x,theta,opt):
    theta=np.array([*theta])
    return EPGForwardMotionModel(theta,opt)

def EPGForwardMotionModelFittingJacobian(x,theta,opt):
    theta=np.array([*theta])
    return Jacobian_MotionCorrection(theta, opt)

def EPGForwardModelFittingJacobian(x,theta,opt,MotionScale=None):
    theta=np.array([*theta])
    if MotionScale is None:
        MotionScale = np.zeros((int(opt["nTR"]),opt['GArr'].shape[0]))
    return Jacobian_noMotionCorrection(theta, opt,MotionScale)
