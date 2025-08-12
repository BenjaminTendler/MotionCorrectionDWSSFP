import numpy as np
import pdb
from numba import njit
import matplotlib.pyplot as plt

##
#Estimate Motion Operator
@njit()
def MotionOperator(opt,TranslationVec,RotationVec, CardiacVec, Mask, VoxDims, RandMotion=False, HeartRate=50, offset=0):
    ##
    #Initialise Matrices
    V = np.zeros((*Mask.shape, opt['nTR'].astype(np.int32)[0]))

    ##
    #Define Z-axis
    zLoc = np.arange(-Mask.shape[2] / 2, Mask.shape[2] / 2)
    
    ##
    #Perform evaluation per slice
    for k in range(Mask.shape[2]):
        if np.any( Mask[:,:,k]):
            V[:,:,k,:] = MotionOperatorSlice(opt, TranslationVec, RotationVec, CardiacVec, Mask[:,:,k][:,:,np.newaxis], VoxDims, zLoc[k]+offset, RandMotion=False, HeartRate=50)[:,:,0,:]
    return V

##
#Estimate motion scaling based on input parameters - Translation and rotation
@njit()
def MotionOperatorSlice(opt,TranslationVec,RotationVec, CardiacVec, Mask, VoxDims, zLoc, RandMotion=False, HeartRate=50):

    """
    opt: Parameter Options (output of ParameterOptionsXXX.py)
    TranslationVec: 3 component vector describing the translation velocity [Vx, Vy, Vz] - units mm/s
    RotationVec: 3 component vector describing the rotational velocity [Omegax, Omegay, Omegaz] - units rad/s
    CardiacVec: 3 component vector describing the cardiac velocity [Vx, Vy, Vz] - units mm/s
    Mask: Sample Mask
    VoxDims: Voxel Dimensions (mm)
    RandomMotion: If True, random motion per TR (Gaussian) for Translations/Rotations with standard deviations equal to TranslationVec and RotationVec
    HeartRate: Heart rate for cardiac modelling (beats/minute)

    """

    ##
    #Define integers
    nTR = opt['nTR'].astype(np.int32)[0]

    ##
    #Synthesise spatial position map (relative to the centre of the image)
    Position = np.zeros((*Mask.shape, 3))
    Position[..., 0] = np.arange(-Mask.shape[0] / 2, Mask.shape[0] / 2)[:, np.newaxis, np.newaxis] * VoxDims[0] * Mask
    Position[..., 1] = np.arange(-Mask.shape[1] / 2, Mask.shape[1] / 2)[np.newaxis, :, np.newaxis] * VoxDims[1] * Mask
    Position[..., 2] = zLoc * VoxDims[2] * Mask

    ##
    #Obtain Cardiac Velocity Profile
    if np.any(CardiacVec) == True:
        CardiacVelocity = CardiacMotionProfile(opt,CardiacVec,HeartRate,Mask,VoxDims,zLoc)

    ##
    #Initialise Matrices
    Translation = np.zeros((*Mask.shape, 3), dtype='f8')
    Rotation = np.zeros((*Mask.shape, 3), dtype='f8')
    RotCross = np.zeros((*Mask.shape, 3))
    V = np.zeros((*Mask.shape,nTR))

    ##
    #Perform motion estimation
    for k in range(0, nTR):

        ##
        #Update Translation/Rotation motion operators and rotation matrix
        if ((RandMotion == False) & (k==0)):
            Translation = (Mask[..., np.newaxis] * TranslationVec[np.newaxis,np.newaxis,np.newaxis,:]).astype('f8')
            Rotation = (Mask[..., np.newaxis] * RotationVec[np.newaxis,np.newaxis,np.newaxis,:]).astype('f8')
            RotMat = Rotxyz(Rotation, opt)
        elif RandMotion == True:
            Translation[...,0] = Mask * np.random.normal(0, TranslationVec[0])
            Translation[...,1] = Mask * np.random.normal(0, TranslationVec[1])
            Translation[...,2] = Mask * np.random.normal(0, TranslationVec[2])
            Rotation[...,0]  = Mask * np.random.normal(0, RotationVec[0])
            Rotation[...,1]  = Mask * np.random.normal(0, RotationVec[1])
            Rotation[...,2]  = Mask * np.random.normal(0, RotationVec[2])
            RotMat = Rotxyz(Rotation, opt)
        
        ##
        #Calculate V(t)
        RotCross[..., 0] = (opt['GOr'][1] * Rotation[..., 2] - opt['GOr'][2] * Rotation[..., 1]) * Position[..., 0]
        RotCross[..., 1] = (opt['GOr'][2] * Rotation[..., 0] - opt['GOr'][0] * Rotation[..., 2]) * Position[..., 1]
        RotCross[..., 2] = (opt['GOr'][0] * Rotation[..., 1] - opt['GOr'][1] * Rotation[..., 0]) * Position[..., 2]
        TranslationGrad = Translation * opt['GOr']
        V[..., k] = np.sum(TranslationGrad + RotCross, axis=-1)

        ##
        #Update position based on Translational/Rotational motion
        Position[..., 0] = RotMat[..., 0, 0] * Position[..., 0] + RotMat[..., 0, 1] * Position[..., 1] + RotMat[..., 0, 2] * Position[..., 2] 
        Position[..., 1] = RotMat[..., 1, 0] * Position[..., 0] + RotMat[..., 1, 1] * Position[..., 1] + RotMat[..., 1, 2] * Position[..., 2] 
        Position[..., 2] = RotMat[..., 2, 0] * Position[..., 0] + RotMat[..., 2, 1] * Position[..., 1] + RotMat[..., 2, 2] * Position[..., 2] 
        Position = Position + Translation * opt['TR'][0] / 1000

    ##
    #Add impact of cardiac motion
    if np.any(CardiacVec) == True:
        V = V + CardiacVelocity

    return V

@njit()
def Rotxyz(Rotation,opt):
    #Initialise Rotation Matrix
    rotmat = np.zeros((*Rotation.shape[:-1],3,3))
    #Calculate Angles
    dx = Rotation[..., 0] * opt['TR'][0] / 1000; dy = Rotation[..., 1] * opt['TR'][0] / 1000; dz = Rotation[..., 2] * opt['TR'][0] / 1000
    #Calculate Rotation Matrix (rox*roty*rotz) - expanded to accelerate evaluations
    rotmat[...,0,0] = np.cos(dy)*np.cos(dz); rotmat[...,0,1] = -np.cos(dy)*np.sin(dz); rotmat[...,0,2] = np.sin(dy)
    rotmat[...,1,0] = np.cos(dx)*np.sin(dz) + np.sin(dx)*np.sin(dy)*np.cos(dz); rotmat[...,1,1] = np.cos(dx)*np.cos(dz)-np.sin(dx)*np.sin(dy)*np.sin(dz); rotmat[...,1,2] = -np.sin(dx)*np.cos(dy)
    rotmat[...,2,0] = np.sin(dx)*np.sin(dz) - np.cos(dx)*np.sin(dy)*np.cos(dz); rotmat[...,2,1] = np.sin(dx)*np.cos(dz)+np.cos(dx)*np.sin(dy)*np.sin(dz); rotmat[...,2,2] = np.cos(dx)*np.cos(dy) 
    
    return rotmat

@njit()
def CardiacMotionProfile(opt,CardiacVec,HeartRate,Mask,VoxDims,zLoc = 0):
    """
    Generate a cardiac motion profile and compute motion scaling.

    Parameters:
    opt : dict
        Dictionary containing necessary options:
        - 'TR' : repetition time in ms
        - 'nTR' : number of time repetitions
        - 'Velocity' : velocity field (2D or 3D)
        - 'GOr' : gradient orientation vector

    Returns:
    MotionScale : numpy.ndarray
        Computed motion scaling values.
    """

    ##
    #Obtain the Cardiac Velocity Distribution Profile
    Velocity = VelocityDistributionCardiac(CardiacVec,Mask,VoxDims,zLoc = zLoc)

    ##
    #Define Cardiac Cycle over this many points
    nPoints = int(1E5)

    ##
    #Define Cardiac Motion over this fraction of the Cardiac Cycle
    nPointsMotion = round(nPoints * 1 / 4)

    ##
    #Create HR velocity profile over a single beat (sinusoidal motion) and ensure it sums to zero
    HRProfile = np.zeros(nPoints)
    HRProfile[:nPointsMotion] = np.sin(np.linspace(np.pi / nPointsMotion, np.pi, nPointsMotion))
    HRProfile[nPointsMotion:] = np.sum(HRProfile) / (nPoints - nPointsMotion) * -1

    ##
    #Define number of TRs per heart beat
    HeartRate_Seconds = HeartRate / 60
    nTRBeat = np.round(1. / HeartRate_Seconds / (opt['TR'] / 1000)).astype(np.int32)

    ##
    #Define number of beats for the simulation duration
    nBeatsSimulation = opt['nTR'] / nTRBeat

    ##
    # Generate full HeartRate Profile for the simulation
    HRProfileFull = np.zeros((HRProfile.shape[0] * np.ceil(nBeatsSimulation).astype(np.int32)[0]), dtype='f8')
    for i in range(np.ceil(nBeatsSimulation).astype(np.int32)[0]):
        HRProfileFull[i * len(HRProfile):(i + 1) * len(HRProfile)] = HRProfile

    ##
    #Define Sampling Rate and Sample HR Profile
    SamplingRate = np.round(nPoints / nTRBeat).astype(np.int32)
    HRProfileFull = HRProfileFull[:((len(HRProfileFull) // SamplingRate[0]) * SamplingRate)[0]]
    n = len(HRProfileFull) // SamplingRate 
    trimmed = HRProfileFull[:(n * SamplingRate)[0]] 
    HRProfileTRs = np.array([np.mean(trimmed[i:i + SamplingRate[0]]) for i in range(0, len(trimmed), SamplingRate[0])])

    ##
    #Link to input velocity and gradient orientation
    CardiacVelocity = np.zeros((*Velocity.shape[:-1], opt['nTR'].astype(np.int32)[0]))
    for k in range(opt['nTR'].astype(np.int32)[0]):
        CardiacVelocity[..., k] = (HRProfileTRs[k] * 
                                    (Velocity[..., 0] * opt['GOr'][0] +
                                     Velocity[..., 1] * opt['GOr'][1] +
                                     Velocity[..., 2] * opt['GOr'][2]))
    return CardiacVelocity

@njit()
def VelocityDistributionCardiac(CardiacVec,Mask,VoxDims, zLoc = np.array(0)):
    """
    Generate a cardiac velocity distribution.
    
    Parameters:
    D : Input mask

    Returns:
    Velocity : numpy.ndarray
        3D velocity distribution with the same shape as D.
    """

    ##
    #Define array size
    x = np.arange(-Mask.shape[0] / 2, Mask.shape[0] / 2)[:, np.newaxis, np.newaxis] * VoxDims[0]  
    y = np.arange(-Mask.shape[1] / 2, Mask.shape[1] / 2)[np.newaxis, :, np.newaxis] * VoxDims[1] 
    z = (np.arange(-Mask.shape[2] / 2, Mask.shape[2] / 2)[np.newaxis, np.newaxis, :] + zLoc) * VoxDims[2] 

    ##
    #Generate radius function
    r = np.sqrt((x) ** 2 + (y) ** 2 + (z) ** 2)

    ##
    #Obtain velocity distribution
    Distribution = -r + 175
    Distribution = Distribution ** 3 / (175 ** 3) * Mask

    ##
    #Replicate across three components scaled by the velocity vector
    Velocity = Distribution[..., np.newaxis]*CardiacVec

    return Velocity
