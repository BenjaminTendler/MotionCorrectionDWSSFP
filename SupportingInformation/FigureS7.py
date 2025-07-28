# %%
##
#Load Packages
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import sys
from scipy.optimize import curve_fit

# %%
##
#Define Path to Code database
DirPath = '/Users/btendler/Documents/MotionCorrection/Paper/Code/'

##
#Define Output Path
OutputPath = '/Users/btendler/Documents/MotionCorrection/Paper/SupportingInformation/FigureS7/'

# %%
##
#Load functions
sys.path.append(''.join([DirPath,'bin']))
from EPGMotion import *
from MotionSimulation import *
from EPGForwardMotionModel import *
from Jacobian_MotionCorrection import *
from ParameterOptionsExperimental import *

# %%
##
#Read Parameters
opt=ParameterOptionsExperimental()

##
#Only perform investigation over beff = 500 data
opt["GArr"] = np.asarray([opt["GArr"][-1]], dtype='f8') 
opt["G"] = opt["GArr"]                     


# %%
##
#Conventional EPG simulates a maximum k-value pathway equal to the number of TRs. This piece of code identifies where increasing the k-value leads to no meaningful change in the signal, with subsequent thresholding to accelerate modelling
opt["kOrder"] = kOrder(opt.copy())

# %%
##
#Define Motion Operators - Translation Velocity (mm/s), Rotation Velocity (deg/s), and Maximum Cardiac Velocity (mm/s) along x,y & z-axes
Trans= np.array([0, 0, 0])
Rot = np.array([0, 0, 0])
Card = np.array([0, 0, 0])

##
#Define Voxel Dimensions 
VoxDims = np.array([1,1,1])

# %%
##
#Estimate different T(t) operators for different maximum amplitudes of the cardiac pulsatility profile

##
#Define Maximum velocity
MaxVel = np.linspace(0, 1.5, num=6)

##
#Initialise Array
T = np.zeros((int(opt['nTR']),MaxVel.shape[0]))
Signal = np.zeros((int(opt['nTR']),MaxVel.shape[0]),dtype = 'c8')


for idx, k in np.ndenumerate(MaxVel):
    Card = np.array([0, 0, k])
    T[:,idx] = np.squeeze(MotionOperator(opt.copy(),Trans,Rot,Card,opt['Mask'][:,np.newaxis,np.newaxis],VoxDims))[:,np.newaxis]
    Signal[:,idx]=EPGMotion(opt, np.squeeze(T[:,idx]))[:,np.newaxis]

# %%
##
#Add Noise

##
#Define SNR Levels
SNR = [np.inf]

##
#Estimate Noise Standard Deviation
NoiseSD = np.zeros((Signal.shape[1],len(SNR)))
for k in range(NoiseSD.shape[0]):
    for l in range(NoiseSD.shape[1]):
        NoiseSD[k,l] = np.mean(abs(Signal[int(opt["nDummy"]):,k]))/SNR[l]

##
#Number of repeats
nRepeats = 1

##
#Initialise SignalNoise repeats
SignalNoise = np.zeros((*Signal.shape,len(SNR),nRepeats),dtype = 'c8')

##
#Add Noise
for k in range(SignalNoise.shape[1]):
    for l in range(SignalNoise.shape[2]):
        for m in range(SignalNoise.shape[3]):
            SignalNoise[:,k,l,m] = (np.real(Signal[:,k]) + np.random.normal(0, NoiseSD[k,l], Signal.shape[0])) + 1j*(np.imag(Signal[:,k]) + np.random.normal(0, NoiseSD[k,l], Signal.shape[0]))

# %%
##
#Define lower and upper parameter bounds (fix S0 equal to 1, define MotionParameters between -1.5 and 1.5 mm/s))
low = [1E-6, 1, -20*np.pi, *np.ones((int(opt["nTR"])-int(opt["SteadyStateTR"])))*-5]
high = [10E-3, 1+ + np.finfo('f8').eps, 20*np.pi, *np.ones((int(opt["nTR"])-int(opt["SteadyStateTR"])))*5]

# %%
##
#Fitting without Motion Information

##
#Initialise fitting parameters (D, S0, Phi)
par_init = [0.5E-4, 1, np.pi/2]

##
#Initialise Array
poptnoMotion = np.zeros((3,*SignalNoise.shape[1:]))

##
#Perform Fitting
for k in range(SignalNoise.shape[1]):
    for l in range(SignalNoise.shape[2]):
        for m in range(SignalNoise.shape[3]):
            #Input Data (1D array: Real & Imaginary Components - Fitting to After Dummy Region)
            Data = np.concatenate((np.real(SignalNoise[int(opt["nDummy"]):,k,l,m]).reshape(SignalNoise[int(opt["nDummy"]):,k,l,m].shape[0]),np.imag(SignalNoise[int(opt["nDummy"]):,k,l,m]).reshape(SignalNoise[int(opt["nDummy"]):,k,l,m].shape[0])))
            #Perform Fitting
            poptnoMotion[:,k,l,m], pcov, infodict, mesg, ier  = curve_fit(lambda x, *theta: EPGForwardModelFitting(x, theta, opt.copy()), 1, Data, p0=par_init, method='trf',absolute_sigma=False,bounds=(low[0:3],high[0:3]),verbose=1,jac=lambda x, *theta: EPGForwardModelFittingJacobian(x,theta,opt.copy()), x_scale='jac',full_output=True,tr_solver='exact',max_nfev=1E5,ftol=1e-5, xtol=1e-5, gtol=1e-5)


# %%
##
#Fitting with Motion Information

##
#Initialise fitting parameters (D, S0, Phi, MotionVector)
par_init = [0.5E-4, 1, np.pi/2, *np.zeros(int(opt["nTR"])-int(opt["SteadyStateTR"]))]

##
#Initialise Array
poptMotion = np.zeros((len(low),*SignalNoise.shape[1:]))

##
#Perform Fitting
for k in range(SignalNoise.shape[1]):
    for l in range(SignalNoise.shape[2]):
        for m in range(SignalNoise.shape[3]):
            #Input Data (1D array: Real & Imaginary Components - Fitting to After Dummy Region)
            Data = np.concatenate((np.real(SignalNoise[int(opt["nDummy"]):,k,l,m]).reshape(SignalNoise[int(opt["nDummy"]):,k,l,m].shape[0]),np.imag(SignalNoise[int(opt["nDummy"]):,k,l,m]).reshape(SignalNoise[int(opt["nDummy"]):,k,l,m].shape[0])))
            #Perform Fitting
            poptMotion[:,k,l,m], pcov, infodict, mesg, ier  = curve_fit(lambda x, *theta: EPGForwardMotionModelFitting(x, theta, opt.copy()), 1, Data, p0=par_init, method='trf',absolute_sigma=False,bounds=(low,high),verbose=1,jac=lambda x, *theta: EPGForwardMotionModelFittingJacobian(x,theta,opt.copy()), x_scale='jac',full_output=True,tr_solver='exact',max_nfev=1E5,ftol=1e-5, xtol=1e-5, gtol=1e-5)

# %%
##
#Reconstruct Signal & Motion Profile

#Create Reconstructed Motion Array
TRecon = np.zeros_like(SignalNoise, dtype = 'f8')
TRecon[-(int(opt["nTR"])-int(opt["SteadyStateTR"])):,...] = poptMotion[3:,...]

#Initialise Signal Array
SignalRecon = np.zeros_like(SignalNoise, dtype = 'c8')

##
#Reconstruct Signal
for k in range(SignalNoise.shape[1]):
    for l in range(SignalNoise.shape[2]):
        for m in range(SignalNoise.shape[3]):
            ##
            #Declare outputs from fitting to the options file 
            optRecon = opt.copy()
            optRecon['D'] = np.asarray([poptMotion[0,k,l,m]], dtype='f8')   
            optRecon['phi'] = np.asarray([poptMotion[2,k,l,m]], dtype='f8') 

            ##
            #Reconstruct signal
            SignalRecon[:,k,l,m] = EPGMotion(optRecon.copy(), TRecon[:,k,l,m])


# %%
##
#Get Mean & Standard deviation (Original Signal)
SignalNoiseMean = np.squeeze(np.mean(SignalNoise,axis=-1))
SignalNoiseSD = np.squeeze(np.std(SignalNoise,axis=-1))

##
#Get Mean & Standard deviation (Reconstructed Signal)
SignalReconMean = np.squeeze(np.mean(SignalRecon,axis=-1))
SignalReconSD = np.squeeze(np.std(SignalRecon,axis=-1))

##
#Get Mean & Standard deviation (Motion Profile)
TReconMean = np.squeeze(np.mean(TRecon,axis=-1))
TReconSD = np.squeeze(np.std(TRecon,axis=-1))

# %%
fig, axs = plt.subplots(6, 1)
fig.set_size_inches(8,24)
#Define x-axis
Time = range(Signal.shape[0])*opt["TR"]/1E3
#Plot Velocity Profile
axs[0].plot(Time,T[:,0],'#1f77b4',linewidth=2,label = r'$T(x) (V_{max} =$ 0 mm/s)')
axs[0].plot(Time,TReconMean[:,0],'#ff7f0e',linewidth=2,linestyle='--',label = 'Estimated T(x)')
axs[0].axvspan(Time[int(opt["SteadyStateTR"])],Time[int(opt["nDummy"])], alpha=0.1,color='#d62728',label = 'Dummy Measurements')
axs[0].axvspan(Time[int(opt["nDummy"])],Time[-1],alpha=0.1,color='#2ca02c',label = 'Measured Data')
#Plot Velocity Profile
axs[1].plot(Time,T[:,1],'#1f77b4',linewidth=2,label = r'$V_{max} =$ 0.3 mm/s')
axs[1].plot(Time,TReconMean[:,1],'#ff7f0e',linewidth=2,linestyle='--')
axs[1].axvspan(Time[int(opt["SteadyStateTR"])],Time[int(opt["nDummy"])], alpha=0.1,color='#d62728')
axs[1].axvspan(Time[int(opt["nDummy"])],Time[-1],alpha=0.1,color='#2ca02c')
#Plot Velocity Profile
axs[2].plot(Time,T[:,2],'#1f77b4',linewidth=2,label = r'$V_{max} =$ 0.6 mm/s')
axs[2].plot(Time,TReconMean[:,2],'#ff7f0e',linewidth=2,linestyle='--')
axs[2].axvspan(Time[int(opt["SteadyStateTR"])],Time[int(opt["nDummy"])], alpha=0.1,color='#d62728')
axs[2].axvspan(Time[int(opt["nDummy"])],Time[-1],alpha=0.1,color='#2ca02c')
#Plot Velocity Profile
axs[3].plot(Time,T[:,3],'#1f77b4',linewidth=2,label = r'$V_{max} =$ 0.9 mm/s')
axs[3].plot(Time,TReconMean[:,3],'#ff7f0e',linewidth=2,linestyle='--')
axs[3].axvspan(Time[int(opt["SteadyStateTR"])],Time[int(opt["nDummy"])], alpha=0.1,color='#d62728')
axs[3].axvspan(Time[int(opt["nDummy"])],Time[-1],alpha=0.1,color='#2ca02c')
#Plot Velocity Profile
axs[4].plot(Time,T[:,4],'#1f77b4',linewidth=2,label = r'$V_{max} =$ 1.2 mm/s')
axs[4].plot(Time,TReconMean[:,4],'#ff7f0e',linewidth=2,linestyle='--')
axs[4].axvspan(Time[int(opt["SteadyStateTR"])],Time[int(opt["nDummy"])], alpha=0.1,color='#d62728')
axs[4].axvspan(Time[int(opt["nDummy"])],Time[-1],alpha=0.1,color='#2ca02c')
#Plot Velocity Profile
axs[5].plot(Time,T[:,5],'#1f77b4',linewidth=2,label = r'$V_{max} =$ 1.5 mm/s')
axs[5].plot(Time,TReconMean[:,5],'#ff7f0e',linewidth=2,linestyle='--')
axs[5].axvspan(Time[int(opt["SteadyStateTR"])],Time[int(opt["nDummy"])], alpha=0.1,color='#d62728')
axs[5].axvspan(Time[int(opt["nDummy"])],Time[-1],alpha=0.1,color='#2ca02c')
for k in range(6):
    axs[k].set_xlim([0,Time[-1]])
    axs[k].set_ylim([-2,2])
    axs[k].set_ylabel('Velocity (mm/s)',fontsize=12)
    axs[k].set_xlabel('Time (s)',fontsize=12)
    axs[k].legend(fontsize=14,loc='upper right', )


# %%
#Save Figure
fig.savefig(''.join([OutputPath,'FigureS7.png']),dpi=300,format='png',bbox_inches='tight')

# %%
##
#Load Data
poptnoMotion = np.zeros((3, 6, 1, 10))
poptMotion = np.zeros((103, 6, 1, 10))

for k in range(10):
    Data = np.load(''.join(['/Users/btendler/Documents/MotionCorrection/Paper/Figures/Figure6/','Data/',str(k+1),'Data.npz']))
    poptnoMotion[:,:,:,k] = Data['arr_1'][:,:,-1,0][:,:,np.newaxis]
    poptMotion[:,:,:,k] = Data['arr_2'][:,:,-1,0][:,:,np.newaxis]

# %%
np.max(np.abs(np.mean((poptMotion[0,:,:,:]-0.001)/0.001*10**2,axis=-1)))

# %%
import os


