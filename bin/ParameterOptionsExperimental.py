import numpy as np
from numba.core import types
from numba.typed import Dict

def ParameterOptionsExperimental():
    opt = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64[:],
    )

    # Scanner properties
    opt['gamma'] = np.asarray([2 * np.pi * 42.58 * 10**6], dtype='f8')  # Gyromagnetic ratio
    
    # Sample Properties
    opt['T1'] = np.asarray([1257], dtype='f8')  # T1 (ms) - Median T1 estimate from the acquired T1 map 
    opt['T2'] = np.asarray([91], dtype='f8')  # T2 (ms) - Median T2 estimate from the acquired T1 map 
    opt['B1'] = np.asarray([1], dtype='f8')
    opt['D'] = np.asarray([1E-3], dtype='f8')  # Diffusion coefficient (mm^2/s)
    opt['Mask'] = np.asarray([1], dtype='f8')  # Diffusion coefficient (mm^2/s)
    
    # Initial sequence parameters - SSFP
    opt['GArr'] = np.asarray([26.7, 63.6], dtype='f8')  # Diffusion Gradient Amplitude (mT/m)
    opt["G"] = opt["GArr"]                      # Define Gradient Amplitude based on options file (This simply enables us to maintain a single repository of code across the simulation and experimental domains)
    opt['tau'] = np.asarray([3.08], dtype='f8')  # Diffusion Gradient Duration (ms)
    opt['TR'] = np.asarray([26], dtype='f8')  # Repetition Time (ms)
    opt['alpha'] = np.asarray([31], dtype='f8')  # Flip Angle (degrees)
    opt['phi'] = np.asarray([0], dtype='f8')  # RF Phase
    
    # Simulation properties
    opt['GOr'] = np.asarray([0, 0, 1], dtype='f8')  # Gradient orientation
    opt['SteadyStateTR'] = np.asarray([100], dtype='f8')  # Minimum TR before sampling (when system is in steady-state)
    opt['nDummy'] = opt['SteadyStateTR'] + np.asarray([25], dtype='f8')  # Additional dummy scans prior to sampling
    opt['nTR'] = np.asarray([384], dtype='f8') + opt['nDummy'] # Number of TRs - set first number to number of experimental TRs
    opt['kOrder'] = np.asarray([23], dtype='f8') #opt['nTR']  # Maximum k-order
    
    return opt
