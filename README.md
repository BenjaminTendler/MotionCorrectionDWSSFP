# MotionCorrectionDWSSFP

This repository provides software associated with ongoing work performing motion-estimation in DW-SSFP. The software is subject to change at any moment whilst the project is under development. For any questions, please email benjamin.tendler@ndcn.ox.ac.uk.

--- 

This repository contains example software to:
- Perform forward simulations of DW-SSFP timeseries data incorporating motion corruption in a single voxel, and subsequent parameter estimation from the DW-SSFP signal (VoxelSimulationExample.ipynb)
- Perform forward simulations of DW-SSFP timeseries imaging data incorporating motion corruption (ImageSimulationExample.ipynb)

# Software requirements

- Software written in Python (3.8.12) 
- Created using the numpy (1.24.3), numba (0.58.1), scipy (1.10.1), matplotlib (3.7.5), nibabel (5.2.1) & integrated into jupyter notebooks
- To install environment in conda, run _conda create -n "MotionCorrectionDWSSFP" python numpy numba scipy matplotlib nibabel ipykernel_
- Once installed, enter the environment with _conda activate MotionCorrectionDWSSFP_

---

# Example

The below figure displays simulated DW-SSFP images incorporating (a) no motion, (b) translational motion, (c) rotational motion, (d) brain pulsatility & (e) pulsatility with a multi-shot readout and added Gaussian noise. 

All images were synthesised using the software provided in this repository.

![Example simulated motion-corrupted DW-SSFP images](https://github.com/BenjaminTendler/MotionCorrectionDWSSFP/blob/main/DWSSFP.png)



