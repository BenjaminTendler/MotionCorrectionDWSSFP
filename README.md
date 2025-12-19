# MotionCorrectionDWSSFP

This repository provides software associated with ongoing work performing motion-estimation in DW-SSFP. The software is subject to change at any moment whilst the project is under development. For any questions, please email benjamin.tendler@ndcn.ox.ac.uk.

--- 

This repository contains example software to:
- Perform forward simulations of DW-SSFP timeseries data incorporating motion corruption in a single voxel, and subsequent parameter estimation from the DW-SSFP signal (_VoxelSimulation.ipynb_)
- Perform forward simulations of DW-SSFP timeseries imaging data incorporating motion corruption (_ImageSimulation.ipynb_)
- Perform parameter estimation from experimental DW-SSFP timeseries data acquired in a Thalamus, Corpus Callosum \& CSF voxel (_VoxelSimulation_ExperimentalData.ipynb_)

# Software requirements

- Software written in Python (3.13.2) 
- Created using numpy (2.2.4), numba (0.61.2), scipy (1.15.2), matplotlib (3.10.0), nibabel (5.3.2) & integrated into jupyter notebooks
- To install environment in conda, run _conda create -n "MotionCorrectionDWSSFP" python=3.13.2 numpy=2.2.4 numba=0.61.2 scipy=1.15.2 matplotlib=3.10.0 nibabel=5.3.2 ipykernel_
- Once installed, enter the environment with _conda activate MotionCorrectionDWSSFP_

---

# Example

The below figure displays simulated DW-SSFP images incorporating (a) no motion, (b) translational motion, (c) rotational motion, (d) brain pulsatility & (e) pulsatility with a multi-shot readout and added Gaussian noise. 

All images were synthesised using the software provided in this repository.

![Example simulated motion-corrupted DW-SSFP images](https://github.com/BenjaminTendler/MotionCorrectionDWSSFP/blob/main/DWSSFP.png)

# Copyright

Copyright, 2024, University of Oxford. All rights reserved



