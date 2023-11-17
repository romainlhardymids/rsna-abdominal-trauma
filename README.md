# RSNA Abdominal Trauma Detection Challenge

# Introduction
This repository contains the training code for my solution to the [RSNA Abdominal Trauma Detection challenge](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/overview), which placed 23rd on the public leaderboard and 24th on the private leaderboard. My solution consists of three stages: 

1. 3D segmentation
2. Slice-level feature extraction
3. Scan-level classification

An overview of the solution is shown in the figure below.

TODO: ADD FIGURE

# Segmentation
The segmentation stage aims to extract bounds for relevant organs in the patient scans (bowel, liver, kidney, and spleen), which we can then leverage in downstream classification models. My choice of segmentator architecture is a 3D UNet model with dimensions 128 x 224 x 224, which I construct by inflating the modules of a 2D UNet model. The model is trained on the subset of patients with available segmentation labels, and used to predict segmentation probability maps for the full set of patients.
