# Multi-Frame OCT and Fundus Image Analysis for BCVA Prediction

This repository contains the source code for a program that predicts Best Corrected Visual Acuity (BCVA) in patients by analyzing multiple frames of Macular Optical Coherence Tomography (OCT), multiple frames of Optic Disc OCT, and fundus images across various eye diseases.

## Overview

The program utilizes advanced image processing and machine learning techniques to analyze the structural changes in the retina and optic disc, which are indicative of different eye conditions. By examining multiple frames, the program can provide a more comprehensive and accurate prediction of a patient's BCVA.

## Key Features

- **Multi-Frame Analysis**: The program can process multiple frames of OCT and fundus images to capture a broader range of retinal and optic disc features.
- **Cross-Disease Prediction**: It is designed to work across different eye diseases, making it a versatile tool for ophthalmologists.
- **High Accuracy**: Employs state-of-the-art algorithms to ensure high prediction accuracy.

## Getting Started

To get started with the program, follow these steps:

train_multimodel_fusion_3mod.py  This code uses ResNet50 as the backbone to determine the patient's BCVA by fusing macular information, optic disc information, and fundus information separately.

Through our experiments, this method has proved that it has obvious performance advantages over the prediction only relying on a single mode.


