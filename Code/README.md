README For Week 2 Reflection
# Image Dataset Exploration

This notebook performs exploratory data analysis (EDA) on an image dataset consisting of blurry and clean image pairs. The goal is to help you visualize the dataset and understand its key characteristics, including image sizes and brightness distributions.

## Overview
- This notebook loads blurry and clean images from two separate folders.
- It randomly selects a few pairs of blurry and clean images to display.
- It calculates and visualizes the distribution of image sizes (width and height) in the dataset.
- It calculates and visualizes the distribution of image brightness levels.

## Dataset Structure
- `input_noise`: Folder containing blurry images.
- `target`: Folder containing corresponding clean images.

**Note**: Replace the paths in the notebook with the paths where your dataset is stored locally.

## Requirements
- Python 3.x
- `numpy`
- `matplotlib`
- `PIL` (Python Imaging Library)

## How to Use
1. Clone or download the repository.
2. Ensure you have the required libraries.





--------------------------------


README for Problem Set 2


# README for Image Denoising and Causal Inference Analysis

## 1. Overview:
This repository contains code that performs the following tasks:
- **Image Denoising**: Using deep learning models to denoise images corrupted by noise and blur.
- **Model Prediction**: Evaluating the performance of image denoising models using metrics such as PSNR, SSIM, and LPIPS.
- **Causal Inference Analysis**: Applying Regression Discontinuity Design (RD) for causal inference in the context of image quality, using noise and blur levels as treatment indicators.

The main focus of the code is to evaluate the effectiveness of the image denoising process, while also performing causal inference to study the relationship between image quality and treatment (denoising) at different noise levels.


## 2. Prerequisites:
### Required Libraries:
To run the code, ensure you have Python 3.9+ installed and the required libraries imported:

```bash
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
```

## 3. Usage Instructions:

### Step-by-Step Guide to Running the Code:
**1. Load and Preprocess Data**:

Place your noisy images in the input_noise folder and corresponding clean images in the target folder.<br>
The code will automatically load and preprocess images by resizing them to 128x128 pixels and normalizing them.

**2. Run Image Denoising**:

Define and train the denoising model (WEDDM) using the noisy and clean images. The model is trained for a specified number of epochs.<br>
During training, loss is computed using Mean Squared Error (MSE) between predicted and clean images.

**3. Evaluate Image Quality**:

After training the model, the code evaluates the performance using three metrics:
- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image quality.
- **SSIM (Structural Similarity Index)**: Measures similarity in structure between images.
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual similarity between images.

**4. Run Causal Inference (RD Analysis)**:

Simulate data that includes treatment (denoising) and different levels of noise and blur.<br>
Perform a Regression Discontinuity Design (RD) analysis to investigate the causal effect of treatment (denoising) on image quality.<br>
Visualize the RD plot and analyze the results.

## 4. Outputs:
Visualizations: Graphs and plots showing the results of the RD analysis and saliency maps of the denoising process.<br>
Metrics: PSNR, SSIM, and LPIPS values to evaluate the model's performance.<br>
RD Analysis Results: Causal inference analysis outputs, showing treatment effects at different cutoff points.

## 5. Expected Outputs:
Saliency Maps: Visualizations showing which parts of the image the model focuses on during the denoising process.<br>
PSNR, SSIM, and LPIPS Metrics: Evaluation of the model's denoising performance using these three metrics.<br>
RD Plot: A graph showing the effect of treatment (denoising) at different noise levels, with a specified cutoff for Regression Discontinuity Design.<br>
RD Analysis Results: Causal inference analysis results, providing insights into how treatment influences PSNR values across noise levels.

