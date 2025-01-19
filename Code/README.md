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
2. Ensure you have the required libraries installed by running:
   ```bash
   pip install numpy matplotlib Pillow

