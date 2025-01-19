# Lin-Zhang-Week2-Reflection-Exercise1

# Deblurring Dataset and Model

This repository contains a dataset for training and evaluating deblurring models. It includes both blurry images and their corresponding sharp images, allowing for model training using paired data.

## Repository Structure

- **data/**: Contains the dataset, including blurry and sharp images.
- **code/**: Contains the Python code, including Jupyter Notebooks for exploratory data analysis and model implementation.
- **README.md**: This file, containing setup instructions.


## Prerequisites

### General Requirements
- **Python 3.8 or higher**
- **Google Colab** (Cloud environment setup)
- **Internet connection** for installing dependencies

### Python Dependencies
Make sure to install the following Python libraries:
- **numpy**
- **pandas**
- **matplotlib**

---

## System Setup Instructions

### Cloud Environment Setup (Google Colab)

1. **Upload the notebook to Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/).
   - Upload the provided Jupyter Notebook (`your_notebook.ipynb`).

2. **Run the notebook**:
   - Now, you can execute the cells in the notebook directly on Google Colab.


### Local Environment Setup (macOS - M2 MacBook Pro)

1. **Install Python**:
   If you don't have Python installed, you can download and install the latest version of Python directly from the official website:  
   [Download Python](https://www.python.org/downloads/)

2. **Install Jupyter Notebook**:
   Once Python is installed, open a terminal and install Jupyter Notebook using `pip`:
   ```bash
   pip install jupyter

3. **Run Jupyter Notebook**:
   To start Jupyter Notebook, run the following command in your terminal, which will open the Jupyter Notebook interface in your web browser:
   ```bash
   jupyter notebook

4. **Import Libraries**:
   Once the Jupyter Notebook is running, you can import the libraries in your notebook as follows:
   ```bash
   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
