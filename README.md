# Lin-Zhang-Week2-Reflection-Exercise1

# Deblurring Dataset and Model

This repository contains a dataset for training and evaluating deblurring models. It includes both blurry images and their corresponding sharp images, allowing for model training using paired data.

## Repository Structure

- **data/**: Contains the dataset, including blurry and sharp images.
- **code/**: Contains the Python code, including Jupyter Notebooks for exploratory data analysis and model implementation (`Week2ReflectionCode.ipynb`). And Python code for explanation, prediction, and causal inference of revised research topic (`ProblemSet2.ipynb`).
- **README.md**: This file, containing setup instructions.


## Prerequisites

### General Requirements
- **Python 3.8 or higher**
- **Google Colab** (Cloud environment setup)
- **Internet connection** for installing dependencies

### Python Dependencies
Make sure to install the following Python libraries listed in the below section

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
   **Once the Jupyter Notebook is running, you can import the libraries in your notebook as follows:**<br>
   For `Week2ReflectionCode.ipynb`：
   ```bash
   import os
   import numpy as np
   import random
   import matplotlib.pyplot as plt
   from PIL import Image
   ```

   For `ProblemSet2.ipynb`：
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
