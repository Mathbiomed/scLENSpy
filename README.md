# scLENS-py Setup and Usage

This document provides instructions on setting up scLENS-py.

---

Project Setup and Usage
This document provides instructions on how to set up the environment and get started with the project.

✅ Prerequisites
---
For a straightforward setup, we highly recommend installing the Anaconda Distribution. It includes Python, the conda package manager, and other essential scientific computing libraries.

Download Anaconda here: https://www.anaconda.com/download


🛠️ Installation
---

Please follow these two steps to set up your environment.

### **1. Create the Conda Environment**

First, create the base conda environment using the provided `environment.yml` file. This will install all necessary packages except for PyTorch.

Bash

```
conda env create -f environment.yml
```

After the environment is created, you must **activate** it:

Bash

```
conda activate sclens-py 
```

_(**Note:** You can find the correct environment name inside the `environment.yml` file, specified under the `name:` key.)_

### **2. Install PyTorch**

Next, install PyTorch with CUDA 12.8 support using pip. This step is performed separately to ensure the correct GPU-accelerated version is installed.

Bash

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> For different system configurations (e.g., CPU-only, different CUDA versions, or macOS), please generate the appropriate installation command on the **official PyTorch website**: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## ▶️ Usage

For detailed examples and a guide on how to use the code, please see the **`Tutorial.ipynb`** notebook.
