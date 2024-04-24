# End-to-End ML Project README

This repository contains the code for an end-to-end machine learning project. Below are the steps to set up and run the project.

## Steps to Set Up the Project

### Step 1: Create a New Environment

```bash
conda create -p venv python==3.8
conda activate venv/
```


### Step 2: Create a .gitignore File
Create the file by right-clicking and include the venv directory in it.


### Step 3: Create a requirements.txt File
pip install -r requirements.txt


### Step 4: Create a setup.py File
This file is used to install the entire project as a package. Additionally, it contains a function to read the packages from requirements.txt.

### Step 5: Create a src Folder
Include exception.py, logger.py, and utils.py files in this folder. Make this folder a package by including an __init__.py file. The src folder will also include another folder named components.

#### Step 5.1: Create a components Folder
Include data_ingestion.py, data_transformation.py, model_trainer.py, and an __init__.py file. These components are interconnected.


