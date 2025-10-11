
# Data Processing Workshop 

This notebook can be accessed via Binder. Click here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Lachie-Perris/One_tree_island_data_analysis.git/HEAD)

Alternatively, you can install it locally using Anaconda.

## Setup Instructions: GitHub Repo with Anaconda Environment

## Step 1: Install Anaconda

If Anaconda is not already installed, download and install it from the official website: [Anaconda Downloads](https://www.anaconda.com/products/distribution)

Follow the installation instructions for your operating system.

## Step 2: Clone the GitHub Repository

Open a terminal or command prompt and run the following command:

    git clone https://github.com/Lachie-Perris/One_tree_island_data_analysis.git

## Step 3: Create and Activate a Virtual Environment

Navigate to the cloned repository:

    cd path/to/cloned/repository

Create a new virtual environment using Anaconda:

    conda create --name onetree python=3.10

Activate the virtual environment:

  On Windows:
  
    conda activate onetree
    
  On macOS/Linux:
  
    source activate onetree

## Step 4: Install Dependencies from Requirements File

    conda install --file requirements.txt
if this fails, then try;

    pip install -r requirements.txt

## Ready to Use
You've successfully cloned the GitHub repository and installed its dependencies in the Anaconda environment. You can now go ahead with your work in the activated virtual environment.

Please remember to activate the virtual environment (conda activate onetree) whenever you work on this project to use the specific dependencies installed in the virtual environment.

Now you should be able to start the Jupyter notebook server by running the command Jupyter Notebook on the terminal for Anaconda prompt:
  
    jupiter notebook 

In the web browser, you will see the Jupyter Notebook dashboard. From here, you can navigate through your file system, open existing notebooks, and create new ones.


