
# Recommendend installation
It is a good practice to use virtual environments when programming in order to separate the configuration of each project
## Conda enviroment installation
Download anaconda: https://www.anaconda.com/download  
Install anaconda  
Open a terminal on linux or the anaconda powershell prompt on windows  
Type in the following instructions:  

```
conda create -n AML_RL python=3.10 pip
conda activate AML_RL
pip install -r requirements.txt
```

Alternatively you can open the Anaconda Navigator, create an environment called AML_RL with python 3.10 and install the packages listed in requirements.txt one by one   

# Use the Conda environment
Open the Anaconda Navigator, select AML_RL environment and launch jupyter-notebook  
OR  
Open a terminal on linux or the anaconda powershell prompt on windows  
Type in the following instructions:  

```
conda activate AML_RL
jupyter-notebook
```
Load the assignments