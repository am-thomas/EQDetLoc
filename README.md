# EQDetLoc
Detecting and Locating Earthquakes in East Africa

## Installation 
* Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for your OS 
* Create a new conda environment with pygmt: 
```
conda create --name detloc --channel conda-forge pygmt python=3.10
```
* Activate the new conda environment as follows: 
```
conda activate detloc
```
* Install Obspy 
```
conda install -c conda-forge obspy
```
* Install seisbench 
```
pip install seisbench
```
* Download the EQDetLoc repository in your desired directory
```
git clone https://github.com/am-thomas/EQDetLoc.git
```
