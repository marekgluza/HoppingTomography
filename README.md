# Hopping tomography

## Installation
Clone the project, make sure that you have cvxpy and Gaussian fermions and run the notebooks to perform some exemplary tomographic reconstructions.

## Dependencies
The package uses 

1.cvxpy 

available from pip, and standard libs like 

2. numpy and 

3. matplotlib.

4. Additionally it makes use of a package with routines for Gaussian fermionic sustems which can be installed in the newest version via source
```bash
git clone git@github.com:marekgluza/Gaussian_fermions.git
cd Gaussian_fermions
pip3 install .
```
or Via pip:
```bash
pip install gaussian_fermions
``` 

## How to run a simple tomographic reconstrcuction:
The repository includes two notebooks which run the reconstruction from simulated data. 
The main text ipynb shows how to reconstruct based on a thermal state and in the appendix reconstructions are shown based on out-of-equilibrium input states.
There are two notebooks for tomography, one produces the main text thermal reconstruction and the other produces the reconstructions for the appendix.



