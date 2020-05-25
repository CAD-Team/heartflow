In order to make sure that we're using the same packages, please use the `hughes-min` environment included with this repo. 

To create it, open a terminal (or an Anaconda powershell, depending on how you installed Anaconda) in this directory and run:

`conda env create -f environment.yml`

Before running the scripts, activate the environment:

`conda activate hughes-min`

To run the simulations, which currently use scipy's iterative solvers, be sure to specify scipy matrices with `--matrix=Scipy`. For example:

`python model_00_cropped_01_autoplaque_dirichlet.py --matrix=Scipy`