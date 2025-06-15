![icon](icon/vlr.png)
# DBNets.0
Dust Busters Nets 2.0 - Simulation-based inference pipeline for characterizing disc substructures and putative embedded planets.
Unlike the first version this is only offered as a python library. DBNets2.0 introduces a powerful upgrade: it simultaneously fits the putative planet mass along with three additional disc properties that can degenerately lead to similar substructures.

Check out the paper here: __available from monday 16th of Jun 2025__


## To install this library

1) open a terminal

2) clone the repository with `git clone https://github.com/dust-busters/DBNets.git` 

2) enter the new directory with `cd DBNets`
   
3) switch to the correct branch with `git checkout dbnets2.0.0` 

5) install the library with `pip install .`

6) download the trained models from ![here](https://dbnets.fisica.unimi.it/dbnets2.0_models/dbnets2.0_models.tar.gz).

If you encounter some errors following the previous instructions, you can try to install the package in a python enviroment. To do that, you can follow the instructions below.

## Install in a virtual enviroment

1) First create a new python enviroment with `python3.10 -m venv <env_name>`

2) activate the new enviroment `source <env_name>/bin/activate`

3) follow the above instructions to install DBNets in the new python enviroment

4) Enjoy!

To use the new enviroment within a jupyter-notebook, for instance for running the examples provided, create a new jupyter kernel using

`python -m ipykernel install --name=<env_name>`.

Once this is done, it is possible to select the new kernel from any jupyter-notebook.

## Tutorials

There are some tutorial notebooks available in ![this repo](examples/). Stay tuned for more documentation and examples, or drop me an ![email](mailto:alessandro.ruzza@unimi.it)!
