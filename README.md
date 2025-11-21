![icon](icon/vlr.png)
# DBNets2.0
Dust Busters Nets 2.0 - Simulation-based inference pipeline for characterizing disc substructures and putative embedded planets.
Unlike the first version this is only offered as a python library. DBNets2.0 introduces a powerful upgrade: it simultaneously fits the putative planet mass along with three additional disc properties that can degenerately lead to similar substructures.

Check out the paper here: ![DOI.10.1051/0004-6361/202554401](https://doi.org/10.1051/0004-6361/202554401)


## To install this library

1) open a terminal

2) clone the repository with `git clone https://github.com/dust-busters/DBNets.git` 

2) enter the new directory with `cd DBNets`

4) install the library with `pip install .`

5) download the trained models using the provided script `dbnets-download`. It should be available from the terminal after the package is correctly installed.

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

There are some tutorial notebooks available in ![this repo](examples/).
The notebook ![example_of_application_dbnets2.ipynb](examples/example_of_application_dbnets2.ipynb) contains a basic usage example of DBNets2.0.

Stay tuned for more documentation and examples, or drop me an [email](mailto:alessandro.ruzza@unimi.it)!
