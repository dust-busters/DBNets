![icon](icon/vlr.png)
# DBNets
Dust Busters Nets - Ensemble of NNs trained to infer the mass of gap opening planets in protoplanetary discs

## To install this library

__Note__: the current version of this tool (DBNets 1.0.0) has been developed and tested with tensorflow<=2.15> and keras 2. The newest keras 3 is not currently supported.

1) open a terminal

2) clone the repository with `git clone https://github.com/dust-busters/DBNets.git` 

2) enter the new directory with `cd DBNets`
   
3) download all the lfs files with `git lfs pull` 

5) install the library with `pip install .`

If you encounter some errors following the previous instructions, you can try to install the package in a python enviroment. To do that, you can follow the instructions below.

## Install in a virtual enviroment

1) First create a new python enviroment with `python3.10 -m venv <env_name>`

2) activate the new enviroment `source <env_name>/bin/activate`

3) follow the above instructions to install DBNets in the new python enviroment

4) Enjoy!

To use the new enviroment within a jupyter-notebook, for instance for running the examples provided, create a new jupyter kernel using

`python -m ipykernel install --name=<env_name>`.

Once this is done, it is possible to select the new kernel from any jupyter-notebook.

## Paper
Ruzza et al. (2024): https://doi.org/10.1051/0004-6361/202348421.

## Aknowledgments

Computational resources have been provided by INDACO Core facility, which is a project of High Performance Computing at the Università degli Studi di Milano (https://www.unimi.it). This work has been supported by Fondazione Cariplo,grant n° 2022-1217, from the European Union’s Horizon Europe Research & Innovation Programme under the Marie Sklodowska-Curie grant agreement No.823823 (DUSTBUSTERS) and from the European Research Council (ERC) under grant agreement no. 101039651 (DiscEvol).
