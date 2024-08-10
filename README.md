# biofts

This repository contains the source code for the `biofts` package which is python package for field theoretic simulations (FTS) of polymer field theories through Complex-Langevin sampling. Although the code is applicable to a large class of polymer field theories, a particular goal of the package is simulations of liquid-liquid phase-separation of intrinsically disordered proteins (IDPs) and other processes relevant to biomolecular condensates. 

If you are new to polymer field theories and FTS, the following references will provide a good starting point:

1. Fredrickson et al., Field-Theoretic Computer Simulation Methods for Polymers and Complex Fluids, Macromolecules 2002, 35, 1, 16–39. <hlink>https://doi.org/10.1021/ma011515t</hlink>
2. V. Ganesan and G. H. Fredrickson, Field-theoretic polymer simulations, 2001 EPL 55 814. <hlink>https://doi.org/10.1209/epl/i2001-00353-8</hlink>
3. Matsen, M.W. (2005). Self-Consistent Field Theory and Its Applications. In Soft Matter (eds G. Gompper and M. Schick). <hlink>https://doi.org/10.1002/9783527617050.ch2</hlink>





A Python package for field theoretic simulations of biomolecule solutions. 

## Installation
To install the package, run the following command in the terminal:
```
pip install biofts
```
## Usage
The package can be imported as follows:
```
import biofts
```
The package contains the following modules:
* `biofts`: The main module for running simulations.


## Examples
The following examples demonstrate how to use the package.

### Example 1: Simulating a homopolymer solution

The following code simulates a homopolymer solution using the default parameters. The simulation is run for 10000 iterations and the results are saved in the folder `results`. The simulation is performed on a single CPU core.
```
import biofts

biofts.simulate_homopolymer_solution(
    iterations=10000,
    output_folder='results',
    num_cores=1
)
```
