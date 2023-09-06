# biofts
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
