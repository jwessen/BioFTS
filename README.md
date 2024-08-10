# biofts

This repository contains the source code for the `biofts` package which is python package for field theoretic simulations (FTS) of polymer field theories through Complex-Langevin sampling. Although the code is applicable to a large class of polymer field theories, a particular goal of the package is simulations of liquid-liquid phase-separation of intrinsically disordered proteins (IDPs) and other processes relevant to biomolecular condensates. 

# Background

If you are new to polymer field theories and FTS, the following references will provide a good starting point:

1. G. H. Fredrickson, V. Ganesan and F. Drolet, Field-Theoretic Computer Simulation Methods for Polymers and Complex Fluids, Macromolecules 2002, 35, 1, 16–39. <hlink>https://doi.org/10.1021/ma011515t</hlink>
2. V. Ganesan and G. H. Fredrickson, Field-theoretic polymer simulations, 2001 EPL 55 814. <hlink>https://doi.org/10.1209/epl/i2001-00353-8</hlink>
3. M. W. Matsen (2005). Self-Consistent Field Theory and Its Applications. In Soft Matter (eds G. Gompper and M. Schick). <hlink>https://doi.org/10.1002/9783527617050.ch2</hlink>

For the serious reader, the following book beautifully covers the topic:
4.  G. H. Fredrickson and K. T. Delaney. (2023). Field-Theoretic Simulations in Soft Matter and Quantum Fluids. Oxford University Press.

For an application of FTS to IDPs with both long-range electrostatic interactions and short-range residue-specific interactions, see:
5. J. Wessén, S. Das., T. Pal, H. S. Chan (2021). Analytical Formulation and Field-Theoretic Simulation of Sequence-Specific Phase Separation of Protein-Like Heteropolymers with Short- and Long-Spatial-Range Interactions, J. Phys. Chem. B 2022, 126, 9222−9245, <hlink>https://doi.org/10.1021/acs.jpcb.2c06181</hlink>

# Overview

`biofts' can be used to study any polymer field theory on the form

\begin{equation}
x = y + z
\end{equation}





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
