# biofts

This repository contains the source code for the `biofts` package which is python package for field theoretic simulations (FTS) of polymer field theories through Complex-Langevin sampling. Although the code is applicable to a large class of polymer field theories, a particular goal of the package is simulations of liquid-liquid phase-separation of intrinsically disordered proteins (IDPs) and other processes relevant to biomolecular condensates. 

## Background

If you are new to polymer field theories and FTS, the following references will provide a good starting point:

1. G. H. Fredrickson, V. Ganesan and F. Drolet, Field-Theoretic Computer Simulation Methods for Polymers and Complex Fluids, Macromolecules 2002, 35, 1, 16–39. <hlink>https://doi.org/10.1021/ma011515t</hlink>
2. V. Ganesan and G. H. Fredrickson, Field-theoretic polymer simulations, 2001 EPL 55 814. <hlink>https://doi.org/10.1209/epl/i2001-00353-8</hlink>
3. M. W. Matsen (2005). Self-Consistent Field Theory and Its Applications. In Soft Matter (eds G. Gompper and M. Schick). <hlink>https://doi.org/10.1002/9783527617050.ch2</hlink>

For the serious reader, the following book beautifully covers the topic:

4.  G. H. Fredrickson and K. T. Delaney. (2023). Field-Theoretic Simulations in Soft Matter and Quantum Fluids. Oxford University Press.

For an application of FTS to IDPs with both long-range electrostatic interactions and short-range residue-specific interactions, see:

5. J. Wessén, S. Das., T. Pal, H. S. Chan. Analytical Formulation and Field-Theoretic Simulation of Sequence-Specific Phase Separation of Protein-Like Heteropolymers with Short- and Long-Spatial-Range Interactions, J. Phys. Chem. B 2022, 126, 9222−9245, <hlink>https://doi.org/10.1021/acs.jpcb.2c06181</hlink>

## Overview

`biofts` can be used to study any polymer field theory described by a Hamiltonian on the form

$$
H[\lbrace\psi_a(\mathbf{r},t) \rbrace] = -\sum_{i=1}^{M_{\rm C}} n_i \ln Q_i[\lbrace \psi_a \rbrace] - \sum_{I=1}^{M_{\rm G}} z_I Q_I[\lbrace \psi_a \rbrace] + \int \mathrm{d}^d \mathbf{r} \frac{1}{2} \sum_{a} \psi_a(\mathbf{r}) \hat{V}_a^{-1} \psi_a(\mathbf{r}) 
$$

Here, $\psi_a(\mathbf{r},t)$ is a field that decouples interactions of type $a$, i.e. the index $a$ runs over all possible interactions in the system such as electrostatic interactions, excluded volume interactions, etc. The system contains $M_{\rm C}$ molecular species in the canonical ensemble (fixed number of molecules $n_i$) and $M_{\rm G}$ molecular species in the grand canonical ensemble (fixed activities $z_I$). The $Q_i$ and $Q_I$ are complex-valued single-molecule partition functions for the canonical and grand canonical species, respectively. The last term contains the inverse operators for the respective interaction potentials $V_a(r)$. [If this formalism is unfamiliar to you, please have a look at the references above.]

The key functionality of `biofts` is to evolve the fields $\psi_a(\mathbf{r})$ in Complex-Langevin time $t$ using the following stochastic differential equation:

$$
\frac{\partial \psi_a(\mathbf{r},t)}{\partial t} = -\frac{\delta H}{\delta \psi_a(\mathbf{r},t)} + \eta_a(\mathbf{r},t)
$$

where $\eta_a(\mathbf{r},t)$ is a real-valued Gaussian noise term. This is achieved by approximating the continuous fields $\psi_a(\mathbf{r},t)$ as a discrete set of field variables living on the sites of a $d$-dimensional rectangular lattice, and then evolving the fields in $t$ using a finite-difference scheme. The resulting field trajectories can then be used to compute thermodynamic averages of observables of interest. 

It is possible in `biofts` to set the noise term $\eta_a(\mathbf{r},t)$ to zero, in which case FTS reduces to self-consistent field theory (SCFT).

## Quick start

First, import the package along with any other packages you need:

```python
import biofts
import numpy as np
```

Setting up a simulation in `biofts` is done through the following steps:

### Step 1: Define the interactions

In FTS, each field corresponds to a a specific interaction in the system, so the first step is to define the interactions in the system. `biofts` currently supports Yukawa-type interactions, $V(r) = l \mathrm{e}^{- \kappa r} / r $, and contact interactions, $V(r) = \gamma^{-1} \delta(r)$. These can be defined as follows:

```python

# Excluded volume contact interactions
v = 0.0068
excluded_volume = biofts.Contact(1./v)

# (Un-screened) Electrostatics
lB = 2.
kappa = 0.0
electrostatics = biofts.Yukawa(lB,kappa)

# Collect all interactions in a tuple
interactions = (excluded_volume,electrostatics)
```

You can define any number of interactions in this way.

### Step 2: Create the simulation box

A simulation box is created by specifying the number of lattice sites in each dimension, the lattice spacing, and the interactions in the system. For example,

```python

# Define the grid
grid_dimensions = [16,16,80] # Number of lattice sites in each dimension. This can be a 1D, 2D, or 3D grid.
side_lengths    = [8.,8.,40.] # Length of the simulation box in each dimension

# Create the simulation box
sb = biofts.SimulationBox(grid_dimensions,side_lengths,interactions)

```

### Step 3: Add the molecular species to the simulation box

Currently, `biofts` currently only supports linear bead-spring polymers where each monomer is associated with a set of generalized charges that governs its interactions with other monomers through the interactions defined in Step 1. For applications to IDPs, each monomer typically represents a residue in the protein sequence. 

The single-molecule partition function for such a polymer species with $N$ monomers is given by

$$
Q[\lbrace \psi_a \rbrace ] = \frac{1}{V} \left( \frac{3}{2 \pi b^2} \right)^{\frac{3(N-1)}{2}} \left( \prod_{\alpha=1}^N \int \mathrm{d} \vec{R}_\alpha \right) \mathrm{e}^{ - \frac{3}{2 b^2} \sum \Delta R^2 - \mathrm{i} q \cdot \psi}
$$

where

$$
\vec{R}_{a+1} - R_{\alpha}
$$

and $q_{a,\alpha}$ are the generalized charges for the polymer species. 

The following code snippet shows how to add a single polymer species, corresponding to a linear chain of E (glutamic acid) and K (lysine) residues:

```python

aa_sequence = 'EKKKKKKEEKKKEEEEEKKKEEEKKKEKKEEKEKEEKEKKEKKEEKEEEE' # Amino-acid sequence
mol_id = 'sv10' # Name of the polymer species
N = len(seq) # Number of monomers in the chain

q = np.zeros( (2,N) ) # Generalized charges. The first index refers to interaction type, the second to monomer index.

# Excluded volume interactions
q[0,:] += 1. # The generalized charge for the excluded volume interaction is the monomer size, set to 1 for all monomers.

# Electrostatic interactions
aa_charges = {'E':-1,'K':1} # Electric charges for each amino acid type
q[1,:] = [ charge_seq[aa] for aa in aa_sequence ]

# Chain density
rho_bulk = 2.0 / N # rho_bulk is chain number density, n/V. Bead number density is n*N/V.

# Create the polymer species
a = 1./np.sqrt(6.) # Gaussian smearing length
b = 1. # Kuhn length
polye = biofts.LinearPolymer(q,a,b,rho_bulk,sb,molecule_id=seq_label)

```

You can add any number of polymer species to the simulation box in this way. Note that variants of the $N=1$ `LinearPolymer` object can be used to represent explicit salt ions and solvent particles. 