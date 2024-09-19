'''
This script demonstrates how to implement the Gaussian-regularized Edwards model (GREM) in BioFTS. The model is defined e.g. in

Michael C. Villet, Glenn H. Fredrickson; 
Efficient field-theoretic simulation of polymer solutions. 
J. Chem. Phys. 14 December 2014; 141 (22): 224115. 
https://doi.org/10.1063/1.4902886

The script shows how to
1. define excluded volume  interactions
2. create a simulation box
3. add a polymer species to the simulation box
4. set-up a Complex Langevin integrator and sampling tasks
5. run the simulation

The current density profile is stored in the output directory and can be visualized using the "visualize_snapshot.py" script.

'''

import biofts 
import numpy as np
import os

b = 1.                                   # Kuhn length, base unit for length scale
output_dir = 'data/example_model_0/'     # Output directory (will be created if it does not exist)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

##### Step 1: Define the interactions ######
# Excluded volume contact interactions
v = 0.0068 * b**3 # Edwards-type excluded volume parameter
excluded_volume = biofts.Contact(1./v)

interactions = (excluded_volume,)

### Step 2: Create the simulation box ####
grid_dimensions = [16,16,64]
a = b/np.sqrt(6.)
side_lengths = a * np.array(grid_dimensions,dtype=float)

sb = biofts.SimulationBox(grid_dimensions, side_lengths, interactions, use_GPU=False)

### Step 3: Add polymer species to simulation box ####
N = 50 # Number of beads in the polymer chain
q = np.zeros( (sb.Nint,N) ) # Generalized charges

# The generalized charge for excluded volume interactions corresponds to relative size
q[0,:] += 1.

# Chain density
rho_bulk = 0.1 / N # rho_bulk is chain number density, n/V. Bead number density is n*N/V.

# Create the polymer species
a = b/np.sqrt(6.) # Gaussian smearing length
biofts.LinearPolymer(q,a,b,rho_bulk,sb,molecule_id='polymer', is_canonical=True)

### Step 4: Set-up Complex Langevin integrator and sampling tasks ####
dt = 1e-3
cl = biofts.ComplexLangevinIntegrator(dt, sb, method='semi-implicit', noise=1)

# Visualization task for monitoring density profiles
visualizer = biofts.Monitor_Density_Profiles_Averaged_to_1d(sb, show_imaginary_part=False)

# Task for storing the latest configuration
save_fields = biofts.Save_Field_Configuration(sb, data_directory=output_dir, load_last_configuration=False)

# Task for storing the latest density configuration. The density profiles can be visualized using the "visualize_snapshot.py" script.
save_densities = biofts.Save_Latest_Density_Profiles(sb, data_directory=output_dir)

sampling_tasks = (visualizer, save_fields, save_densities)

### Step 5: Run the simulation ####
n_steps = 3000+1
sample_interval = 10

cl.run_ComplexLangevin(n_steps, sample_interval, sampling_tasks=sampling_tasks )

