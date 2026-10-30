'''

Docstring

'''

import biofts 
import numpy as np
import os

# Parameters
b       = 1. # Kuhn length, base unit for length scale
gamma   = 0.4 * b**(-3) # Compressibility parameter
rho0    = 10. * b**(-3) # Overall polymer bead + solvent number density
nu_w    = 1. # Solvent relative size

output_dir = 'data/example_model_pure_solvent/'     # Output directory (will be created if it does not exist)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

##### Step 1: Define the interactions ######
compressibility = biofts.Contact(gamma) # Compressibility

interactions = (compressibility, )

### Step 2: Create the simulation box ####
#grid_dimensions = [32,32,32]
grid_dimensions = [16,16,16]
a = b/np.sqrt(6.)
side_lengths = a * np.array(grid_dimensions,dtype=float)

sb = biofts.SimulationBox(grid_dimensions, side_lengths, interactions, use_GPU=False)

### Step 3: Add molecular species to simulation box ####
zw = rho0 / nu_w * np.exp(nu_w*rho0/gamma) # Solvent fugacity. Use MFT solution to have nu_w rho_w = rho0 at zero polymer density
#zw = (1-phi) * rho0 / nu_w # Solvent density
q = np.zeros( (sb.Nint,1) ) # Generalized charges
q[0,0] += nu_w # Generalized charge for excluded volume interactions
biofts.LinearPolymer(q, a, b, zw, sb, molecule_id='Solvent', is_canonical=False)


### Step 4: Set-up Complex Langevin integrator and sampling tasks ####
dt = 1e-4
cl = biofts.ComplexLangevinIntegrator(dt, sb, method='semi-implicit', noise=1.)

# Visualization task for monitoring density profiles
visualizer = biofts.Monitor_Density_Profiles_Averaged_to_1d(sb, show_imaginary_part=False)
# Task for storing the latest configuration
save_fields = biofts.Save_Field_Configuration(sb, data_directory=output_dir, load_last_configuration=False)
# Task for storing the latest density configuration. The density profiles can be visualized using the "visualize_snapshot.py" script.2
save_densities = biofts.Save_Latest_Density_Profiles(sb, data_directory=output_dir)

sampling_tasks = (visualizer, save_fields, save_densities)

### Step 5: Run the simulation ####
n_steps = 1000+1
sample_interval = 20

cl.run_ComplexLangevin(n_steps, sample_interval, sampling_tasks=sampling_tasks )

