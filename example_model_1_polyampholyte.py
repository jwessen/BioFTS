'''
This script demonstrates the usage of BioFTS using field-theoretic model of a 
polyampholyte solution in implicit solvent. The model is described in 

McCarty et al. J. Phys. Chem. Lett. 2019, 10, 1644−1652.

The script shows how to
1. define exluded volume- and electrostatic interactions
2. create a simulation box
3. add a polymer species to the simulation box
4. set-up a Complex Langevin integrator and sampling tasks
5. run the simulation

The current density profile is stored in the output directory and can be visualized using the "visualize_snapshot.py" script.

'''

import biofts 
import numpy as np
import os

# Polyampholyte sequences from Das & Pappu, Proc. Natl. Acad. Sci. U.S.A. 2013, 110, 13392−13397.
seqs = {}
seqs['sv1']  = 'EKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEKEK'
seqs['sv2']  = 'EEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEEEKKKEK'
seqs['sv3']  = 'KEKKKEKKEEKKEEKEKEKEKEEKKKEEKEKEKEKKKEEKEKEEKKEEEE'
seqs['sv4']  = 'KEKEKKEEKEKKEEEKKEKEKEKKKEEKKKEEKEEKKEEKKKEEKEEEKE'
seqs['sv5']  = 'KEKEEKEKKKEEEEKEKKKKEEKEKEKEKEEKKEEKKKKEEKEEKEKEKE'
seqs['sv6']  = 'EEEKKEKKEEKEEKKEKKEKEEEKKKEKEEKKEEEKKKEKEEEEKKKKEK'
seqs['sv7']  = 'EEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEK'
seqs['sv8']  = 'KKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKKKKEEEEKE'
seqs['sv9']  = 'EEKKEEEKEKEKEEEEEKKEKKEKKEKKKEEKEKEKKKEKKKKEKEEEKE'
seqs['sv10'] = 'EKKKKKKEEKKKEEEEEKKKEEEKKKEKKEEKEKEEKEKKEKKEEKEEEE'
seqs['sv11'] = 'EKEKKKKKEEEKKEKEEEEKEEEEKKKKKEKEEEKEEKKEEKEKKKEEKK'
seqs['sv12'] = 'EKKEEEEEEKEKKEEEEKEKEKKEKEEKEKKEKKKEKKEEEKEKKKKEKK'
seqs['sv13'] = 'KEKKKEKEKKEKKKEEEKKKEEEKEKKKEEKKEKKEKKEEEEEEEKEEKE'
seqs['sv14'] = 'EKKEKEEKEEEEKKKKKEEKEKKEKKKKEKKKKKEEEEEEKEEKEKEKEE'
seqs['sv15'] = 'KKEKKEKKKEKKEKKEEEKEKEKKEKKKKEKEKKEEEEEEEEKEEKKEEE'
seqs['sv16'] = 'EKEKEEKKKEEKKKKEKKEKEEKKEKEKEKKEEEEEEEEEKEKKEKKKKE'
seqs['sv17'] = 'EKEKKKKKKEKEKKKKEKEKKEKKEKEEEKEEKEKEKKEEKKEEEEEEEE'
seqs['sv18'] = 'KEEKKEEEEEEEKEEKKKKKEKKKEKKEEEKKKEEKKKEEEEEEKKKKEK'
seqs['sv19'] = 'EEEEEKKKKKEEEEEKKKKKEEEEEKKKKKEEEEEKKKKKEEEEEKKKKK'
seqs['sv20'] = 'EEKEEEEEEKEEEKEEKKEEEKEKKEKKEKEEKKEKKKKKKKKKKKKEEE'
seqs['sv21'] = 'EEEEEEEEEKEKKKKKEKEEKKKKKKEKKEKKKKEKKEEEEEEKEEEKKK'
seqs['sv22'] = 'KEEEEKEEKEEKKKKEKEEKEKKKKKKKKKKKKEKKEEEEEEEEKEKEEE'
seqs['sv23'] = 'EEEEEKEEEEEEEEEEEKEEKEKKKKKKEKKKKKKKEKEKKKKEKKEEKK'
seqs['sv24'] = 'EEEEKEEEEEKEEEEEEEEEEEEKKKEEKKKKKEKKKKKKKEKKKKKKKK'
seqs['sv25'] = 'EEEEEEEEEEEKEEEEKEEKEEKEKKKKKKKKKKKKKKKKKKEEKKEEKE'
seqs['sv26'] = 'KEEEEEEEKEEKEEEEEEEEEKEEEEKEEKKKKKKKKKKKKKKKKKKKKE'
seqs['sv27'] = 'KKEKKKEKKEEEEEEEEEEEEEEEEEEEEKEEKKKKKKKKKKKKKKKEKK'
seqs['sv28'] = 'EKKKKKKKKKKKKKKKKKKKKKEEEEEEEEEEEEEEEEEEKKEEEEEKEK'
seqs['sv29'] = 'KEEEEKEEEEEEEEEEEEEEEEEEEEEKKKKKKKKKKKKKKKKKKKKKKK'
seqs['sv30'] = 'EEEEEEEEEEEEEEEEEEEEEEEEEKKKKKKKKKKKKKKKKKKKKKKKKK'

aa_charges = {'E':-1,'K':1}        # Electric charges for each amino acid type
b = 1.                             # Kuhn length, base unit for length scale
output_dir = 'data/example_model_1/'     # Output directory (will be created if it does not exist)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

##### Step 1: Define the interactions ######
# Excluded volume contact interactions
v = 0.0068 * b**3 # Edwards-type excluded volume parameter
excluded_volume = biofts.Contact(1./v)

# (Un-screened) Electrostatics
lB = 2*b      # Bjerrum length, lB = e^2 / (4 pi epsilon k_B T) in model units
kappa = 0.0   # Inverse Debye screening length
electrostatics = biofts.Yukawa(lB,kappa)

interactions = (excluded_volume,electrostatics,)

### Step 2: Create the simulation box ####
grid_dimensions = [10,10,40]
a = b/np.sqrt(6.)
side_lengths = a * np.array(grid_dimensions,dtype=float)

sb = biofts.SimulationBox(grid_dimensions, side_lengths, interactions, use_GPU=False)

### Step 3: Add polymer species to simulation box ####
mol_id = 'sv10'
aa_sequence = seqs[mol_id]
N = len(aa_sequence)
q = np.zeros( (sb.Nint,N) ) # Generalized charges

# The generalized charge for exluded volume interactions corresponds to relative size
q[0,:] += 1.

# Electric charges
q[1,:] = [ aa_charges[aa] for aa in aa_sequence ]

# Chain density
rho_bulk = 2.0 / N # rho_bulk is chain number density, n/V. Bead number density is n*N/V.

# Create the polymer species
a = b/np.sqrt(6.) # Gaussian smearing length
biofts.LinearPolymer(q,a,b,rho_bulk,sb,molecule_id=mol_id)

### Step 4: Set-up Complex Langevin integrator and sampling tasks ####
dt = 1e-3
cl = biofts.ComplexLangevinIntegrator(dt, sb, method='semi-implicit', noise=1)

# Visualization task for monitoring density profiles
visualizer = biofts.Monitor_Density_Profiles_Averaged_to_1d(sb, show_imaginary_part=False)

# Task for storing the latest configuration
save_fields = biofts.Save_Field_Configuration(sb, data_directory=output_dir, load_last_configuration=True)

# Task for storing the latest density configuration. The density profiles can be visualized using the "visualize_snapshot.py" script.
save_densities = biofts.Save_Latest_Density_Profiles(sb, data_directory=output_dir)

sampling_tasks = (visualizer, save_fields, save_densities)

### Step 5: Run the simulation ####
n_steps = 3000+1
sample_interval = 50

cl.run_ComplexLangevin(n_steps, sample_interval, sampling_tasks=sampling_tasks )

