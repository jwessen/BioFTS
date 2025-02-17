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

# Sampling task for calculating and storing the structure factor
class Store_Structure_Factor:
    def __init__(self, simulation_box, data_directory, run_id='', param_str = '', start_from_previous=False):
        self.sb = simulation_box
        self.data_directory = data_directory
        self.run_id = run_id
        self.param_str = param_str
        
        self.out_file = data_directory + 'structure_factor_run_' + run_id + '.txt' # File to store the structure factor trajectory
        if (not start_from_previous) or (not os.path.exists(self.out_file)):
            # Header: CL time, structure factor
            with open(self.out_file,'w') as file:
                file.write('# Columns: CL time, structure factor \n' )
        
        self.k = np.sqrt(self.sb.k2.copy())
        self.k_unique = np.sort(np.unique(self.k))[1:] # Exclude the zero mode

        self.k_idx = np.array([ self.k==k for k in self.k_unique ])
        k_file = data_directory + 'k_values_run_' + run_id + '.txt'
        np.savetxt(k_file, self.k_unique)

    def spherical_average(self, S):
        # Spherical average of the structure factor
        S_avg = np.array([ np.mean(S[k_idx]) for k_idx in self.k_idx ])
        return S_avg

    def sample(self, sample_index):
        rhob = self.sb.species[0].rho[0].copy() # Bead density
        rhob = self.sb.ft( rhob )    # Fourier transform of bead density at k
        w    = self.sb.ft( self.sb.Psi[0].conj() ).conj() # Fourier transform of field at -k
        v = 1./self.sb.interactions[0].gamma # Excluded volume parameter
        S = rhob * w / self.sb.V * 1j / v
        S = self.spherical_average(S)
        all_S = np.concatenate( (S.real, S.imag) )
        with open(self.out_file,'a') as file:
            file.write( f'{self.sb.t} ' + ' '.join([f'{s}' for s in all_S]) + ' \n' )

    def finalize(self):
        # Compute the average structure factor. Disregard the first 10% of the data for equilibration.
        data = np.loadtxt(self.out_file)[:,1:]
        S_avg = np.mean(data[int(0.1*len(data)):], axis=0)
        S_R = S_avg[:len(S_avg)//2]
        S_I = S_avg[len(S_avg)//2:]
        out_data = [self.k_unique, S_R, S_I]
        avg_file = self.data_directory + 'avg_structure_factor_run_' + self.run_id + '.txt'
        
        np.savetxt(avg_file, np.transpose(out_data), header=self.param_str)


start_from_previous = False
b = 1.                                   # Kuhn length, base unit for length scale
output_dir = 'data/example_model_0/'     # Output directory (will be created if it does not exist)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

##### Step 1: Define the interactions ######
# Excluded volume contact interactions
v = 0.01 * b**3 # Edwards-type excluded volume parameter
excluded_volume = biofts.Contact(1./v)

interactions = (excluded_volume,)

### Step 2: Create the simulation box ####
grid_dimensions = [32,32,32]
a = b/np.sqrt(6.)
side_lengths = a * np.array(grid_dimensions,dtype=float)

sb = biofts.SimulationBox(grid_dimensions, side_lengths, interactions, use_GPU=False)

### Step 3: Add polymer species to simulation box ####
N = 50 # Number of beads in the polymer chain
q = np.ones( (sb.Nint,N) ) # The generalized charge for excluded volume interactions corresponds to relative size

# Chain density
rho_bulk = 2. / N # rho_bulk is chain number density, n/V. Bead number density is n*N/V.

# Create the polymer species
a = b/np.sqrt(6.) # Gaussian smearing length
biofts.LinearPolymer(q,a,b,rho_bulk,sb,molecule_id='polymer', ensemble='canonical')

### Step 4: Set-up Complex Langevin integrator and sampling tasks ####
dt = 1e-3
cl = biofts.ComplexLangevinIntegrator(dt, sb, method='semi-implicit', noise=1)

# Visualization task for monitoring density profiles
visualizer = biofts.Monitor_Density_Profiles_Averaged_to_1d(sb, show_imaginary_part=False)

# Task for storing the latest configuration
save_fields = biofts.Save_Field_Configuration(sb, data_directory=output_dir, load_last_configuration=start_from_previous)

# Task for storing the latest density configuration. The density profiles can be visualized using the "visualize_snapshot.py" script.
save_densities = biofts.Save_Latest_Density_Profiles(sb, data_directory=output_dir)

# Task for storing the structure factor
param_str = 'v:'+str(v) + ' rhop:' + str(rho_bulk) + ' rhob:' + str(rho_bulk*N) + ' N:' + str(N) + ' a:' + str(a) + ' grid:' + str(grid_dimensions) + ' dt:' + str(dt)
save_structure_factor = Store_Structure_Factor(sb, output_dir, run_id='', param_str=param_str, start_from_previous=start_from_previous)

sampling_tasks = (visualizer, save_fields, save_densities, save_structure_factor)

### Step 5: Run the simulation ####
n_steps = 3000+1
sample_interval = 10

cl.run_ComplexLangevin(n_steps, sample_interval, sampling_tasks=sampling_tasks )

