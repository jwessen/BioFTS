# Classes representing visualization- and sampling tasks for BioFTS simulations to be passed to the run_ComplexLangevin method of a ComplexLangevinIntegrator object.

from abc import ABC, abstractmethod

# The abstract base class for a visualization task.
class SamplingTask(ABC):
  
    @abstractmethod 
    def sample(self,sample_index):
        pass

    @abstractmethod
    def finalize(self):
        pass

# Visualizer for d-dimensional density profiles. Shows 1d density profile over the last dimension, averaged over the other dimensions.
class Monitor_Density_Profiles_Averaged_to_1d(SamplingTask):
    def __init__(self, simulation_box, species_to_plot=None, show_imaginary_part=False,pause_at_end=True):
        import matplotlib.pyplot as plt
        self.np = simulation_box.np

        self.plt = plt
        self.simulation_box = simulation_box
        self.pause_at_end = pause_at_end

        if species_to_plot is None:
            #self.species_to_plot = self.np.array( range(len(simulation_box.species)) , dtype=int)
            self.species_to_plot = tuple(range(len(simulation_box.species)))
        else:
            self.species_to_plot = tuple(species_to_plot)
        
        self.show_imaginary_part = show_imaginary_part

        self.fig, self.axes = plt.subplots( figsize=(8.,6.5), nrows=2 )
        plt.ion()

        # Chemical potential of first species
        self.mu = [ [] for _ in range(len(self.species_to_plot)) ]
        self.t  = []

    
    def av_density_to_1d(self,rho_in):
        rho = self.np.copy(rho_in)
        while len(rho.shape)>1:
            rho = self.np.mean(rho,axis=0)
        return rho
    
    def calculate_1d_center_of_mass_index(self,rho):
        np = self.np
        angle = np.linspace(0,2.*np.pi,len(rho))
        xi   = np.cos( angle )
        zeta = np.sin( angle )
        av_xi   = np.sum( rho * xi   ) / np.sum(rho)
        av_zeta = np.sum( rho * zeta ) / np.sum(rho)
        av_angle = np.arctan2(av_zeta, av_xi)
        av_angle  = (av_angle + 2. * np.pi ) % (2. * np.pi)
        com     = np.round( len(rho) * av_angle / ( 2. * np.pi ) ).astype(int)
        return com

    def sample(self,sample_index):
        np = self.np

        for i in range(len(self.species_to_plot)):
            s = self.species_to_plot[i]
            self.mu[s].append( np.log(self.simulation_box.species[s].Q) )
        self.t.append( self.simulation_box.t )

        # Find center-of-mass of first molecule species using real part of density
        rhob = self.simulation_box.species[self.species_to_plot[0]].rhob.real
        rhob = self.av_density_to_1d( rhob )

        com = self.calculate_1d_center_of_mass_index(rhob) # Center-of-mass index
        shift = int(-com +len(rhob)/2.) # Shift to center-of-mass

        # All densities to plot
        rho = np.array([ self.av_density_to_1d(self.simulation_box.species[s].rhob) for s in self.species_to_plot ] )
        for j in range(len(rho)):
            rho[j] = np.roll(rho[j], shift)

        # z-coordinate
        z = np.linspace(0,self.simulation_box.side_lengths[-1],self.simulation_box.grid_dimensions[-1])

        # Visualize current field configuration
        ax = self.axes[0]
        ax.clear()
        for j in range(len(rho)):
            clr = 'C'+str(j)
            ax.plot(z,rho[j].real,'-' ,color=clr,label=self.simulation_box.species[j].molecule_id)
            if self.show_imaginary_part:
                ax.plot(z,rho[j].imag,'--',color=clr)
        
        ax.set_xlabel(r'$z$')
        ax.set_ylabel(r'$\rho(z)$')
        ax.legend(loc='upper right')

        title = "i=" + str(sample_index) + ", t=" + str(np.round(self.simulation_box.t,decimals=5))
        ax.set_title(title)

        ax = self.axes[1]
        ax.clear()
        for i in range(len(self.species_to_plot)):
            s = self.species_to_plot[i]
            clr = 'C'+str(i)
            ax.plot(self.t,np.array(self.mu[i]).real,'-',color=clr,label=self.simulation_box.species[s].molecule_id)
            if self.show_imaginary_part:
                ax.plot(self.t,np.array(self.mu[i]).imag,'--',color=clr)
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\mu$')

        self.fig.canvas.draw()
        self.plt.pause(0.01)
        
    def finalize(self):
        if self.pause_at_end:
            # Turn of interactive mode and keep plot open
            self.plt.ioff()
            self.plt.show()
            self.plt.close()
        else:
            # Close plot
            self.plt.close()

# Store the full number density arrays to a binary file
class Save_Latest_Density_Profiles(SamplingTask):
    def __init__(self, simulation_box, data_directory=''):
        
        self.simulation_box = simulation_box
        #self.np = simulation_box.np
        import numpy as np
        self.np = np

        self.filename = data_directory + 'density_profiles.npz'

    # Save the field configuration (Psi) and the time (t) to file
    def sample(self,sample_index):
        np = self.np
        if self.simulation_box.use_GPU:
            rho = [ species.rhob.get() for species in self.simulation_box.species ]
        else:
            rho = [ species.rhob for species in self.simulation_box.species ]
        
        # Overwrite the latest field configuration to file
        np.savez(self.filename, rho=rho)

    # Do nothing
    def finalize(self):
        pass


# Save density profiles to file
class Save_1d_Density_Profiles(SamplingTask):
    def __init__(self, simulation_box, data_directory='', species_to_plot=None, remove_old_data_files=False):
        self.simulation_box = simulation_box
        self.np = simulation_box.np

        if species_to_plot is None:
            #self.species_to_plot = self.np.array( range(len(simulation_box.species)) , dtype=int)
            self.species_to_plot =  tuple( range(len(simulation_box.species)))
        else:
            self.species_to_plot = tuple(species_to_plot)
        
        self.data_directory = data_directory

        # Names of data files to save density profiles to
        self.data_files = [ self.data_directory + 'density_profile_' + self.simulation_box.species[s].molecule_id + '.txt' for s in self.species_to_plot ]

        # Remove old data files
        if remove_old_data_files:
            for file in self.data_files:
                with open(file, 'w') as f:
                    pass

    def av_density_to_1d(self,rho_in):
        np = self.np
        rho = np.copy(rho_in)
        while len(rho.shape)>1:
            rho = np.mean(rho,axis=0)
        return rho

    # Store all density profiles in a dictionary as separate files named after molecule ids
    def sample(self,sample_index):
        np = self.np
        for s in self.species_to_plot:
            rho = self.av_density_to_1d(self.simulation_box.species[s].rhob).real
            mu = np.round( np.log(self.simulation_box.species[s].Q).real , decimals=5)
            t = np.round(self.simulation_box.t,decimals=5)
            file = self.data_files[s]
            
            # np.log(self.simulation_box.species[0].Q).real
            # np.log(self.simulation_box.species[s].Q).real

            # Append density profile to file as a new line. First column is time, second is sample index, third is chemical potential mu.  The rest are the density profile values.
            with open(file, 'a') as f:
                f.write(str(t) + ' ' + str(int(sample_index)) + ' ' + str(mu) + ' ' + ' '.join(map(str, rho)) + '\n')
            
    def finalize(self):
        pass

# Save latest field configuration to file
class Save_Field_Configuration(SamplingTask):

    def __init__(self, simulation_box, data_directory='', load_last_configuration=True):

        self.simulation_box = simulation_box
        #self.np = simulation_box.np
        import numpy as np
        self.np = np

        self.filename = data_directory + 'field_configuration.npy'

        if load_last_configuration:
            self.load_field_configuration()

    # Save the field configuration (Psi) and the time (t) to file
    def sample(self,sample_index):
        np = self.np
        if self.simulation_box.use_GPU:
            Psi = self.simulation_box.Psi.get()
        else:
            Psi = np.array(self.simulation_box.Psi)
        t = self.simulation_box.t
        
        # Overwrite the latest field configuration to file
        np.save(self.filename, np.array([Psi,t],dtype=object) )

    # Set field configuration from file
    def load_field_configuration(self):

        # Check if file exists
        import os.path
        if os.path.isfile(self.filename):
            np = self.np
            Psi, t = np.load(self.filename, allow_pickle=True)
            self.simulation_box.Psi = self.simulation_box.np.asarray( Psi )
            self.simulation_box.t = t

            # Calculate densities
            for molecule in self.simulation_box.species:
                molecule.calc_densities()
        else:
            print("File",self.filename,"does not exist. Starting from t=0.")
            # Throw exception ValueError
            #raise ValueError('File' + self.filename + '.npy' + ' does not exist!')

    # Do nothing
    def finalize(self):
        pass



# Store simulation trajectory using the h5py package
class Save_Trajectory(SamplingTask):
    def __init__(self, simulation_box, data_directory='', load_last_configuration=True):
        import h5py
        self.h5py = h5py

        # Print h5py version
        print(h5py.version.info)

        import numpy as np
        self.np = np
        
        self.simulation_box = simulation_box
        self.filename = data_directory + 'trajectory.h5'

        if load_last_configuration:
            self.load_field_configuration()
            # self.file = h5py.File(self.filename, 'a') # Append to existing file

            # self.ds_time = self.file['time']
            # self.ds_Psi = self.file['Psi']
        else:
            trajectory_file = h5py.File(self.filename, 'w') # Overwrite existing file

            # Define the datasets:
            #   - time: CL time
            #   - Psi: Field configuration
            trajectory_file.create_dataset('time', (0,), dtype='f',chunks=True, maxshape=(None,))
            field_data_shape = (0,) + self.simulation_box.Psi.shape
            trajectory_file.create_dataset('Psi', field_data_shape, 
                                           dtype='complex', 
                                           compression='gzip', 
                                           compression_opts=9, 
                                           chunks=True, 
                                           maxshape=(None,)+self.simulation_box.Psi.shape)

            # self.ds_time = self.file.create_dataset('time', (0,), dtype='f',chunks=True, maxshape=(None,))
            # field_data_shape = (0,) + self.simulation_box.Psi.shape
            # self.ds_Psi = self.file.create_dataset('Psi', field_data_shape, dtype='complex',chunks=True, maxshape=(None,)+self.simulation_box.Psi.shape)

    def sample(self, sample_index):
        np = self.np
        if self.simulation_box.use_GPU:
            Psi = self.simulation_box.Psi.get()
        else:
            Psi = np.array(self.simulation_box.Psi)
        t = self.simulation_box.t

        with self.h5py.File(self.filename, 'a') as trajectory_file:
            # Get the number of samples already stored in the datasets
            n_samples = trajectory_file['time'].shape[0]

            # Append to the datasets
            trajectory_file['time'].resize((n_samples+1,))
            trajectory_file['time'][n_samples] = t

            trajectory_file['Psi'].resize((n_samples+1,) + Psi.shape)
            trajectory_file['Psi'][n_samples] = Psi
        # # Append to the datasets
        # self.ds_time.resize((n_samples+1,))
        # self.ds_time[n_samples] = t

        # self.ds_Psi.resize((n_samples+1,) + Psi.shape)
        # self.ds_Psi[n_samples] = Psi

    def load_field_configuration(self):
        # Check if file exists
        import os.path
        if os.path.isfile(self.filename):

            trajectory_file = self.h5py.File(self.filename, 'r')
            self.simulation_box.t = trajectory_file['time'][-1]
            self.simulation_box.Psi = self.simulation_box.np.asarray( trajectory_file['Psi'][-1] )
            trajectory_file.close()

            # Calculate densities
            for molecule in self.simulation_box.species:
                molecule.calc_densities()
        else:
            # Throw exception ValueError
            raise ValueError('Cannot initialize from file since ' + self.filename + ' does not exist!')
    
    def finalize(self):
        pass