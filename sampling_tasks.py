# Classes representing visualization- and sampling tasks for BioFTS simulations to be passed to the run_ComplexLangevin method of a ComplexLangevinIntegrator object.

import numpy as np
import matplotlib.pyplot as plt
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
    def __init__(self, simulation_box, species_to_plot=None, show_imaginary_part=False):
        self.simulation_box = simulation_box

        if species_to_plot is None:
            self.species_to_plot = np.array( range(len(simulation_box.species)) , dtype=int)
        else:
            self.species_to_plot = species_to_plot
        
        self.show_imaginary_part = show_imaginary_part

        self.fig, self.ax = plt.subplots( figsize=(7.,4.5) )
        plt.ion()
    
    def av_density_to_1d(self,rho_in):
        rho = np.copy(rho_in)
        while len(rho.shape)>1:
            rho = np.mean(rho,axis=0)
        return rho
    
    def calculate_1d_center_of_mass_index(self,rho):
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
        self.ax.clear()
        for j in range(len(rho)):
            clr = 'C'+str(j)
            self.ax.plot(z,rho[j].real,'-' ,color=clr,label=self.simulation_box.species[j].molecule_id)
            if self.show_imaginary_part:
                self.ax.plot(z,rho[j].imag,'--',color=clr)
        
        self.ax.set_xlabel(r'$z$')
        self.ax.set_ylabel(r'$\rho(z)$')
        self.ax.legend(loc='upper right')

        title = "i=" + str(sample_index) + ", t=" + str(np.round(self.simulation_box.t,decimals=5))
        self.ax.set_title(title)
        self.fig.canvas.draw()
        plt.pause(0.01)
        
    def finalize(self):
        # Turn of interactive mode and keep plot open
        plt.ioff()
        plt.show()
        plt.close()

    

