from abc import ABC, abstractmethod

# The abstract base class for an interaction potential.
class InteractionPotential(ABC):

    @abstractmethod 
    def V_inverse(self,k2): # Returns the ( Fourier transformed potential )^(-1) as function of k^2
        pass

    @abstractmethod
    def set_simulation_box(self,simulation_box):
        pass

class Yukawa(InteractionPotential):
    """
    Class for the Yukawa potential V(r) = l / r * e^(-kappa*r) 

    Attributes
    ----------
    l : float
        Interaction strength parameter (analogue of the Bjerrum length l_B = e^2 / (4*pi*epsilon_0*epsilon_r*k_B*T) in electrostatics).
    kappa : float
        Screening length parameter (analogue of the inverse Debye length kappa_D = sqrt(4*pi*l_B*sum_i(z_i^2*n_i) in electrostatics).

    Methods
    -------
    V_inverse(k2)
        Returns the inverse Fourier transform of the potential as a function of k^2.
    set_simulation_box(simulation_box)
        Sets the numpy module of the simulation box to the numpy module of the simulation box.  
    """
    def __init__(self, l, kappa=0):
        """
        Initializes the interaction potential with given parameters.

        Parameters
        ----------
        l : float
            Interaction strength parameter (analogue of the Bjerrum length l_B = e^2 / (4*pi*epsilon_0*epsilon_r*k_B*T) in electrostatics).
        kappa : float, optional
            Screening length parameter (analogue of the inverse Debye length kappa_D = sqrt(4*pi*l_B*sum_i(z_i^2*n_i) in electrostatics). Default is 0.
        """
        import numpy as np
        self.np = np

        self.l = l
        self.kappa = kappa
    
    def V_inverse(self,k2):
        return (k2 + self.kappa**2) / (4.*self.np.pi*self.l)
    
    def set_simulation_box(self, simulation_box):
        self.np = simulation_box.np

# Contact potential V(r) = delta(r) / gamma. Exact compressibility if gamma=0
class Contact(InteractionPotential):
    """
    Class for the contact potential V(r) = delta(r) / gamma. 
    
    In implicit solvent models, gamma is the inverse of the Edwards-type excluded volume parameter. In explicit solvent models, a finite gamma results in "soft compressibility" while setting gamma=0 results in exact compressibility. The gamma=0 case is not well-defined for FTS but is OK for SCFT

    Attributes
    ----------
    gamma : float
        Interaction strength parameter (inverse of the Edwards-type excluded volume parameter for implicit solvent models).

    Methods
    -------
    V_inverse(k2)
        Returns the inverse Fourier transform of the potential as a function of k^2.
    set_simulation_box(simulation_box)

    """

    def __init__(self,gamma):
        import numpy as np
        self.np = np

        self.gamma = gamma
    
    def V_inverse(self,k2):
        #return self.gamma * self.np.ones(k2.shape)
        return self.gamma * (k2*0 + 1.)
    
    def set_simulation_box(self, simulation_box):
        self.np = simulation_box.np