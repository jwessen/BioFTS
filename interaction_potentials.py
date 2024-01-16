from abc import ABC, abstractmethod

# The abstract base class for an interaction potential.
class InteractionPotential(ABC):

    @abstractmethod 
    def V_inverse(self,k2): # Returnes the ( Fourier transformed potential )^(-1) as function of k^2
        pass

    @abstractmethod
    def set_simulation_box(self,simulation_box):
        pass

# Screened Coulomb potential 
# V(r) = l / r * e^(-kappa*r)
class Yukawa(InteractionPotential):
    def __init__(self, l, kappa=0):
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
    def __init__(self,gamma):
        import numpy as np
        self.np = np

        self.gamma = gamma
    
    def V_inverse(self,k2):
        return self.gamma * self.np.ones(k2.shape)
    
    def set_simulation_box(self, simulation_box):
        self.np = simulation_box.np