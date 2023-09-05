import numpy as np
from abc import ABC, abstractmethod

# The abstract base class for an interaction potential.
class InteractionPotential(ABC):
  
    @abstractmethod 
    def V_inverse(self,k2): # Returnes the ( Fourier transformed potential )^(-1) as function of k^2
        pass

# Screened Coulomb potential 
# V(r) = l / r * e^(-kappa*r)
class Yukawa(InteractionPotential):
    def __init__(self, l, kappa=0):
        self.l = l
        self.kappa = kappa
    
    def V_inverse(self,k2):
        return (k2 + self.kappa**2) / (4.*np.pi*self.l)

# Contact potential V(r) = delta(r) / gamma. Exact compressibility if gamma=0
class Contact(InteractionPotential):
    def __init__(self,gamma):
        self.gamma = gamma
    
    def V_inverse(self,k2):
        return self.gamma * np.ones(k2.shape)