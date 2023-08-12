import numpy as np
import sys

# Rectangular d-dimensional simulation box
class SimulationBox:
    def __init__(self, grid_dimensions, side_lengths, interactions, use_GPU = False):
        self.grid_dimensions = np.array(grid_dimensions,dtype=int)  # (Nx, Ny, Nz, ...) Number of grid points in every dimension
        self.side_lengths    = side_lengths                         # (Lx, Ly, Lz, ...) Side lengths of the simulation box
        if len(grid_dimensions) != len(side_lengths):
            print("[ERROR] grid_dimensions and side_lengths do not have the same shape!")
            sys.exit()
        
        self.d  = len(grid_dimensions) # Number of spatial dimensions
        self.dx = np.array(side_lengths,dtype=float) / np.array( side_lengths , dtype=float )
        
        self.V  = np.prod(self.side_lengths)
        self.dV = np.prod(self.dx)

        # The below code constructs the wave-vectors for d=1,2,3. Any way to do it for generic dimensions??
        if self.d==1:
            kx = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[0],self.dx[0])
            self.k2 = kx**2
        elif self.d==2:
            kx = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[0],self.dx[0])
            ky = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[1],self.dx[1])
            self.k2 = np.array( [[ kx[i]**2 + ky[j]**2 for j in range(self.grid_dimensions[1]) ] 
                                                       for i in range(self.grid_dimensions[0]) ])
        elif self.d==3:
            kx = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[0],self.dx[0])
            ky = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[1],self.dx[1])
            kz = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[2],self.dx[2])
            self.k2 = np.array( [[[ kx[i]**2 + ky[j]**2 + kz[k]**2 for k in range(self.grid_dimensions[2]) ] 
                                                                   for j in range(self.grid_dimensions[1]) ]
                                                                   for i in range(self.grid_dimensions[0]) ])
        else:
            print("[ERROR] d =",self.d,"is not implemented!")
            sys.exit()
        
        self.Nint = len(interactions)
        field_shape = np.insert(self.grid_dimensions, 0, self.Nint)
        self.Psi = np.zeros(field_shape,dtype=complex)
        
        # Tuple of species (e.g. LinearPolymer objects)
        self.species = ()

        self.G0 = np.array([ I.V_inverse(self.k2) for I in interactions], dtype=float)

        if use_GPU:
            print("[ERROR] GPU not implemented yet.")
            sys.exit
        else:
            self.ft  = np.fft.fftn
            self.ift = np.fft.ifftn

    def add_species(self, molecule):
        self.species += (molecule,)

if __name__ == "__main__":
    import interaction_potentials as int_pot

    # Compressibility
    gamma = 3.0
    compr = int_pot.Contact(gamma)

    # (Un-screened) Electrostatics
    lB = 2.
    electr = int_pot.Yukawa(lB)

    grid_dimensions = [10,13,73]
    a = 1/np.sqrt(6.)
    side_lengths = a * np.array(grid_dimensions,dtype=float)

    interactions = (compr,electr,)
    sb = SimulationBox(grid_dimensions,side_lengths,interactions)





        
        




