import sys

# Rectangular d-dimensional simulation box
class SimulationBox:
    def __init__(self, grid_dimensions, side_lengths, interactions, use_GPU = False):

        if use_GPU:
            print("[ERROR] GPU not implemented yet.")
            sys.exit()
        else:
            import numpy as np
            self.np = np

        self.grid_dimensions = np.array(grid_dimensions,dtype=int)  # (Nx, Ny, Nz, ...) Number of grid points in every dimension
        self.side_lengths    = side_lengths                         # (Lx, Ly, Lz, ...) Side lengths of the simulation box
        if len(grid_dimensions) != len(side_lengths):
            print("[ERROR] grid_dimensions and side_lengths do not have the same shape!")
            sys.exit()
        
        self.d  = len(grid_dimensions) # Number of spatial dimensions
        self.dx = np.array(side_lengths,dtype=float) / np.array( grid_dimensions , dtype=float )

        self.V  = np.prod(self.side_lengths)
        self.dV = np.prod(self.dx)

        self.t = 0. # Current Complex Langevin time

        # The below code constructs the wave-vectors for d=1,2,3. Any way to do it for generic dimensions??
        if self.d==1:
            kx = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[0],self.dx[0])
            self.k2 = kx**2
            self.idx_str = 'i'
        elif self.d==2:
            kx = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[0],self.dx[0])
            ky = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[1],self.dx[1])
            self.k2 = np.array( [[ kx[i]**2 + ky[j]**2 for j in range(self.grid_dimensions[1]) ] 
                                                       for i in range(self.grid_dimensions[0]) ])
            self.idx_str = 'ij'
        elif self.d==3:
            kx = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[0],self.dx[0])
            ky = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[1],self.dx[1])
            kz = 2.*np.pi*np.fft.fftfreq(self.grid_dimensions[2],self.dx[2])
            self.k2 = np.array( [[[ kx[i]**2 + ky[j]**2 + kz[k]**2 for k in range(self.grid_dimensions[2]) ] 
                                                                   for j in range(self.grid_dimensions[1]) ]
                                                                   for i in range(self.grid_dimensions[0]) ])

            self.idx_str = 'ijk'
        else:
            print("[ERROR] d =",self.d,"dimensions is not implemented!")
            sys.exit()
        
        self.interactions = interactions
        self.Nint = len(self.interactions)
        self.field_shape = np.insert(self.grid_dimensions, 0, self.Nint)
        # All fields
        self.Psi = np.zeros(self.field_shape,dtype=complex)
        
        # Tuple of species (e.g. LinearPolymer objects)
        self.species = ()

        self.G0 = np.array([ I.V_inverse(self.k2) for I in self.interactions], dtype=float)

        # if use_GPU:
        #     print("[ERROR] GPU not implemented yet.")
        #     sys.exit()
        # else:
        #     self.ft  = np.fft.fftn
        #     self.ift = np.fft.ifftn

    def ft(self,field):
        if self.np.any( field.shape != self.grid_dimensions ):
            print( field.shape, self.grid_dimensions)
            sys.exit()
        return self.np.fft.fftn(field)

    def ift(self,field):
        if self.np.any( field.shape != self.grid_dimensions ):
            print( field.shape, self.grid_dimensions)
            sys.exit()
        return self.np.fft.ifftn(field)

    def set_fields_to_homogeneous_saddle(self):
        self.Psi *= 0
        for molecule in self.species:
            if molecule.is_canonical == False:
                print("Warning! Species",molecule,"is in grand-canonical ensemble. Treating as canonical to compute saddle.")
        
        G0_MFT = self.np.array([ I.V_inverse( self.np.array([0]) ) for I in self.interactions], dtype=float)[:,0]
        
        rho_bulks = np.array([ molecule.rho_bulk for molecule in self.species ])
        for I in range(self.Nint):
            if G0_MFT[I] != 0:
                qs = self.np.array([ np.sum(molecule.q[I]) for molecule in self.species] )
                rho = self.np.sum( rho_bulks * qs )
                self.Psi[I] -= 1j * rho / G0_MFT[I]



if __name__ == "__main__":
    import interaction_potentials as int_pot
    import numpy as np

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





        
        




