import sys
import datetime 

# Rectangular d-dimensional simulation box
class SimulationBox:
    """
    A class to represent a d-dimensional simulation box with periodic boundary conditions.

    Attributes
    ----------
    grid_dimensions : tuple of int
        Number of grid points in every dimension.
    side_lengths : numpy array of float
        Side lengths of the simulation box.
    interactions : tuple of InteractionPotential objects
        Interaction potentials between species. Each interaction potential results in a field in the simulation box, representing the conjugate of the corresponding charge density.
    species : tuple of species objects (E.g. LinearPolymer objects)
        Species in the simulation box. The species objects have attributes like charge densities, etc.
    Psi : numpy array of complex float
        Fields in the simulation box. Each field represents the conjugate of the charge density of a species.
    t : float
        Current Complex Langevin time. Initialized to 0.
    d : int
        Number of spatial dimensions.
    dx : numpy array of float
        Grid spacings in every dimension.
    V : float
        Volume of the simulation box.
    dV : float
        Volume element of the simulation box.
    k2 : numpy array of float
        Squared wave-vectors for Fourier transforms.
    np : numpy or cupy module
        Numpy or cupy module for numerical computations. If use_GPU is True, then np is cupy, else it is numpy.
    use_GPU : bool
        If True, then use cupy for numerical computations, else use numpy.
    output_dir : str
        Output directory where log file is placed. If None, then log messages are printed to the console.

    Methods
    -------
    ft(field)
        Returns the Fourier transform of a field.
    ift(field)
        Returns the inverse Fourier transform of a field.
    calculate_MFT_solution_before_shift()
        Calculates the mean field theory solution before the field shift, such that the fields now describe fluctuations about the saddle point. Therefore, the fields should average to zero. It is important that this function is called by the ComplexLangevinIntegrator before the simulation is started. 
    """

    def __init__(self, grid_dimensions, side_lengths, interactions, use_GPU = False, output_dir = None):

        self.use_GPU = use_GPU
        if use_GPU:
            import cupy as np
            self.np = np 
        else:
            import numpy as np
            self.np = np
            self.np.random.seed()

        self.grid_dimensions = tuple(grid_dimensions) # (Nx, Ny, Nz, ...) Number of grid points in every dimension
        self.side_lengths    = np.array(side_lengths) # (Lx, Ly, Lz, ...) Side lengths of the simulation box
        if len(grid_dimensions) != len(side_lengths):
            msg = "[ERROR] grid_dimensions and side_lengths do not have the same shape!"
            self.log(msg, level='error')
        
        self.d  = len(grid_dimensions) # Number of spatial dimensions
        self.dx = np.array(side_lengths,dtype=float) / np.array( grid_dimensions , dtype=float )

        self.V  = np.prod(self.side_lengths)
        self.dV = np.prod(self.dx)

        self.t = 0. # Current Complex Langevin time
        self.output_dir = output_dir
        if self.output_dir is not None:
            with open(self.output_dir + 'log.txt', 'w') as f:
                msg = "Simulation box initialized at " + str(datetime.datetime.now())
                f.write(msg + '\n')

        # The below code constructs the wave-vectors for d=1,2,3. Is there a clean way to do it for generic number of dimensions?
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
            msg = "[ERROR] d =",self.d,"dimensions is not implemented!"
            self.log(msg, level='error')
        
        self.interactions = interactions
        for interaction in self.interactions:
            interaction.set_simulation_box(self)

        self.Nint = len(self.interactions)
        self.field_shape = (self.Nint,) + self.grid_dimensions

        # All fields
        self.Psi = np.zeros(self.field_shape,dtype=complex)
        
        # Tuple of species (e.g. LinearPolymer objects)
        self.species = ()

        self.G0 = np.array([ I.V_inverse(self.k2) for I in self.interactions], dtype=float)

    def ft(self,field):
        return self.np.fft.fftn(field) * self.dV

    def ift(self,field):
        return self.np.fft.ifftn(field) / self.dV
    
    def calculate_MFT_solution_before_shift(self):
        np = self.np
        self.Psi_MFT = np.zeros(self.Nint,dtype=complex)

        all_species_canonical = all(species.ensemble == 'canonical' for species in self.species)

        if all_species_canonical:
            # Calculate the mean field solution exactly.
            rho_charge_bulk = np.sum([ species.rho_charges_bulk for species in self.species ], axis=0) # rho_charge_bulk[a] is total bulk charge density of type-a charges (summed over all species)

            for I in range(self.Nint):
                Vinv = self.interactions[I].V_inverse(0)
                if Vinv == 0:
                    if rho_charge_bulk[I] != 0:
                        msg = f"rho_charge_bulk[{I}] is non-zero but Vinv[{I}] = 0 so the mean field solution is not well-defined. This means that the system net charge is non-zero for an infinite-range interaction potential. You may need to either (1) add explicit counterions or (2) use a Debye-screened electrostatic interaction potential."
                        self.log(msg, level='error')
                    self.Psi_MFT[I] = 0
                else:
                    self.Psi_MFT[I] = -1j * rho_charge_bulk[I] / Vinv
        else:
            self.log("Grand-canonical ensemble species detected. Solving MFT numerically.")
            # Canonical ensemble species charge density contributions
            rho_charge_bulk_C = np.sum([ species.rho_charges_bulk for species in self.species if species.ensemble=='canonical' ], axis=0)
            all_Vinv = np.array([ I.V_inverse(0) for I in self.interactions], dtype=float)

            # This is only used to figure out xi_a (i.e. if there are imaginary charges)
            rho_charge_bulk = np.sum([ species.rho_charges_bulk for species in self.species], axis=0)
            # xi_a = 1 if imag(rho_charge_bulk)==0, otherwise xi_a = -1j
            xi_a = np.where(np.imag(rho_charge_bulk)==0, 1, -1j)
            q_GC = np.array([ species.q_tot for species in self.species if species.ensemble=='grand-canonical' ])  # q_GC[I,a] is the charge of grand-canonical species I for charge type a
            xi_q_GC = np.einsum('Ia,a->Ia',q_GC,xi_a)

            def root_func(x):
                x = np.array(x) # x_a = xi_a * 1j * bar(psi)_a where bar(psi)_a is MFT solution
                
                #exp_arg_prev = - xi_q_GC.dot(x)
                exp_arg = -np.einsum('Ia,a->I',xi_q_GC,x)

                z_I = np.array([species.rho_bulk for species in self.species if species.ensemble=='grand-canonical']) # Activity parameters of grand-canonical species
                rho_bulk_G = z_I * np.exp(exp_arg) # Number densities of grand-canonical species
                rho_charge_bulk_GC = np.einsum('Ia,I->a',q_GC,rho_bulk_G) # Charge densities of grand-canonical species

                LHS = (rho_charge_bulk_C + rho_charge_bulk_GC)*xi_a
                RHS = all_Vinv * x
                return (LHS - RHS).real

            x0 = np.ones(self.Nint)

            self.log("Importing root from scipy.optimize")
            from scipy.optimize import root
            self.log("  Done. Solving....")
            sol = root(root_func, x0)
            self.log(sol)
            print(sol.success)
            if sol.success == True:
                self.log("MFT solution found. Proceeding.")
            else:
                msg = "MFT solution not found."
                self.log(msg, level='error')

            sol_x = sol.x

            self.Psi_MFT = sol_x / xi_a * (-1j)
            I = 0
            exp_arg = np.einsum('a,Ia->I',sol_x,xi_q_GC).real
            for species in self.species:
                if species.ensemble == 'grand-canonical':
                    new_bulk = species.rho_bulk * np.exp(-exp_arg[I])
                    print("New bulk:",new_bulk)
                    species.set_rho_bulk(new_bulk)
                    I+=1
                
                self.log("MFT bulk number density:",species.molecule_id,species.rho_bulk)

    def log(self, *message, level=''):
        message = ' '.join(map(str, message))
        if self.output_dir is not None:
            with open(self.output_dir + 'log.txt', 'a') as f:
                f.write(message + '\n')
        print(message)
        
        if level == 'error':
            raise ValueError(message)

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
