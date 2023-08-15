import numpy as np
import sys

class ComplexLangevinIntegrator:

    def __init__(self, dt, simulation_box, noise = 1., method='semi-implicit'):
        self.dt = dt # time-step
        self.simulation_box = simulation_box
        self.noise = noise

        available_methods = {'euler'        : self._Euler_step ,\
                             'semi-implicit': self._SemiImplicit_step}
        if method not in available_methods:
            print("[ERROR] The integration method does not exist. Available methods are",list( available_methods.keys()) )
            sys.exit()
        setup_function = {'euler'        : self._setup_Euler ,\
                          'semi-implicit': self._setup_SemiImplicit}[method]
        setup_function()
        
        # Stepping function to be used for CL evolution
        self.take_step = available_methods[method]

        self.simulation_box.set_fields_to_homogeneous_saddle()
        
    def _Euler_step(self):
        # rho[I,x,y,z,...] is type-I charge density
        rho = np.sum( np.array([molecule.rho for molecule in self.simulation_box.species]) , axis=0)

        # Fourier transformed fields
        f_Psi = np.array([ self.simulation_box.ft( Psi ) for Psi in self.simulation_box.Psi])

        # Deterministic field shifts
        tmp = np.array([ self.simulation_box.ift( self.simulation_box.G0[I] * f_Psi[I]) for I in range(len(f_Psi))] )
        dPsi = -self.dt * ( 1j*rho + tmp )

        if self.noise != 0:
            # Add Langevin noise to field shifts
            std = np.sqrt(2. * self.dt / self.simulation_box.dV) * self.noise
            eta = std * np.random.standard_normal( self.simulation_box.field_shape )
            dPsi += eta

        # Update fields
        self.simulation_box.Psi += dPsi

        # Re-calculate all densites
        for molecule in self.simulation_box.species:
            molecule.calc_densities()

    def _setup_Euler(self):
        pass

    def _SemiImplicit_step(self):
        # rho[I,x,y,z,...] is type-I charge density
        rho = np.sum( np.array([molecule.rho for molecule in self.simulation_box.species]) , axis=0)

        # Fourier transformed fields
        f_Psi = np.array([ self.simulation_box.ft( Psi ) for Psi in self.simulation_box.Psi])

        # Deterministic field shifts
        tmp = np.array([ self.simulation_box.ift( self.simulation_box.G0[I] * f_Psi[I]) for I in range(len(f_Psi))] )
        dPsi = -self.dt * ( 1j*rho + tmp )

        dPsi = np.array([ self.simulation_box.ft(Psi) for Psi in dPsi ])
        
        idx = ''.join( ['i','j','k'][:self.simulation_box.d] )
        mult_string = idx+'IJ,J'+idx+'->I'+idx
        dPsi = np.einsum( mult_string, self.Minv, dPsi)

        dPsi = np.array([ self.simulation_box.ift(Psi) for Psi in dPsi ])

        if self.noise != 0:
            # Add Langevin noise to field shifts
            std = np.sqrt(2. * self.dt / self.simulation_box.dV) * self.noise
            eta = std * np.random.standard_normal( self.simulation_box.field_shape )
            dPsi += eta

        # Update fields
        self.simulation_box.Psi += dPsi

        # Re-calculate all densites
        for molecule in self.simulation_box.species:
            molecule.calc_densities()
        
    

    def _setup_SemiImplicit(self): 
        K = np.einsum( 'IJ,I...->...IJ' , np.eye(self.simulation_box.Nint) , self.simulation_box.G0 , dtype=complex)
        for mol in self.simulation_box.species:
            K += mol.calc_quadratic_coefficients()
        M = np.einsum('...,IJ->...IJ',np.ones(self.simulation_box.grid_dimensions), np.eye( self.simulation_box.Nint ), dtype=complex )
        M += self.dt * K
        self.Minv = np.linalg.inv(M)



if __name__ == "__main__":
    import simulation_box as sim_box
    import interaction_potentials as int_pot
    import linear_polymer as lp

    #### for plotting #####
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    #matplotlib.rcParams['text.latex.unicode'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 0.8 
    plt.rcParams['lines.linewidth'] = 3 
    ######################

    def av_density_to_1d(rho_in):
        rho = np.copy(rho_in)
        while len(rho.shape)>1:
            rho = np.mean(rho,axis=0)
        return rho


    ##### Set-up simulation box ######
    # Compressibility
    gamma = 0.5
    compr = int_pot.Contact(gamma)

    # (Un-screened) Electrostatics
    lB = 1.0
    kappa = 0.1
    electr = int_pot.Yukawa(lB,kappa)

    grid_dimensions = [31,33,100]
    b = 1.
    a = b/np.sqrt(6.)
    side_lengths = a * np.array(grid_dimensions,dtype=float)

    interactions = (compr,electr,)
    sb = sim_box.SimulationBox(grid_dimensions,side_lengths,interactions)

    ### Define a di-block polyampholyte ####
    N = 20
    q = np.zeros( (sb.Nint,N) )
    q[0,:] += 1. # sizes for excluded volume interactions
    q[1,:int(N/2)] += 1 # charge-positive residues
    q[1,int(N/2):] -= 1 # charge-negative residues

    print("Net charges:", [np.mean(qq) for qq in q])

    rho_bulk = 2.0 / N # rho_bulk is chain number density, n/V. Bead number density is n*N/V.

    polye = lp.LinearPolymer(q,a,b,rho_bulk,sb)
    polye.calc_densities()

    ### Define neutral solvents ####
    N = 1
    q = np.zeros( (sb.Nint,N) )
    q[0,:] += 1. # sizes for excluded volume interactions
    rho_bulk = 10.
    solvent = lp.LinearPolymer(q,a,b,rho_bulk,sb)
    solvent.calc_densities()

    # Set-up the CL integrator
    dt = 0.01
    cl = ComplexLangevinIntegrator(dt, sb, method='euler',noise=1)

    for i in range(100+1):
        print(i)
        av_fields = np.array([ np.mean(Psi) for Psi in sb.Psi])
        cl.take_step()

        if i%20==0:
            rhob = av_density_to_1d(polye.rhob)
            plt.plot(rhob.real,'-' ,color='C0')
            plt.plot(rhob.imag,'--',color='C0')

            rhow = av_density_to_1d(solvent.rhob)
            plt.plot(rhow.real,'-' ,color='C1')
            plt.plot(rhow.imag,'--',color='C1')
            plt.show()