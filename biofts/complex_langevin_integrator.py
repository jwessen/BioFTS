import sys

class ComplexLangevinIntegrator:

    def __init__(self, dt, simulation_box, noise = 1., method='semi-implicit'):
        
        self.np = simulation_box.np

        self.dt = dt # time-step
        self.simulation_box = simulation_box
        self.noise = noise

        

        available_methods = {'euler'        : self._Euler_step ,\
                             'semi-implicit': self._SemiImplicit_step}
        if method not in available_methods:
            msg = "[ERROR] The integration method does not exist. Available methods are",list( available_methods.keys()) 
            self.simulation_box.log(msg, level='error')
        self.setup_function = {'euler'        : self._setup_Euler ,\
                               'semi-implicit': self._setup_SemiImplicit}[method]
        
        # Stepping function to be used for CL evolution
        self.shift = available_methods[method]

    def _CL_noise(self):
        if self.noise == 0:
            return 0
        std = self.np.sqrt(2. * self.dt / self.simulation_box.dV) * self.noise
        eta = std * self.np.random.standard_normal( self.simulation_box.field_shape )
        return eta
    
    # Return True if the step was successful, False if there were NaNs
    def take_step(self):
        #Calculate the field shifts
        dPsi = self.shift()

        # Update fields
        self.simulation_box.Psi += dPsi

        # Re-calculate all densities
        for molecule in self.simulation_box.species:
            molecule.calc_densities()

        # Return False if there are NaNs in the fields or densities
        if self.np.isnan( self.simulation_box.Psi ).any() or self.np.array([self.np.isnan( species.rho ) for species in self.simulation_box.species]).any():
            return False
        else:
            return True
        
    def _Euler_step(self):
        # rho[I,x,y,z,...] is type-I charge density
        rho = self.np.sum( self.np.array([molecule.rho for molecule in self.simulation_box.species]) , axis=0)

        # rho_charge_bulk[I] is the bulk charge density of type-I charges
        rho_charge_bulk = self.np.sum( [ molecule.rho_charges_bulk for molecule in self.simulation_box.species ], axis=0)
        for I in range(len(rho)):
            rho[I] -= rho_charge_bulk[I]

        # Fourier transformed fields
        f_Psi = self.np.array([ self.simulation_box.ft( Psi ) for Psi in self.simulation_box.Psi])

        # Deterministic field shifts
        tmp = self.np.array([ self.simulation_box.ift( self.simulation_box.G0[I] * f_Psi[I]) for I in range(len(f_Psi))] )
        dPsi = -self.dt * ( 1.j*rho + tmp ) + self._CL_noise()
        
        return dPsi
        
    def _setup_Euler(self):
        pass

    def _SemiImplicit_step(self):
        # Start with Euler step
        dPsi = self._Euler_step()

        # Do the semi-implicit thing
        dPsi = self.np.array([ self.simulation_box.ft(Psi) for Psi in dPsi ])
        
        idx = ''.join( ['i','j','k'][:self.simulation_box.d] )
        mult_string = idx+'IJ,J'+idx+'->I'+idx

        dPsi = self.np.einsum( mult_string, self.Minv, dPsi)
        dPsi = self.np.array([ self.simulation_box.ift(Psi) for Psi in dPsi ])

        return dPsi

    def _setup_SemiImplicit(self): 
        K = self.np.einsum( 'IJ,I...->...IJ' , self.np.eye(self.simulation_box.Nint) , self.simulation_box.G0 , dtype=complex)
        for mol in self.simulation_box.species:
            K += mol.calc_quadratic_coefficients()
        M = self.np.einsum('...,IJ->...IJ',self.np.ones(self.simulation_box.grid_dimensions), self.np.eye( self.simulation_box.Nint ), dtype=complex )
        M += self.dt * K
        self.Minv = self.np.linalg.inv(M)

    def run_ComplexLangevin(self, n_steps, sample_interval=1, sampling_tasks = ()):

        self.simulation_box.calculate_MFT_solution_before_shift()
        self.setup_function()

        for i in range(n_steps):
            if i%sample_interval==0:
                self.simulation_box.log("i:",i,"t:",self.np.round(self.simulation_box.t,decimals=5))

                for task in sampling_tasks:
                    task.sample(i)

            success = self.take_step()
            if not success:
                self.simulation_box.log("[ERROR] NaNs detected in fields or densities. Stopping simulation at step",i)
                break
            self.simulation_box.t += self.dt
        
        for task in sampling_tasks:
            task.finalize()

if __name__ == "__main__":
    import simulation_box as sim_box
    import interaction_potentials as int_pot
    import linear_polymer as lp
    import numpy as np

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