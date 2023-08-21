import numpy as np
import sys

class LinearPolymer:
    def __init__(self, q, a, b, rho_bulk, simulation_box, is_canonical=True):
        self.q = np.copy(q)              # q[I,alpha] is charge type I of bead alpha
        self.a = a                       # Smearing length
        self.b = b                       # Kuhn Length
        self.is_canonical = is_canonical # True if canonical ensemble, False if grand-canonical ensemble
        self.rho_bulk = rho_bulk         # Molecule number density (n_i/V) for canonical ensemble, activity parameter 

        self.Nint = len(q)     # Number of interactions
        self.N    = len(q[0])  # Polymerization degree

        if self.Nint != simulation_box.Nint:
            print("[ERROR] Number of interactions inferred from charges does not match simulation_box.Nint!")
            sys.exit()

        self.simulation_box  = simulation_box # Simulation box that it lives in
        self.grid_dimensions = simulation_box.grid_dimensions
        self.ft  = simulation_box.ft
        self.ift = simulation_box.ift

        # Add this molecular species to the simulation box's list over contained species.
        self.simulation_box.species += (self,)
        
        self.k2 = simulation_box.k2
        self.V  = simulation_box.V
        self.dV = simulation_box.dV

        self.Gamma   = np.exp(-self.k2*self.a**2/2.) # Gaussian smearing
        self.Phi     = np.exp(-self.k2*self.b**2/6.) # Gaussian chain n.n propagator

        self.Q = 1. + 0j  # Single molecule partition function
        propagator_shape = np.insert(self.grid_dimensions, 0, self.N)
        self.qF = np.zeros( propagator_shape , dtype=complex )
        self.qB = np.zeros( propagator_shape , dtype=complex )

        density_shape = np.insert(self.grid_dimensions, 0, self.Nint)
        self.rho = np.zeros( density_shape , dtype=complex )   # self.rho[I,x,y,z] is type-I charge density at point (x,y,z)

        # Bead-center number density operator
        self.rhob = np.zeros( density_shape , dtype=complex ) + self.rho_bulk/self.N

        self.calc_densities()

        # Residue-specifc densities?

    # Calculates the density operators for current field configuration in simulation_box
    def calc_densities( self ):

        if np.any( np.isnan(self.simulation_box.Psi) ):
            print("#1")
            sys.exit()

        # Calculate propagators from fluctuations about the mean. Density operators for canonical ensemble species are unchanged. 
        Psi_s = [ self.ift( self.Gamma*self.ft( Psi - np.mean(Psi) ) ) for Psi in self.simulation_box.Psi ] 
        #Psi_s = [ self.ift( self.Gamma*self.ft( Psi ) ) for Psi in self.simulation_box.Psi ] 

        if np.any( np.isnan(Psi_s) ):
            print("#2")
            sys.exit()

        #W =  1j*self.q.T.dot( Psi_s )
        W = 1j * np.einsum('Ia,I...->a...',self.q, Psi_s)

        if np.any( np.isnan(W) ):
            print("#3")
            sys.exit()

        self.qF[0]  = np.exp( -W[0]  )
        self.qB[-1] = np.exp( -W[-1] )
    
        for i in range( self.N-1 ):
            # forwards propagator
            self.qF[i+1] = np.exp( -W[i+1] )*self.ift( self.Phi*self.ft(self.qF[i]) )
            # backwards propagator
            j = self.N-i-1
            self.qB[j-1] = np.exp( -W[j-1] )*self.ift( self.Phi*self.ft(self.qB[j]) )

            if np.any( np.isnan(self.qF) ):
                print("#4, i:",i)
                sys.exit()
            if np.any( np.isnan(self.qB) ):
                print("#5, j:",j)
                sys.exit()

        self.Q = np.sum( self.qF[-1] ) * self.dV / self.V

        if np.isnan(self.Q):
            print("#6")
            sys.exit()
        qs = self.qF * self.qB * np.exp(W) # Residue-specific bead center number density! (if multiplied by rho_bulk)

        if np.any( np.isnan(qs) ):
            print("#7")
            sys.exit()

        self.rho  = self.rho_bulk * np.einsum('Ia,a...->I...',self.q,qs)
        self.rhob = np.sum(qs,axis=0) * self.rho_bulk
        if self.is_canonical:
            self.rho  /= self.Q
            self.rhob /= self.Q
        else:
            print("Grand-canonical not implemented!! Think about normalization.")
            sys.exit()

        for I in range(self.Nint):
            self.rho[I] = self.ift( self.Gamma*self.ft(self.rho[I]) )
    
    # Calculates the coefficients of the quadratic term in the expansion of Q.
    def calc_quadratic_coefficients(self):
        res_diff = np.array( [[ np.abs(alpha-beta) for alpha in range(self.N) ] for beta in range(self.N) ])
        connection_tensor = np.exp( - np.einsum('ab,...->ab...', res_diff, self.simulation_box.k2) * self.b**2/6. )
        g = np.einsum( 'Ia,Jb,ab...->...IJ', self.q, self.q, connection_tensor )

        if self.is_canonical:
            for I in range(self.Nint):
                for J in range(self.Nint):
                    g[I,J][ tuple( self.simulation_box.grid_dimensions * 0 ) ] *= 0

        return g * self.rho_bulk

if __name__ == "__main__":
    import simulation_box as sim_box
    import interaction_potentials as int_pot

    ##### Set-up simulation box ######
    # Compressibility
    gamma = 3.0
    compr = int_pot.Contact(gamma)

    # (Un-screened) Electrostatics
    lB = 2.
    electr = int_pot.Yukawa(lB)

    grid_dimensions = [10,12,23]
    b = 1.
    a = b/np.sqrt(6.)
    side_lengths = a * np.array(grid_dimensions,dtype=float)

    interactions = (compr,electr,)
    sb = sim_box.SimulationBox(grid_dimensions,side_lengths,interactions)

    ### Define a di-block polyampholyte ####
    N = 12
    q = np.zeros( (sb.Nint,N) )
    q[0,:] += 1. # sizes for excluded volume interactions
    q[1,:int(N/2)] += 1 # charge-positive residues
    q[1,int(N/2):] -= 1 # charge-negative residues

    # rho_bulk is chain number density, n/V. Bead number density is n*N/V.
    rho_bulk = 0.5 / N

    polye = LinearPolymer(q,a,b,rho_bulk,sb)
    polye.calc_densities()

    





