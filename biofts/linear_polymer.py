import sys

class LinearPolymer:
    def __init__(self, q, a, b, rho_bulk, simulation_box, molecule_id = '', ensemble='canonical'):

        self.np = simulation_box.np
        np = self.np

        if rho_bulk==0:
            simulation_box.log("Zero density detected, not adding molecule",molecule_id,"to simulation box.")
        elif rho_bulk<0:
            msg = "[ERROR] Negative bulk density detected for molecule",molecule_id,"! Exiting."
            simulation_box.log(msg, level='error')
        elif ensemble not in ['canonical','grand-canonical']:
            msg = "[ERROR] Ensemble",ensemble,"not recognized for molecule",molecule_id,"! The ensemble must be either 'canonical' or 'grand-canonical'."
            simulation_box.log(msg, level='error')
        else:
            simulation_box.log("Adding molecule",molecule_id,"to simulation box.")

            self.q = np.copy(q)              # q[I,alpha] is charge type I of bead alpha
            self.a = a                       # Smearing length
            self.b = b                       # Kuhn Length
            self.ensemble = ensemble         # Either 'canonical' or 'grand-canonical'
            #self.rho_bulk = rho_bulk        # Molecule number density (n_i/V) for canonical ensemble, activity parameter for grand-canonical ensemble
            self.molecule_id = molecule_id   # Identifier for this molecule
        
            self.Nint = len(q)     # Number of interactions
            self.N    = len(q[0])  # Polymerization degree

            if self.Nint != simulation_box.Nint:
                simulation_box.log("[ERROR] Number of interactions inferred from charges does not match simulation_box.Nint!", level='error')

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
            propagator_shape = (self.N,) + self.grid_dimensions
            self.qF = np.zeros( propagator_shape , dtype=complex )
            self.qB = np.zeros( propagator_shape , dtype=complex )

            density_shape = (self.Nint,) + self.grid_dimensions
            self.rho = np.zeros( density_shape , dtype=complex )   # self.rho[I,x,y,z] is type-I charge density at point (x,y,z)

            # Bead-center number density operator
            self.rhob = np.zeros( density_shape , dtype=complex ) + rho_bulk * self.N

            # Bulk charge density for each charge type I 
            self.q_tot = np.sum(self.q, axis=1)
            #self.rho_charges_bulk = self.q_tot * rho_bulk

            self.set_rho_bulk(rho_bulk)
    
    def set_rho_bulk(self, rho_bulk):
        self.rho_bulk = rho_bulk
        self.rho_charges_bulk = self.q_tot * self.rho_bulk 
        self.calc_densities()

    # Calculates the density operators for current field configuration in simulation_box
    def calc_densities( self ):
        np = self.np

        if np.any( np.isnan(self.simulation_box.Psi) ):
            self.simulation_box.log("LinearPolymer: NaNs detected in self.simulation_box.Psi.", level='error')

        # Calculate propagators from fluctuations about the mean. 
        #Psi_s = np.asarray( [ self.ift( self.Gamma*self.ft( Psi - np.mean(Psi) ) ) for Psi in self.simulation_box.Psi ] )
        Psi_s = np.array( [ self.ift( self.Gamma*self.ft( Psi - np.mean(Psi) ) ) for Psi in self.simulation_box.Psi ] )

        if np.any( np.isnan(Psi_s) ):
            self.simulation_box.log("LinearPolymer: NaNs detected in Psi_s.", level='error')

        W = 1j * np.einsum('Ia,I...->a...',self.q, Psi_s)

        if np.any( np.isnan(W) ):
            self.simulation_box.log("LinearPolymer: NaNs detected in W.", level='error')

        self.qF[0]  = np.exp( -W[0]  )
        self.qB[-1] = np.exp( -W[-1] )
    
        for i in range( self.N-1 ):
            # forwards propagator
            self.qF[i+1] = np.exp( -W[i+1] )*self.ift( self.Phi*self.ft(self.qF[i]) )
            # backwards propagator
            j = self.N-i-1
            self.qB[j-1] = np.exp( -W[j-1] )*self.ift( self.Phi*self.ft(self.qB[j]) )

            if np.any( np.isnan(self.qF) ):
                self.simulation_box.log("LinearPolymer: NaNs detected in self.qF, bead",i, level='error')
            if np.any( np.isnan(self.qB) ):
                self.simulation_box.log("LinearPolymer: NaNs detected in self.qB, bead",j, level='error')

        self.Q = np.sum( self.qF[-1] ) * self.dV / self.V
        if np.isnan(self.Q):
            self.simulation_box.log("LinearPolymer: self.Q is NaN.", level='error')
        elif self.Q == 0:
            self.simulation_box.log("LinearPolymer: self.Q is zero.", level='error')
        
        qs = self.qF * self.qB * np.exp(W) # Residue-specific bead center number density! (if multiplied by rho_bulk)
        if np.any( np.isnan(qs) ):
            self.simulation_box.log("LinearPolymer: NaNs detected in qs.", level='error')

        self.rho  = self.rho_bulk * np.einsum('Ia,a...->I...',self.q,qs)
        self.rhob = np.sum(qs,axis=0) * self.rho_bulk
        if self.ensemble == 'canonical':
            self.rho  /= self.Q
            self.rhob /= self.Q

        av_Psi = np.array([ np.mean(Psi) for Psi in self.simulation_box.Psi ])
        
        exp_av_W = np.exp( -1j * np.sum( av_Psi.dot(self.q) ) )
        if np.isnan(exp_av_W):
            self.simulation_box.log("LinearPolymer: exp_av_W is NaN.", level='error')
        elif exp_av_W == 0:
            self.simulation_box.log("LinearPolymer: exp_av_W is zero.", level='error')
        self.Q *= exp_av_W

        if np.isnan(self.Q):
            self.simulation_box.log("LinearPolymer: Final self.Q is NaN.", level='error')
        elif self.Q == 0:
            self.simulation_box.log("LinearPolymer: Final self.Q is zero.", level='error')

        if self.ensemble == 'grand-canonical': # Only do this if in grand-canonical ensemble
            self.rho *= exp_av_W
            self.rhob *= exp_av_W

        for I in range(self.Nint):
            self.rho[I] = self.ift( self.Gamma*self.ft(self.rho[I]) )

    # Calculates the coefficients of the quadratic term in the expansion of Q.
    def calc_quadratic_coefficients(self):
        np = self.np

        res_diff = np.array( [[ np.abs(alpha-beta) for alpha in range(self.N) ] for beta in range(self.N) ])
        connection_tensor = np.exp( - np.einsum('ab,...->ab...', res_diff, self.simulation_box.k2) * self.b**2/6. )
        connection_tensor = np.einsum('ab...,...->ab...', connection_tensor, self.Gamma**2)
        g = np.einsum( 'Ia,Jb,ab...->...IJ', self.q, self.q, connection_tensor )

        if self.ensemble == 'canonical':
            idx = tuple( [0 for _ in range(len(self.grid_dimensions))] )
            g[ idx ] *= 0

        return g * self.rho_bulk
    
    # Calculates the residue-specific densities rho_residue with shape (N,Nx,Ny,Nz,...) for the current field configuration in simulation_box
    def calc_residue_specific_densities(self):
        np = self.np

        Psi_s = np.asarray( [ self.ift( self.Gamma*self.ft( Psi - np.mean(Psi) ) ) for Psi in self.simulation_box.Psi ] )
        W = 1j * np.einsum('Ia,I...->a...',self.q, Psi_s)

        qs = self.qF * self.qB * np.exp(W)

        Q = np.sum( self.qF[-1] ) * self.dV / self.V
        
        rho_residue = self.rho_bulk * qs 
        if self.ensemble == 'canonical':
            rho_residue = rho_residue / Q
        else:
            self.simulation_box.log("LinearPolymer: calc_residue_specific_densities() is not implemented for grand-canonical ensemble.", level='error')

        return rho_residue
    
    def chemical_potential(self):
        '''
        Returns the chemical potential of the polymer species in the simulation box.
        If the species is in the grand-canonical ensemble, the function instead returns
        the instantaneous chain bulk density of the species.
        '''

        np = self.np

        if self.ensemble == 'canonical':
            mu = np.log(self.rho_bulk) - np.log(self.Q) - 1j * self.rho_charges_bulk.dot(self.simulation_box.Psi_MFT)
            #print("mu:",mu, "rho_bulk:",self.rho_bulk, "Q:",self.Q)
            return mu
        else:
            return self.rho_bulk * self.Q # Instantaneous chain bulk density. rho_bulk is the activity parameter.

if __name__ == "__main__":
    import simulation_box as sim_box
    import interaction_potentials as int_pot
    import numpy as np

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

    





