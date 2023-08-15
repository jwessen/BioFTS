import simulation_box as sim_box
import interaction_potentials as int_pot
import linear_polymer as lp
import complex_langevin_integrator as cli

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
rho0 = 10. # bulk density
b = 1.
a = b/np.sqrt(6.)
chi0 = 0.5 # Homopolymer Flory-Huggins chi-parameter. Heteropolymer has effective chi-parameter of chi_eff = chi0 * (fraction of H monomers)
phi  = 0.5 # Volume fraction of polymer beads

# Compressibility
gamma = 0.2
compr = int_pot.Contact(gamma)

# Hydrophobicity
kappa_h = 1. # Inverse range of the HP interaction
l_h     = kappa_h**2 * chi0 / (2*np.pi*rho0)

hydrophobicity = int_pot.Yukawa(l_h,kappa_h)

grid_dimensions = [10,20,100]
side_lengths = a * np.array(grid_dimensions,dtype=float)

interactions = (compr,hydrophobicity,)
sb = sim_box.SimulationBox(grid_dimensions,side_lengths,interactions)

### Define a di-block polyampholyte ####
seq = 'HP'*15
N = len(seq)
q = np.zeros( (sb.Nint,N) ,dtype=complex )
q[0,:] += 1. # sizes for excluded volume interactions
for i in range(N):
    if seq[i] == 'H':
        q[1,i] = -1j
    elif seq[i] == 'P':
        q[1,i] = 0

rho_bulk = phi * rho0 / N # rho_bulk is chain number density, n/V. Bead number density is n*N/V.
hp_chain = lp.LinearPolymer(q,a,b,rho_bulk,sb)
hp_chain.calc_densities()

### Define neutral solvents ####
N = 1
q = np.zeros( (sb.Nint,N) )
q[0,:] += 1. # sizes for excluded volume interactions
rho_bulk = (1.-phi) * rho0
solvent = lp.LinearPolymer(q,a,b,rho_bulk,sb)
solvent.calc_densities()

# Set-up the CL integrator
dt = 0.0001
cl = cli.ComplexLangevinIntegrator(dt, sb, method='semi-implicit',noise=1)

for i in range(1000+1):
    print(i)
    av_fields = np.array([ np.mean(Psi) for Psi in sb.Psi])
    print("mean fields:",av_fields)
    cl.take_step()

    if i%50==0:
        rhob = av_density_to_1d(hp_chain.rhob)
        plt.plot(rhob.real,'-' ,color='C0')
        plt.plot(rhob.imag,'--',color='C0')

        rhow = av_density_to_1d(solvent.rhob)
        plt.plot(rhow.real,'-' ,color='C1')
        plt.plot(rhow.imag,'--',color='C1')
        plt.show()