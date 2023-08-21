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
chi0 = 3.0 # Homopolymer Flory-Huggins chi-parameter. Heteropolymer has effective chi-parameter of chi_eff = chi0 * (fraction of H monomers)
phi  = 0.3 # Volume fraction of polymer beads

# Compressibility
gamma = 0.4
compr = int_pot.Contact(gamma)

# Hydrophobicity
#kappa_h = 0.1 # Inverse range of the HP interaction
#l_h     = kappa_h**2 * chi0 / (2*np.pi*rho0)
#hydrophobicity = int_pot.Yukawa(l_h,kappa_h)

hydrophobicity = int_pot.Contact(2*rho0/chi0)

grid_dimensions = [24,24,128] # [20,20,128]
side_lengths = a * np.array(grid_dimensions,dtype=float) 

interactions = (compr,hydrophobicity,)
sb = sim_box.SimulationBox(grid_dimensions,side_lengths,interactions)

### Define a di-block polyampholyte ####
seq = 'H'*30 
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
dt = 1e-4
cl      = cli.ComplexLangevinIntegrator(dt, sb, method='euler',noise=1)
cl_SCFT = cli.ComplexLangevinIntegrator(dt, sb, method='euler',noise=0)

### Set initial field configuration ####
sb.Psi[0][0,0] += 100*1j*np.array([np.sin(np.pi * x / grid_dimensions[-1] ) * x**2 for x in range(grid_dimensions[-1])] ) / grid_dimensions[-1]**2
sb.Psi[1][0,0] += 100   *np.array([np.sin(np.pi * x / grid_dimensions[-1] ) * x**2 for x in range(grid_dimensions[-1])] ) / grid_dimensions[-1]**2


plt.axis()
plt.ion()
plt.show()

for i in range(10000+1):
    print(i)
    av_fields = np.array([ np.mean(Psi) for Psi in sb.Psi])
    print("mean fields:",av_fields)
    # if i<100:
    #     cl.take_step()
    # else:
    #     cl_SCFT.take_step()
    #cl.take_step()
    cl_SCFT.take_step()

    if i%20==0:
        rhob = av_density_to_1d(hp_chain.rhob)
        cm = ( np.mean( np.array(range(len(rhob))) * rhob ) / np.mean(rhob) ).real
        rhob = np.roll(rhob,int(-cm +len(rhob)/2.) )

        plt.clf()

        z = np.linspace(0,side_lengths[-1],grid_dimensions[-1]) * 3.8
        plt.plot(z,rhob.real,'-' ,color='C0')
        plt.plot(z,rhob.imag,'--',color='C0')

        rhow = av_density_to_1d(solvent.rhob)
        rhow = np.roll(rhow,int(-cm +len(rhow)/2.) )

        plt.plot(z,rhow.real,'-' ,color='C1')
        plt.plot(z,rhow.imag,'--',color='C1')
        
        plt.title("i=" + str(i) + ", t=" + str(i*dt) )

        plt.draw()
        plt.pause(0.1)