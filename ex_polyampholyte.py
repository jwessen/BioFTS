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
# Compressibility
gamma = 1./0.0068
compr = int_pot.Contact(gamma)

# (Un-screened) Electrostatics
lB = 1.0
kappa = 0.1
electr = int_pot.Yukawa(lB)

grid_dimensions = [30,30,100]
b = 1.
a = b/np.sqrt(6.)
side_lengths = a * np.array(grid_dimensions,dtype=float) * 0.5

interactions = (compr,electr,)
sb = sim_box.SimulationBox(grid_dimensions,side_lengths,interactions)

### Define a di-block polyampholyte ####
seq = 'E'*10 + 'K'*10
N = len(seq)
q = np.zeros( (sb.Nint,N) )
q[0,:] += 1. # sizes for excluded volume interactions
for i in range(N):
    if seq[i] == 'E':
        q[1,i] = -1
    elif seq[i] == 'K':
        q[1,i] = +1
print("Net charges:", [np.mean(qq) for qq in q])

rho_bulk = 2.0 / N # rho_bulk is chain number density, n/V. Bead number density is n*N/V.

polye = lp.LinearPolymer(q,a,b,rho_bulk,sb)
polye.calc_densities()

# Set-up the CL integrator
dt = 0.001
cl = cli.ComplexLangevinIntegrator(dt, sb, method='semi-implicit',noise=1)

for i in range(1000+1):
    print(i)
    av_fields = np.array([ np.mean(Psi) for Psi in sb.Psi])
    print("av fields:",av_fields)
    cl.take_step()

    if i%100==0:
        rhob = av_density_to_1d(polye.rhob)
        z = np.linspace(0,side_lengths[-1],grid_dimensions[-1]) * 3.8
        plt.plot(z,rhob.real,'-' ,color='C0')
        plt.plot(z,rhob.imag,'--',color='C0')

        plt.show()
