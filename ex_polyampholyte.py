import simulation_box as sim_box
import interaction_potentials as int_pot
import linear_polymer as lp
import complex_langevin_integrator as cli

import sequences_EK as sek

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
lB = 0.38
kappa = 0.1
electr = int_pot.Yukawa(lB)

grid_dimensions = [20,20,128]
b = 1.
a = b/np.sqrt(6.)
side_lengths = a * np.array(grid_dimensions,dtype=float)

interactions = (compr,electr,)
#interactions = (electr,compr,)
sb = sim_box.SimulationBox(grid_dimensions,side_lengths,interactions)

### Define a polyampholyte sequence ####
seq = sek.seqs['sv30']
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
print("Number of chains:",rho_bulk * sb.V)

polye = lp.LinearPolymer(q,a,b,rho_bulk,sb)
polye.calc_densities()

# Set-up the CL integrator
dt = 0.005
cl = cli.ComplexLangevinIntegrator(dt, sb, method='semi-implicit',noise=1)

plt.axis()
plt.ion()
plt.show()

av_rhob = np.zeros(grid_dimensions[-1],dtype=complex)
counts = 0

for i in range(20000+1):
    print(i)
    av_fields = np.array([ np.mean(Psi) for Psi in sb.Psi])
    print("av fields:",av_fields)
    cl.take_step()

    if i%10==0:
        rhob = av_density_to_1d(polye.rhob)
        cm = ( np.mean( np.array(range(len(rhob))) * rhob ) / np.mean(rhob) ).real
        rhob = np.roll(rhob,int(-cm +len(rhob)/2.) )

        plt.clf()
        if i>800:
            counts += 1
            av_rhob += rhob
            print("rhob droplet:", av_rhob[ int( len(rhob)/2 ) ] / counts )
            plt.plot(z,av_rhob.real/counts,'-' ,color='C1')
            plt.plot(z,av_rhob.imag/counts,'--',color='C1')

        z = np.linspace(0,side_lengths[-1],grid_dimensions[-1]) * 3.8
        plt.plot(z,rhob.real,'-' ,color='C0')
        plt.plot(z,rhob.imag,'--',color='C0')
        plt.title("i=" + str(i) + ", t=" + str(i*dt) )

        plt.draw()
        plt.pause(0.1)

