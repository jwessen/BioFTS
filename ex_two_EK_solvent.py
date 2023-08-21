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

def fasta_to_charge(fasta):
    N = len(fasta)
    sig = np.zeros(N)
    for i in range(N):
        if fasta[i] == 'E':
            sig[i] = -1
        elif fasta[i] == 'K':
            sig[i] = +1

    return sig

def av_density_to_1d(rho_in):
    rho = np.copy(rho_in)
    while len(rho.shape)>1:
        rho = np.mean(rho,axis=0)
    return rho


seq_names = ['sv30'] #, 'sv10']
rhop = np.array( [ 2.0] ) / 50.
rho0 = 1500.
nu_s = 0.1

##### Set-up simulation box ######
# Compressibility
gamma = 0.01
compr = int_pot.Contact(gamma)

# (Un-screened) Electrostatics
lB = 0.38
kappa = 0.1 #38
electr = int_pot.Yukawa(lB)#,kappa)

grid_dimensions = [20,20,128]
b = 1.
a = b/np.sqrt(6.)
side_lengths = a * np.array(grid_dimensions,dtype=float)

interactions = (compr,electr,)
sb = sim_box.SimulationBox(grid_dimensions,side_lengths,interactions)

### Define EK species ####
for i in range(len(seq_names)):
    sn = seq_names[i]
    seq = sek.seqs[sn]
    N = len(seq)
    q = np.zeros( (sb.Nint,N) )
    q[0,:] += 1. # sizes for excluded volume interactions
    q[1,:] = fasta_to_charge(seq)

    print("Average charges:", [np.mean(qq) for qq in q])

    rho_bulk = rhop[i] #phi[i] / N * rho0 # rho_bulk is chain number density, n/V. Bead number density is n*N/V.
    print("Number of chains:",rho_bulk * sb.V)

    lp.LinearPolymer(q,a,b,rho_bulk,sb)

### Define solvent species ####
N = 1
q = np.zeros( (sb.Nint,N) )
q[0,:] += nu_s # sizes for excluded volume interactions
rho_bulk = (rho0 - 50.*np.sum(rhop))  / nu_s
lp.LinearPolymer(q,a,b,rho_bulk,sb)
print("Number of solvent particles:",rho_bulk * sb.V)
print("Solvent volume fraction:", rho_bulk*nu_s/rho0)
print("Effective excluded volume parameter v:", 1./( gamma + nu_s**2 * rho_bulk ) )

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
        #rhob = np.sum( [ sb.species[j].rhob for j in range(len(sb.species)-1) ], axis=0)
        rhob = np.copy( sb.species[0].rhob )
        rhob = av_density_to_1d( rhob )
        
        rho = np.array([ av_density_to_1d(s.rhob) for s in sb.species ] )

        cm = ( np.mean( np.array(range(len(rhob))) * rhob ) / np.mean(rhob) ).real
        shift = int(-cm +len(rhob)/2.)
        for j in range(len(rho)):
            rho[j] = np.roll(rho[j], shift)

        plt.clf()
        # if i>800:
        #     counts += 1
        #     av_rhob += rhob
        #     print("rhob droplet:", av_rhob[ int( len(rhob)/2 ) ] / counts )
        #     plt.plot(z,av_rhob.real/counts,'-' ,color='C1')
        #     plt.plot(z,av_rhob.imag/counts,'--',color='C1')

        z = np.linspace(0,side_lengths[-1],grid_dimensions[-1]) * 3.8
        for j in range(len(rho)-1):
            clr = 'C'+str(j)
            plt.plot(z,rho[j].real,'-' ,color=clr)
            plt.plot(z,rho[j].imag,'--',color=clr)
        plt.title("i=" + str(i) + ", t=" + str(i*dt) )

        plt.draw()
        plt.pause(0.1)

