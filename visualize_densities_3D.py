'''
Visualize 3D density profiles using Mayavi. The density profiles are assumed to be stored in a .npz file with the following structure:

    file = np.load(file_path)
    labels = file['labels']
    rho = file['rho']

where 'labels' is a list of strings with the names of the density profiles and 'rho' is a list of 3D numpy arrays with the density profiles. This structure is used by the BioFTS "Save_Latest_Density_Profiles" sampling task.

You can give the path to the density profiles file as an argument or modify the default path in the script.

The script should run after installing Mayavi and related dependencies as

    pip install mayavi
    pip install PyQt5
    pip install configobj 

Please consult the Mayavi documentation if any issues arise: https://docs.enthought.com/mayavi/mayavi/
'''

print("Loading Mayavi...")
from mayavi import mlab
print("   Done!")
import numpy as np
import sys
import matplotlib.colors as mcolors

##### Functions for shifting to center of mass ####
def get_com(rho):
    Nx, Ny, Nz = rho.shape
    x, y, z = np.mgrid[0:Nx, 0:Ny, 0:Nz]

    psi = 2.*np.pi / Nx * np.array([ x , y , z])
    xi   = np.cos( psi )
    zeta = np.sin( psi )

    rho_R = rho.real.clip(min=0)
    M_tot = np.sum(rho_R)
    
    av_xi   = np.array( [ np.sum( rho_R * xi[i]   ) for i in range(0,3) ] ) / M_tot
    av_zeta = np.array( [ np.sum( rho_R * zeta[i] ) for i in range(0,3) ] ) / M_tot
    av_phi  = np.array( [ np.arctan2( av_zeta[i] , av_xi[i] ) for i in range(0,3) ] )
    av_phi  = (av_phi + 2. * np.pi ) % (2. * np.pi)
    #com     = np.round( Nx * av_phi / ( 2. * np.pi ) ).astype(int)

    com = [ int(av_phi[i] * [Nx,Ny,Nz][i] / (2.*np.pi) ) for i in range(0,3) ]

    return com

def shift( rho, com=[] ): # Shift to center of mass
    if len(com) == 0:
        com = get_com(rho)
    Nx, Ny, Nz = rho.shape
    rho_new = np.zeros(rho.shape,dtype=float)
    for i in range(Nx):
        ii = int( (i+Nx//2+com[0])%Nx )
        for j in range(Ny):
            jj = int( (j+Ny//2+com[1])%Ny )
            for k in range(Nz):
                kk = int( (k+Nz//2+com[2])%Nz )
                rho_new[ii,jj,kk] = rho[i,j,k]             
    return rho_new

#### Load data ####
if len(sys.argv) < 2:
    file_path = 'data/example_model_2_GC/density_profiles.npz'
else:
    file_path = sys.argv[1]

file = np.load(file_path)

# Load all density profiles in the file
all_labels = file['labels']
all_rho    = file['rho']
all_rho    = [ rho.real for rho in all_rho ] # Remove imaginary part

com = get_com(all_rho[0]) # Get center of mass of first species
print("Center of mass:",com)
all_rho = [ shift(rho,com) for rho in all_rho ] # Shift to center of mass

for i in range(len(all_labels)):
    rho = all_rho[i]
    label = all_labels[i]
    color = tuple( np.array( mcolors.to_rgb( 'C'+str(i) ) )**(0.5) )

    vmax = np.mean(rho) * 2
    vmin = 0

    fig = mlab.figure(bgcolor=(0,0,0))
    src = mlab.pipeline.scalar_field(rho)
    mlab.pipeline.volume(src, color=color,vmin=vmin,vmax=vmax)
    mlab.outline(color=(1,1,1), line_width=2)

    mlab.title(label,size=1.,color=color)
    mlab.show()

