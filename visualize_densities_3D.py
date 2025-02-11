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

##### Functions for shifting to center of mass #####
def get_com(rho): # Get center of mass
    Nx = rho.shape
    D = len(Nx) # Number of dimensions
    com = np.zeros(D,dtype=int)
    for i in range(D):
        axis = tuple( [ (i+j)%D for j in range(1,D) ])
        rho_av = np.mean(rho,axis=axis )
        psi = 2.*np.pi / Nx[i] * np.arange(Nx[i])
        xi   = np.cos( psi )
        zeta = np.sin( psi )
        av_xi   = np.sum( rho_av * xi )
        av_zeta = np.sum( rho_av * zeta )
        av_phi  = np.arctan2( av_zeta , av_xi )
        
        com[i] = ( int( Nx[i] * av_phi / ( 2. * np.pi ) ) + Nx[i] ) % Nx[i]

    return com

def shift(rho, com=[]): # Shift to center of mass
    if len(com) == 0:
        com = get_com(rho)
    Nx = rho.shape
    D = len(Nx)
    rho_new = np.zeros(rho.shape, dtype=float)
    for idx in np.ndindex(rho.shape):
        new_idx = tuple((idx[i] + 3 * Nx[i] // 2 - com[i]) % Nx[i] for i in range(D))
        rho_new[new_idx] = rho[idx]
    return rho_new

#### Load data ####
if len(sys.argv) < 2:
    file_path = 'data/example_model_1/'
    file_path += 'density_profiles.npz'
else:
    file_path = sys.argv[1]

file = np.load(file_path)

# Load all density profiles in the file
all_labels = file['labels']
all_rho    = file['rho']
all_rho    = [ rho.real for rho in all_rho ] # Remove imaginary part

com = get_com(all_rho[0]) # Get center of mass of first species
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

