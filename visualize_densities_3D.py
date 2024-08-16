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

from mayavi import mlab
import numpy as np
import sys
import matplotlib.colors as mcolors

#### Load data ####
if len(sys.argv) < 2:
    file_path = 'data/example_model_1/density_profiles.npz'
else:
    file_path = sys.argv[1]

file = np.load(file_path)

# Load all density profiles in the file
all_labels = file['labels']
all_rho    = file['rho']
all_rho    = [ rho.real for rho in all_rho ] # Remove imaginary part

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

