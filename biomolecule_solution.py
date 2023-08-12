import numpy as np

class BioSol:
    def __init__(self ,box_dimensions, side_lengths, show_prints=True, **kwargs):
        if show_prints:
            print("Initializing...")

        self.box_dimensions = box_dimensions
        self.side_lengths = side_lengths

        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
            self.eta   = np.zeros(self.box_dimensions,dtype=complex) # Compressibility field
            if show_prints:
                print("  Compressibility with gamma:",self.gamma)


        # Electrostatic interactions
        if 'lB' in kwargs:
            self.lB = kwargs['lB']
            if 'kD' in kwargs:
                self.kappaD = kwargs['kappaD']
            else:
                self.kappaD = 0
            
            self.psi = np.zeros(self.box_dimensions,dtype=complex) # Electrostatic potential field

            if show_prints:
                print('  Electrostatics with lB:',self.lB,'and kappaD:',self.kappaD)


        
