import biomolecule_solution as bs
import numpy as np

a = 1./np.sqrt(6.)
b = 3.8

lB = 7. / b

box_dimensions = np.array( [10,10,50] , dtype=int)
side_lengths   = box_dimensions * a

BS = bs.BioSol(box_dimensions, side_lengths,
               lB = lB
               )