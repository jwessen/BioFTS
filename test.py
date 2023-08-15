import numpy as np
import sys

d = 1
idx = ''.join( ['i','j','k','l'][:d] )
mult_string = idx+'IJ,J'+idx+'->I'+idx

print( mult_string )
# spatial_index_string 
# 'ijkIJ,Jijk->Iijk'