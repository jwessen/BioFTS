import numpy as np
import sys

Nx = int(10)
Ny = int(12)
Nz = int(29)

A = np.array([[[ i*np.pi + np.sqrt(j) + k**2 for k in range(Nz) ] for j in range(Ny)] for i in range(Nx)] )
A_flat = A.reshape(Nx*Ny*Nz)

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            I = k + Nz*j + Nz*Ny*i

            if A[i,j,k] != A_flat[I]:
                print("Error!")

# for I in range(Nx*Ny*Nz):
#     k = I % Nz
#     j = I % 
#     i = int( ((I-k-Nz*j)) % (Nz*Ny) )

#     if A[i,j,k] != A_flat[I]:
#         print("Error!")
#         print(I,i,j,k)
#         sys.exit()

# a = 1./np.sqrt(6.)

# box_dim = (Nx,Ny,Nz)
# sides = np.array([box_dim],dtype=float) * a

# dx = sides / box_dim

# ks = ( 2.*np.pi*np.fft.fftfreq(box_dim[i],dx[i]) for i in range(len(box_dim)) )

class Person:
    def __init__(self,name):
        self.name = name


john = Person('John')
mary = Person('Mary')
arthur = Person('Arthur')

friends = (john,mary)
friends += (arthur,)

for f in friends:
    print(f.name)