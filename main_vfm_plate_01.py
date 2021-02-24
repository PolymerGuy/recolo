"""
main file
VFM pressure reconstruction and test application
@author: Rene Kaufmann
09.08.2019
"""   
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
from recon import hermite_16_const, plate_iso_qs_lin, plate_analyt_hom, plate_analyt_sine

###
###
# plate and model parameters
mat_E = 70.e9         # Young's modulus [Pa]
mat_nu = 0.23        # Poisson's ratio []
Mp = 51
Np = 51 
mat_Lx = 0.2 
mat_Ly = 0.2  
mat_t = 1e-3
p_0 = 100.
dx = mat_Lx/float(Mp)
dy = mat_Ly/float(Np)
###
mat_D = mat_E*(mat_t**3.)/(12.*(1.-mat_nu))   # flexural rigidity [N m]
mat_D11 = (mat_t**3.)/12.*mat_E/(1.-mat_nu**2.)
mat_D12 = (mat_t**3.)/12.*mat_E*mat_nu/(1.-mat_nu**2.)
# pressure reconstruction parameters
win_size = 24

### calculate model plate bending
[w_p, p_in, kxx, kyy, kxy, dx, dy, slope_x, slope_y] = \
    plate_analyt_sine(Mp, Np, mat_Lx, mat_Ly, p_0, mat_t, mat_E, mat_nu)
##
#[w_p, p_in, kxx, kyy, kxy, dx, dy, slope_x, slope_y] = \
#    plate_analyt_hom(Mp, Np, mat_Lx, mat_Ly, p_0, mat_t, mat_E, mat_nu)


# define piecewise virtual fields
#[kxxfield, kyyfield, kxyfield, wfield] = hermite_16_ho(prw, mat_Lx, mat_Ly, 18, 18)
[kxxfield, kyyfield, kxyfield, wfield] = hermite_16_const(win_size, float(dx*win_size))


print("Shape of things:")
print(kxx.shape)
print(kxxfield.shape)

p_field =  plate_iso_qs_lin(win_size, kxx, kyy, kxy, mat_D11, mat_D12, kxxfield, kyyfield, kxyfield, wfield)
##

print(p_field.shape)
plt.figure()
plt.imshow(p_field)
plt.colorbar()

plt.figure()
plt.imshow(p_in)
plt.colorbar()
plt.show()

