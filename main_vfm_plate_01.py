"""
main file
VFM pressure reconstruction and test application
@author: Rene Kaufmann
09.08.2019
"""   
import matplotlib.pyplot as plt
from recon import plate_iso_qs_lin,plate_analyt_sinusoidal, Hermite16, plate_analyt_const

###
###
# plate and model parameters
mat_E = 70.e9         # Young's modulus [Pa]
mat_nu = 0.23        # Poisson's ratio []
Mp = 141
Np = 141 
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
win_size = 12

### calculate model plate bending
#[w_p, p_in, kxx, kyy, kxy, dx, dy, slope_x, slope_y] = \
    #plate_analyt_sine(Mp, Np, mat_Lx, mat_Ly, p_0, mat_t, mat_E, mat_nu)

#fields = plate_analyt_sinusoidal(Mp, Np, mat_Lx, mat_Ly, p_0, mat_t, mat_E, mat_nu)
fields = plate_analyt_const(Mp, Np, mat_Lx, mat_Ly, p_0, mat_t, mat_E, mat_nu)


#plt.plot(fields.okxx[:,50])
#plt.plot(fields.okyy[:,50])
#plt.plot(fields.okxy[:,50])
#plt.show()


# define piecewise virtual fields
virtual_fields = Hermite16(win_size, float(dx*win_size))

recon_press =  plate_iso_qs_lin(win_size, fields, mat_D11, mat_D12, virtual_fields)
##

print(recon_press.shape)
print(fields.press.shape)
plt.figure()
plt.imshow(recon_press)
plt.colorbar()

plt.figure()
plt.imshow(fields.press)
plt.colorbar()
plt.show()

plt.plot(recon_press[:,66])
plt.plot(fields.press[:,70])
plt.show()
