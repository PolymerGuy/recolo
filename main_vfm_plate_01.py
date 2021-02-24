"""
main file
VFM pressure reconstruction and test application
@author: Rene Kaufmann
09.08.2019
"""
# =============================================================================
# for i in list(globals().keys()):
#     exec('del {}'.format(i)) if i[0] != '_' else None
# =============================================================================
    
import sys
#sys.modules[__name__].__dict__.clear()
import pdb   # pdb.set_trace()
#import matplotlib.image as mpimg
#import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
###
# insert at 1, 0 is the script path (or '' in REPL)


#from plot_2D_01 import plot_2D
# from plate_analyt_sine_01 import plate_analyt_sine
#from plate_analyt_hom_01 import plate_analyt_hom
#from .plate_analyt_hom_01 import plate_analyt_hom
# from hermite_16_ho_01 import hermite_16_ho

#from hermite_16_const_01 import hermite_16_const
#from plate_iso_qs_01 import plate_iso_qs
#from plate_iso_qs_lin_01 import plate_iso_qs_lin

from recon import hermite_16_const, plate_iso_qs_lin, plate_analyt_hom

###
###
#directory  = "C:/Users/rk7g15/Dropbox/Matlab/VFM/plate_model/"
# plate and model parameters
mat_E = 70e9         # Young's modulus [Pa]
mat_nu = 0.23        # Poisson's ratio []
Mp = 101 
Np = 101 
mat_Lx = 0.2 
mat_Ly = 0.2  
mat_t = 1e-3
p_0 = 100
dx = mat_Lx/float(Mp)
dy = mat_Ly/float(Np)
###
mat_D = mat_E*math.pow(mat_t, 3)/(12*(1-math.pow(mat_nu, 1)))   # flexural rigidity [N m]
mat_D11 = math.pow(mat_t, 3)/12*mat_E/(1-mat_nu**2)
mat_D12 = math.pow(mat_t, 3)/12*mat_E*mat_nu/(1-mat_nu**2)
# pressure reconstruction parameters
prw = int(24)
interval = 1
### calculate model plate bending
#[w_p, p_in, k_xx, k_yy, k_xy, dx, dy, slope_x, slope_y] = \
#    plate_analyt_sine(Mp, Np, mat_Lx, mat_Ly, p_0, mat_t, mat_E, mat_nu)
##
[w_p, p_in, kxx, kyy, kxy, dx, dy, slope_x, slope_y] = \
    plate_analyt_hom(Mp, Np, mat_Lx, mat_Ly, p_0, mat_t, mat_E, mat_nu)
### calculate slopes
# [slope_x, slope_y] = np.gradient(w_p, dx, dy)
#    slope_x(10:10,10:10,j)=1;
#    slope_y(10:10,10:10,j)=1;
### smooth slopes
# TBD
# calculate curvatures
# =============================================================================
# [aux_k_xx, aux_k_s12] = np.gradient(slope_x, dx, dy)
# [aux_k_s21, aux_k_yy] = np.gradient(slope_y, dx, dy)
# aux_k_xy = .5*(aux_k_s12 + aux_k_s21)
# kxx = aux_k_xx[1:-1, 1:-1]
# kyy = aux_k_yy[1:-1, 1:-1]
# kxy = aux_k_xy[1:-1, 1:-1]
# =============================================================================
#del aux_k_xx, aux_k_yy, aux_k_xy, aux_k_s12, aux_k_s21
#[sx, sy] = np.shape(kxx)
# define piecewise virtual fields
#[kxxfield, kyyfield, kxyfield, wfield] = hermite_16_ho(prw, mat_Lx, mat_Ly, 18, 18)
[kxxfield, kyyfield, kxyfield, wfield] = hermite_16_const(prw, float(dx*prw))
plt.imshow(wfield)
plt.colorbar()
plt.show()
#plt.imshow(kxxfield)
#plt.colorbar()
#plt.show()
#plt.imshow(kyyfield)
#plt.colorbar()
#plt.show()
#plt.imshow(kxyfield)
#plt.colorbar()
#plt.show()
# reconstruct pressure

print("Shape of things:")
print(kxx.shape)
print(kxxfield.shape)

p =  plate_iso_qs_lin(prw, kxx, kyy, kxy, mat_D11, mat_D12, kxxfield, kyyfield, kxyfield, wfield)
##
pshape = np.shape(p)
aux = np.sum(p*mat_Lx/Mp*pshape[0]*mat_Ly/Np*pshape[1])
# =============================================================================
# m = np.arange(0, np.floor(sx-prw)-1, interval)
# n = np.arange(0, np.floor(sy-prw)-1, interval)
# maxj = np.max(np.size(m));
# maxk = np.max(np.size(n));
# #
# #pdb.set_trace()
# f = np.zeros((maxk, maxk))
# p = f
# for k in range(0, maxk):
#     for j in range(0, maxj):
#         p[k,j] = plate_iso_qs(m[k], n[j], prw, kxx, kyy, kxy, \
#             mat_D11, mat_D12, kxxfield, kyyfield, kxyfield, wfield, dx)
# p = np.delete(p, (0), axis=0)
# p = np.delete(p, (0), axis=1)
# # force
# f = np.multiply(p, (prw**2*dx**2))
# =============================================================================
###
print(p.shape)
plt.imshow(p)
plt.colorbar()
plt.show()
###
#with open('plate_data.dat', 'w') as file: # r for read, a for append
#    file.write('p')
#    file.write('p_in')
#with open('plate_data.dat', 'r') as file: 
#    file.read()
# plot results
###
# plot_2D(p)