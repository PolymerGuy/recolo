"""
VFM pressure reconstruction from curvatures 
(quasi-static bending assumtion and homogeneous, isotropic, thin plate) 
with square, piecewise virtual fields.
Using linear indexing (very fast, limited by memory)
@author: Rene Kaufmann
08.08.2019
"""
# import pdb

import numpy as np
import matplotlib.pyplot as plt
###


def pad_and_find_neigbours(ikyy,neighbour, iprw):
    padi3d = np.pad(ikyy.astype(float),(iprw,), mode='constant', constant_values=(np.nan,))  
    return padi3d.flatten()[neighbour]

def neighbour_map(iprw,size_with_pads_y,size_with_pads_x):
    stencil = np.zeros(iprw*iprw, dtype = np.int)
    for i in range(1, iprw+1):
        stencil[ (i-1)*iprw:i*iprw ] = np.arange( (i-iprw/2)*size_with_pads_x-iprw/2, (i-iprw/2)*size_with_pads_x+iprw/2, 1 ,dtype=np.int) 

    aux_neighbour = np.ones((size_with_pads_x*size_with_pads_y,iprw*iprw),dtype=np.int) * np.arange(size_with_pads_x*size_with_pads_y, dtype=np.int)[:,np.newaxis]
    return aux_neighbour + stencil
   

def plate_iso_qs_lin(iprw, ikxx, ikyy, ikxy, iD11, iD12, ikxxfield, ikyyfield, ikxyfield, iwfield):
    ###
    # iprw                                         : size of reconstruction window ( = size of virtual field )
    # ikxx, ikyy, ikxy                             : curvature data
    # iD11, iD12                                   : bending stiffness matrix components
    # ikxxfield, ikyyfield, ikxyfield, iwfield     : virtual fields
    # ipoint_size                                  : physical size of one data point (length over which experimental data is integrated)
    ###

    if np.mod(iprw,2) != 0:
        raise ValueError("Reconstruction window size has to be an even number")   

    
    # linearize indices
    padi3d = np.ones_like(ikxx,dtype=np.bool)
    padi3d = np.pad(padi3d,(iprw,), mode='constant', constant_values=(False,))
    size_with_pads_x,size_with_pads_y = ikxx.shape
    size_with_pads_x +=2*iprw  
    size_with_pads_y +=2*iprw

    neighbour = neighbour_map(iprw,size_with_pads_x,size_with_pads_y)

    neighbour = neighbour[padi3d.flatten(),:]

    kxxl = pad_and_find_neigbours(ikxx,neighbour,iprw)    

    kyyl = pad_and_find_neigbours(ikyy,neighbour,iprw)

    kxyl = pad_and_find_neigbours(ikxy,neighbour,iprw)
    
    kxxf = ikxxfield.flatten()
    kyyf = ikyyfield.flatten()
    kxyf = ikxyfield.flatten()
    
    A11 =  np.sum(kxxl*kxxf.transpose() + kyyl* kyyf.transpose() + 2.*(kxyl*kxyf.transpose()), axis=1)

    A11 = A11[~np.isnan(A11)] 
                    
    A12 = np.sum( (kxxl* kyyf.transpose()) + (kyyl* kxxf.transpose()) - 2.*((kxyl* kxyf.transpose())), axis=1 )
    A12 = A12[~np.isnan(A12)] 
    
    U3 = np.sum(iwfield.flatten())
    

    dim = ikxx.shape
    aux_p = (A11*iD11+A12*iD12)/U3
    op = aux_p.reshape([dim[0]-iprw+1, dim[1]-iprw+1])

    return op