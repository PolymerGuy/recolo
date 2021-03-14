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
from scipy.ndimage import shift
import matplotlib.pyplot as plt


###


def pad_and_find_neigbours(ikyy, neighbour, iprw):
    padi3d = np.pad(ikyy.astype(float), (iprw,), mode='constant', constant_values=(np.nan,))
    return padi3d.flatten()[neighbour]


def neighbour_map(iprw, size_with_pads_y, size_with_pads_x):
    stencil = np.zeros(iprw * iprw, dtype=np.int)
    for i in range(iprw):
        stencil[(i) * iprw:(i + 1) * iprw] = np.arange((i - iprw / 2) * size_with_pads_x - iprw / 2,
                                                       (i - iprw / 2) * size_with_pads_x + iprw / 2, 1, dtype=np.int)

    aux_neighbour = np.ones((size_with_pads_x * size_with_pads_y, iprw * iprw), dtype=np.int) * np.arange(
        size_with_pads_x * size_with_pads_y, dtype=np.int)[:, np.newaxis]
    return aux_neighbour + stencil


def plate_iso_qs_lin(iprw, fields, iD11, iD12, vfields, shift_res=True, return_valid=True):
    ###
    # iprw                                         : size of reconstruction window ( = size of virtual field )
    # ikxx, ikyy, ikxy                             : curvature data
    # iD11, iD12                                   : bending stiffness matrix components
    # ikxxfield, ikyyfield, ikxyfield, iwfield     : virtual fields
    # ipoint_size                                  : physical size of one data point (length over which experimental data is integrated)
    ###

    irho = 8000.
    ip_t = 5e-3

    if np.mod(iprw, 2) != 0:
        raise ValueError("Reconstruction window size has to be an even number")

        # linearize indices
    padi3d = np.ones_like(fields.curv_xx, dtype=np.bool)
    padi3d = np.pad(padi3d, (iprw,), mode='constant', constant_values=(False,))
    size_with_pads_x, size_with_pads_y = fields.curv_xx.shape
    size_with_pads_x += 2 * iprw
    size_with_pads_y += 2 * iprw

    neighbour = neighbour_map(iprw, size_with_pads_x, size_with_pads_y)

    neighbour = neighbour[padi3d.flatten(), :]

    kxxl = pad_and_find_neigbours(fields.curv_xx, neighbour, iprw)

    kyyl = pad_and_find_neigbours(fields.curv_yy, neighbour, iprw)

    kxyl = pad_and_find_neigbours(fields.curv_xy, neighbour, iprw)

    al = pad_and_find_neigbours(fields.acceleration, neighbour, iprw)

    kxxf = vfields.okxxfield.flatten()
    kyyf = vfields.okyyfield.flatten()
    kxyf = vfields.okxyfield.flatten()

    A11 = np.sum(kxxl * kxxf.transpose() + kyyl * kyyf.transpose() + 2. * (kxyl * kxyf.transpose()), axis=1)

    A11 = A11[~np.isnan(A11)]

    A12 = np.sum((kxxl * kyyf.transpose()) + (kyyl * kxxf.transpose()) - 2. * ((kxyl * kxyf.transpose())), axis=1)
    A12 = A12[~np.isnan(A12)]

    a_u3 = irho * ip_t * np.sum(al * vfields.owfield.flatten().transpose(), axis=1)
    a_u3 = a_u3[~np.isnan(a_u3)]

    U3 = np.sum(vfields.owfield.flatten())

    dim = fields.curv_xx.shape
    #aux_p = (A11 * iD11 + A12 * iD12) / U3
    aux_p = (A11 * iD11 + A12 * iD12+ a_u3)/U3



    op = aux_p.reshape([dim[0] - iprw + 1, dim[1] - iprw + 1])
    op = (np.pad(op, (int(iprw / 2), int(iprw / 2)), mode="constant", constant_values=0.))[1:, 1:]

    if return_valid:
        op = op[iprw: -iprw, iprw: -iprw]

    if shift_res:
        op = shift(op, shift=0.5, order=3)

    return op, np.sum(A11 + A12) * 0.2 * 0.2 * 0.001
