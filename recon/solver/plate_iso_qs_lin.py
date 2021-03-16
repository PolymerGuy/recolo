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
from scipy.signal import convolve2d
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


def plate_iso_qs_lin(fields, iD11, iD12, vfields, rho, thickness):

    A11 = convolve2d(fields.curv_xx, vfields.okxxfield, mode="valid") + convolve2d(fields.curv_yy, vfields.okyyfield,
                                                                                   mode="valid") + 2. * convolve2d(
        fields.curv_xy, vfields.okxyfield, mode="valid")
    A11 = np.real(A11)

    A12 = convolve2d(fields.curv_xx, vfields.okyyfield, mode="valid") + convolve2d(fields.curv_yy, vfields.okxxfield,
                                                                                   mode="valid") - 2. * convolve2d(
        fields.curv_xy, vfields.okxyfield, mode="valid")
    A12 = np.real(A12)

    a_u3 = rho * thickness * convolve2d(fields.acceleration, vfields.owfield, mode="valid")
    a_u3 = np.real(a_u3)

    U3 = np.sum(vfields.owfield)

    op = (A11 * iD11 + A12 * iD12 + a_u3) / U3

    return op, 0
