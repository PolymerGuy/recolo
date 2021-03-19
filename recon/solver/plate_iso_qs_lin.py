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


def plate_iso_qs_lin(fields, plate, virtual_field):

    A11 = convolve2d(fields.curv_xx, virtual_field.okxxfield, mode="valid") + convolve2d(fields.curv_yy, virtual_field.okyyfield,
                                                                                         mode="valid") + 2. * convolve2d(
        fields.curv_xy, virtual_field.okxyfield, mode="valid")
    A11 = np.real(A11)

    A12 = convolve2d(fields.curv_xx, virtual_field.okyyfield, mode="valid") + convolve2d(fields.curv_yy, virtual_field.okxxfield,
                                                                                         mode="valid") - 2. * convolve2d(
        fields.curv_xy, virtual_field.okxyfield, mode="valid")
    A12 = np.real(A12)

    a_u3 = plate.density * plate.thickness * convolve2d(fields.acceleration, virtual_field.owfield, mode="valid")
    a_u3 = np.real(a_u3)

    U3 = np.sum(virtual_field.owfield)

    op = (A11 * plate.bend_stiff_11 + A12 * plate.bend_stiff_12 + a_u3) / U3

    return op
