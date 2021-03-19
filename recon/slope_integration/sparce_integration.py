"""
Sparse matrix integration to solve grad(a) = b for a
Requires integration constant (= reference value at position [1,1] = iconst)
Assumes uniform spacing

01.10.2019

@author: Rene Kaufmann
"""

import numpy as np
from numpy import matlib

from scipy.sparse import csr_matrix


###
def int2D(igrad_x, igrad_y, iconst, idx, idy):
    # TODO: Make code pythonic
    [s_x, s_y] = np.shape(igrad_x)
    diffx = matlib.repmat(idx, s_x, 1)
    diffy = matlib.repmat(idy, s_y, 1)
    # initialize variables
    b = np.zeros(2 * s_x * s_y)
    aux_A = np.zeros([2 * s_x * s_y, 6])
    current_length = 0

    # define forward differences for left hand edge in x-direction
    aux_ix = 0
    aux_iy = np.arange(0, s_y, 1)
    aux_i = aux_ix * (np.transpose(aux_iy) - 1) + aux_iy

    ind_m = matlib.repmat(aux_iy + current_length, 2, 1)
    ind_n = np.array([aux_i, aux_i + s_y])

    diffopx = matlib.repmat(np.divide(np.array([-1, 1]), diffx[1]), s_y, 1)

    aux_A[np.arange(0, s_y) + current_length] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopx),
                                                               axis=1)

    b[np.arange(0, s_y, 1) + current_length] = igrad_x[:, 0]
    current_length = current_length + s_y

    # central differences in x-direction
    aux_ix, aux_iy = np.meshgrid(np.arange(2, s_y), np.arange(0, (s_x)))
    aux_i = np.ndarray.flatten(np.transpose(aux_iy)) + np.multiply(np.ndarray.flatten(np.transpose(aux_ix) - 1), s_y)
    aux_i = np.array([aux_i])
    end_i = s_y * (s_x - 2)
    del ind_m, ind_n
    ind_m = matlib.repmat(current_length + np.arange(0, end_i, 1), 2, 1)
    ind_n = np.concatenate((aux_i - s_y, aux_i + s_y), axis=0)

    diffopx = np.divide(1, np.ndarray.flatten(diffx[aux_ix - 1, 0]) + np.ndarray.flatten(diffx[aux_ix, 0]))
    diffopx = np.array([diffopx])
    diffopx = np.multiply(np.transpose(diffopx), np.array([-1, 1]))

    aux_A[np.arange(0, end_i) + current_length, :] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopx),
                                                                    axis=1)

    aigrad_x = np.ndarray.flatten(igrad_x)
    aigrad_x = np.array(aigrad_x)

    b[np.arange(0, end_i) + current_length] = aigrad_x[aux_i]

    current_length = current_length + end_i

    # define forward differences for right hand edge in x-direction
    aux_ix = s_x
    aux_iy = np.arange(0, s_y, 1)
    aux_i = s_y * (np.transpose(aux_ix) - 1) + np.transpose(aux_iy)

    ind_m = matlib.repmat(aux_iy + current_length, 2, 1)
    ind_n = np.array([aux_i - s_y, aux_i])

    diffopx = matlib.repmat(np.divide(np.array([-1, 1]), diffx[-1]), s_y, 1)

    aux_A[np.arange(0, s_y) + current_length] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopx),
                                                               axis=1)

    b[np.arange(0, s_y, 1) + current_length] = igrad_x[:, -1]
    current_length = current_length + s_y

    ###
    # define forward differences for left hand edge in y-direction
    aux_ix = np.arange(0, s_x, 1)
    aux_iy = 0
    aux_i = s_y * (np.transpose(aux_ix)) + aux_iy

    ind_m = matlib.repmat(aux_ix + current_length, 2, 1)
    ind_n = np.array([aux_i, aux_i + 1])

    diffopy = matlib.repmat(np.divide(np.array([-1, 1]), diffy[1]), s_x, 1)

    aux_A[np.arange(0, s_x) + current_length] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopy),
                                                               axis=1)

    b[np.arange(0, s_y, 1) + current_length] = igrad_y[:, 0]
    current_length = current_length + s_x

    # central differences in y-direction
    aux_ix, aux_iy = np.meshgrid(np.arange(0, (s_x)), np.arange(1, s_y - 1))
    aux_i = np.ndarray.flatten(np.transpose(aux_iy)) + np.multiply((np.ndarray.flatten(np.transpose(aux_ix))), s_y)
    aux_i = np.array([aux_i])
    end_i = s_x * (s_y - 2)

    ind_m = matlib.repmat(current_length + np.arange(0, end_i, 1), 2, 1)
    ind_n = np.concatenate((aux_i - 1, aux_i + 1), axis=0)
    diffopy = np.divide(1, np.ndarray.flatten(diffy[aux_iy - 1, 0]) + np.ndarray.flatten(diffy[aux_iy, 0]))
    diffopy = np.array([diffopy])
    diffopy = np.multiply(np.transpose(diffopy), np.array([-1, 1]))

    aux_A[np.arange(0, end_i) + current_length, :] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopy),
                                                                    axis=1)

    aigrad_y = np.ndarray.flatten(igrad_y)
    aigrad_y = np.array(aigrad_y)

    b[np.arange(0, end_i) + current_length] = aigrad_y[aux_i]

    current_length = current_length + end_i

    # define forward differences for right hand edge in y-direction
    aux_ix = np.arange(0, s_x, 1)
    aux_iy = s_y - 1
    aux_i = s_y * (np.transpose(aux_ix)) + aux_iy

    ind_m = matlib.repmat(aux_ix + current_length, 2, 1)
    ind_n = np.array([aux_i - 1, aux_i])

    diffopy = matlib.repmat(np.divide(np.array([-1, 1]), diffy[-1]), s_x, 1)

    aux_A[np.arange(0, s_x) + current_length] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopy),
                                                               axis=1)

    b[np.arange(0, s_x, 1) + current_length] = igrad_y[:, -1]

    ###
    # sparse matrix transformations
    data = np.ndarray.flatten(aux_A[:, 4:6])
    indices = np.ndarray.flatten(aux_A[:, 0:2])
    indptr = np.ndarray.flatten(aux_A[:, 2:4])
    A = csr_matrix((data, (indices, indptr)), shape=(2 * s_x * s_y, s_x * s_y)).toarray()  #

    # add integration constant
    b1 = b - A[:, 0] * iconst
    b2 = np.array(b1)
    # solve linear system
    sa1, sa2 = np.shape(A)
    aux_a1 = np.linalg.lstsq(A[:, 1:sa2], b2)
    #aux_a1 = np.linalg.solve(A[:, 1:sa2], b2)

    aux_a2 = aux_a1[0]
    iconst = np.array([iconst])
    aux_a3 = np.concatenate((iconst, aux_a2))

    oa = np.reshape(aux_a3, (s_y, s_x))

    return oa


