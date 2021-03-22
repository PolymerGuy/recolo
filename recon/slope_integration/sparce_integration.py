"""
Sparse matrix integration to solve grad(a) = b for a
Requires integration constant (= reference value at position [1,1] = iconst)
Assumes uniform spacing

01.10.2019

@author: Rene Kaufmann
"""

import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix


###
def int2D(igrad_x, igrad_y, iconst, idx, idy):
    # TODO: Make code pythonic
    n_x, n_y = np.shape(igrad_x)

    diffx = np.ones((n_x,1)) * idx
    diffy = np.ones((n_y,1)) * idy

    indices = np.arange(n_x*n_y,dtype=np.int).reshape((n_x,n_y))


    # initialize variables
    b = np.zeros(2 * n_x * n_y)
    aux_A = np.zeros((2 * n_x * n_y, 6))
    current_length = 0

    # define forward differences for left hand edge in x-direction

    indy = indices[0,:]
    inds = indy

    ind_m = (indy + current_length) * np.ones((2, 1))
    ind_n = np.array((inds, inds + n_y))

    dfdx = (np.array([-1, 1]/diffx[1])) * np.ones((n_y, 1))

    aux_A[indy + current_length] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), dfdx),
                                                               axis=1)

    b[np.arange(0, n_y, 1) + current_length] = igrad_x[0, :]
    current_length = current_length + n_y

    # central differences in x-direction
    indx, indy = np.meshgrid(np.arange(1, n_x-1), np.arange(0, n_y))

    inds = indices[1:-1,:].flatten()

    inds = np.array([inds])
    end_i = n_y * (n_x - 2)
    del ind_m, ind_n
    ind_m = matlib.repmat(current_length + np.arange(0, end_i, 1), 2, 1)
    ind_n = np.concatenate((inds - n_y, inds + n_y), axis=0)

    dfdx = 1./(diffx[indx - 1, 0].flatten() + diffx[indx, 0].flatten())
    dfdx = np.array([dfdx])
    dfdx = np.multiply(np.transpose(dfdx), np.array([-1, 1]))

    aux_A[np.arange(0, end_i) + current_length, :] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), dfdx),
                                                                    axis=1)

    b[np.arange(0, end_i) + current_length] = igrad_x.flatten()[inds]


    current_length = current_length + end_i

    # define forward differences for right hand edge in x-direction
    indx = n_x
    indy = np.arange(0, n_y)
    inds = n_y * (np.transpose(indx) - 1) + np.transpose(indy)
    inds = indices[-1,:]

    #inds = indx * (indy - 1.) + n_y

    ind_m = matlib.repmat(indy + current_length, 2, 1)
    ind_n = np.array([inds - n_y, inds])

    dfdx = matlib.repmat(np.divide(np.array([-1, 1]), diffx[-1]), n_y, 1)

    aux_A[np.arange(0, n_y) + current_length] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), dfdx),
                                                               axis=1)

    b[np.arange(0, n_y, 1) + current_length] = igrad_x[-1, :]
    current_length = current_length + n_y

    ###
    # define forward differences for left hand edge in y-direction
    indx = np.arange(0, n_x, 1)
    inds = indices[:,0]

    ind_m = (indx + current_length) * np.ones((2, 1))
    ind_n = np.array([inds, inds + 1])
    print("inds should be zero", ind_n)



    diffopy = matlib.repmat(np.divide(np.array([-1, 1]), diffy[1]), n_x, 1)

    aux_A[np.arange(0, n_x) + current_length] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopy),
                                                               axis=1)

    b[np.arange(0, n_y, 1) + current_length] = igrad_y[:, 0]
    current_length = current_length + n_x

    # central differences in y-direction
    indx, indy = np.meshgrid(np.arange(0, (n_x)), np.arange(1, n_y - 1))
    inds = np.ndarray.flatten(np.transpose(indy)) + np.multiply((np.ndarray.flatten(np.transpose(indx))), n_y)
    inds = indices[:,1:-1].flatten()

    inds = np.array([inds])
    end_i = n_x * (n_y - 2)

    ind_m = matlib.repmat(current_length + np.arange(0, end_i, 1), 2, 1)
    ind_n = np.concatenate((inds - 1, inds + 1), axis=0)

    diffopy = 1./(diffy[indy - 1, 0].flatten() + (diffy[indy, 0].flatten()))
    diffopy = np.array([diffopy])
    diffopy = np.multiply(np.transpose(diffopy), np.array([-1, 1]))

    aux_A[np.arange(0, end_i) + current_length, :] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopy),
                                                                    axis=1)

    aigrad_y = np.ndarray.flatten(igrad_y)
    aigrad_y = np.array(aigrad_y)

    b[np.arange(0, end_i) + current_length] = aigrad_y[inds]

    current_length = current_length + end_i

    # define forward differences for right hand edge in y-direction
    indx = np.arange(0, n_x, 1)
    inds = indices[:,-1]

    ind_m = matlib.repmat(indx + current_length, 2, 1)
    ind_n = np.array([inds - 1, inds])

    diffopy = matlib.repmat(np.divide(np.array([-1, 1]), diffy[-1]), n_x, 1)

    aux_A[np.arange(0, n_x) + current_length] = np.concatenate((np.transpose(ind_m), np.transpose(ind_n), diffopy),
                                                               axis=1)

    b[np.arange(0, n_x, 1) + current_length] = igrad_y[:, -1]

    ###
    # sparse matrix transformations
    data = aux_A[:, 4:6].flatten()
    indices = aux_A[:, 0:2].flatten()
    indptr = aux_A[:, 2:4].flatten()
    A = csr_matrix((data, (indices, indptr)), shape=(2 * n_x * n_y, n_x * n_y)).toarray()  #

    # add integration constant
    b1 = b - A[:, 0] * iconst
    b2 = np.array(b1)
    # solve linear system
    sa1, sa2 = np.shape(A)
    aux_a1 = np.linalg.lstsq(A[:, 1:sa2], b2)

    aux_a2 = aux_a1[0]
    iconst = np.array([iconst])
    aux_a3 = np.concatenate((iconst, aux_a2))

    oa = np.reshape(aux_a3, (n_y, n_x))

    return oa






