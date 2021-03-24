import numpy as np
from numpy import matlib

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


def int2D(igrad_x, igrad_y, iconst, idx, idy):
    """
    Sparse matrix integration to solve grad(a) = b for a
    Requires integration constant (= reference value at position [1,1] = iconst)
    Assumes uniform spacing

    01.10.2019

    @author: Rene Kaufmann
    """
    # TODO: Make code pythonic
    # TODO: Clean up!

    n_x, n_y = np.shape(igrad_x)

    indices = np.arange(n_x * n_y, dtype=int).reshape((n_x, n_y))
    indices_y, indices_x = np.meshgrid(np.arange(n_y), np.arange(n_x))

    # initialize variables
    b = np.zeros(2 * n_x * n_y)
    datas = np.zeros((2 * n_x * n_y, 2))
    indptrs = np.zeros((2 * n_x * n_y, 2))
    cov_inds = np.zeros((2 * n_x * n_y, 2))
    current_length = 0

    # define forward differences for left hand edge in x-direction
    flat_inds = indices[0, :]
    ind_m = (flat_inds + current_length) * np.ones((2, 1))
    ind_n = np.array((flat_inds, flat_inds))

    dfdx = (np.array([-1, 1]) / idx) * np.ones((n_y, 1))

    datas[np.arange(0, np.size(flat_inds)) + current_length, :] = dfdx
    indptrs[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_n)
    cov_inds[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_m)

    b[np.arange(0, n_y, 1) + current_length] = igrad_x[0, :]
    current_length = current_length + n_y

    # central differences in x-direction
    indx = indices_x[1:-1]
    flat_inds = indices[1:-1, :].flatten()
    end_i = np.size(flat_inds)

    ind_m = (current_length + np.arange(0, end_i, 1)) * np.ones((2, 1))
    # Indices to plus-minus one pixel along x
    ind_n = np.concatenate(([indices[0:-2, :].flatten()], [indices[2:, :].flatten()]))

    dfdx = 1. / (2. * idx) * np.ones_like(indx).flatten()
    dfdx = np.array([-1, 1]) * dfdx[:, np.newaxis]

    datas[np.arange(0, np.size(flat_inds)) + current_length, :] = dfdx
    indptrs[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_n)
    cov_inds[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_m)

    b[np.arange(end_i) + current_length] = igrad_x.flatten()[flat_inds]

    current_length = current_length + end_i

    # define forward differences for right hand edge in x-direction
    indy = indices_y[-1, :]
    flat_inds = indices[-1, :]

    ind_m = (indy + current_length) * np.ones((2, 1))
    ind_n = np.array([flat_inds - n_y, flat_inds])

    dfdx = np.array([-1, 1]) / idx * np.ones((n_y, 1))

    datas[np.arange(0, np.size(flat_inds)) + current_length, :] = dfdx
    indptrs[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_n)
    cov_inds[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_m)

    b[indy + current_length] = igrad_x[-1, :]
    current_length = current_length + n_y

    # define forward differences for left hand edge in y-direction
    indx = np.arange(0, n_x)
    flat_inds = indices[:, 0]

    ind_m = (indx + current_length) * np.ones((2, 1))
    ind_n = np.array([flat_inds, flat_inds + 1])

    dfdy = (np.array([-1, 1]) / idy) * np.ones((n_x, 1))

    datas[np.arange(0, np.size(flat_inds)) + current_length, :] = dfdy
    indptrs[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_n)
    cov_inds[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_m)

    b[np.arange(0, n_x, 1) + current_length] = igrad_y[:, 0]
    current_length = current_length + n_x

    # central differences in y-direction
    indx, indy = np.meshgrid(np.arange(0, (n_x)), np.arange(1, n_y - 1))
    flat_inds = indices[:, 1:-1].flatten()

    flat_inds = np.array([flat_inds])
    end_i = n_x * (n_y - 2)

    ind_m = (current_length + np.arange(0, end_i, 1)) * np.ones((2, 1))
    ind_n = np.concatenate((flat_inds - 1, flat_inds + 1), axis=0)

    # diffopy = 1. / (diffy[indy - 1, 0].flatten() + (diffy[indy, 0].flatten()))
    dfdy = 1. / (2. * idy) * np.ones_like(indy).flatten()
    dfdy = dfdy.transpose()[:, np.newaxis] * np.array([-1, 1])

    datas[np.arange(0, np.size(flat_inds)) + current_length, :] = dfdy
    indptrs[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_n)
    cov_inds[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_m)

    aigrad_y = igrad_y.flatten()

    b[np.arange(0, end_i) + current_length] = aigrad_y[flat_inds]

    current_length = current_length + end_i

    # define forward differences for right hand edge in y-direction
    indx = np.arange(0, n_x, 1)
    flat_inds = indices[:, -1]
    ind_m = (indx + current_length) * np.ones((2, 1))
    ind_n = np.array([flat_inds - 1, flat_inds])
    dfdy = (np.array([-1, 1]) / idy) * np.ones((n_x, 1))

    datas[np.arange(0, np.size(flat_inds)) + current_length, :] = dfdy
    indptrs[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_n)
    cov_inds[np.arange(0, np.size(flat_inds)) + current_length, :] = np.transpose(ind_m)
    b[np.arange(0, n_x, 1) + current_length] = igrad_y[:, -1]

    # sparse matrix transformations
    A = csr_matrix((datas.flatten(), (cov_inds.flatten(), indptrs.flatten())), shape=(2 * n_x * n_y, n_x * n_y))  #
    A_Arr = A.toarray()  #

    # add integration constant
    b2 = b - A_Arr[:, 0] * iconst
    # solve linear system, skipping the integration constant
    aux_a2 = lsqr(A[:, 1:], b2)[0]

    aux_a3 = np.concatenate(([iconst], aux_a2))

    return aux_a3.reshape((n_x, n_y))
