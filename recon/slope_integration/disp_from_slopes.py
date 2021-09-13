import logging
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr


def disp_from_slopes(slopes_x, slopes_y, pixel_size, zero_at="bottom",zero_at_size=1, extrapolate_edge=0, filter_sigma=0, downsample=1):
    """ Calculate displacement fields by integrating slope fields using sparse matrix integration.
        The slopes can be filtered by a gaussian low-pass filter, downsampled and extrapolated before integration.
        Parameters
        ----------
        slopes_x : np.ndarray
            Slope field with shape (n_frames,n_pix_x,n_pix_y)
        slopes_y : np.ndarray
            Slope field with shape (n_frames,n_pix_x,n_pix_y)
        pixel_size : float
            The size of a pixel in slopes_x and slopes_y
        zero_at : string
            The position where a zero displacement boundary condition is enforced.
            Note that the boundary condition is not enforced exactly.
            Keywords:
                "top", "top_corners", "left", "right", "bottom", "bottom_corners"
        zero_at_size : int
            If zero_at is set to a corner, a rectangluar window with side lengths of zero_at_size  are used.
        extrapolate_edge : int
            Extrapolate edge by n-pixels by padding with the boundary values
        filter_sigma : float
            The standard deviation of the gaussian low pass filter
        downsample : int
            Downsample the field before integration by n-pixels.

        Returns
        -------
        ndarray
            The displacement fields with shape (n_frames,n_pix_x,n_pix_y)
        """
    logger = logging.getLogger(__name__)

    if slopes_x.ndim != 3 or slopes_y.ndim != 3:
        raise ValueError("The slope fields have to have the shape (n_frames,n_pix_x,n_pix_y)")
    n_frames,n_pix_x, n_pix_y = slopes_x.shape

    if type(downsample) != int or downsample < 1:
        raise ValueError("The downsampling factor has to be an integer larger or equal to 1")

    disp_fields = []

    for i in range(n_frames):
        logger.info("Integrating frame %i" % (i))
        slope_y = slopes_y[i,:, :]
        slope_x = slopes_x[i,:, :]

        if filter_sigma >0:
            slope_y = gaussian_filter(slope_y, sigma=filter_sigma)
            slope_x = gaussian_filter(slope_x, sigma=filter_sigma)

        if downsample>0:
            slope_y = slope_y[::downsample, ::downsample]
            slope_x = slope_x[::downsample, ::downsample]

        if extrapolate_edge > 0:
            slope_x = np.pad(slope_x, pad_width=(extrapolate_edge, extrapolate_edge), mode="edge")
            slope_y = np.pad(slope_y, pad_width=(extrapolate_edge, extrapolate_edge), mode="edge")

        disp_field = int2D(slope_x, slope_y, pixel_size * downsample, pixel_size * downsample)

        if zero_at == "top":
            edge_mean = np.mean(disp_field[0, :])
        elif zero_at == "top corners":
            edge_mean = (np.mean(disp_field[:zero_at_size, :zero_at_size]) + np.mean(disp_field[:zero_at_size, -zero_at_size:])) / 2.
        elif zero_at == "left":
            edge_mean = np.mean(disp_field[:, 0])
        elif zero_at == "right":
            edge_mean = np.mean(disp_field[:, -1])
        elif zero_at == "bottom":
            edge_mean = np.mean(disp_field[-1, :])
        elif zero_at == "bottom corners":
            edge_mean = (np.mean(disp_field[-zero_at_size:, :zero_at_size]) + np.mean(disp_field[-zero_at_size:, -zero_at_size:])) / 2.
        else:
            raise ValueError("No valid zero_at received")

        disp_field = disp_field - edge_mean

        disp_fields.append(disp_field)

    return np.array(disp_fields)


def int2D(igrad_x, igrad_y, idx, idy, const_at_edge=False):
    """
    Sparse matrix integration to solve grad(a) = b for a
    Requires integration constant (= reference value at position [1,1] = iconst)
    Assumes uniform spacing

    Based on:
    https://math.stackexchange.com/questions/1340719/numerically-find-a-potential-field-from-gradient

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
    # solve linear system, skipping the integration constant
    aux_a2 = lsqr(A[:, :], b)[0]
    aux_a3 = aux_a2.reshape((n_x, n_y))

    # Set the value of the first point to zero
    aux_a3 = aux_a3-aux_a3[0,0]

    return aux_a3