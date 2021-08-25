"""
Tools for performing displacement measurements using the grid method, see [1] for details and discussion.

References
----------
    .. [1] The grid method for in-plane displacement and strain measurement: a review and analysis Michel Grediac,
    Frédéric Sur, Benoît Blaysat

"""

import logging
import numpy as np
from scipy import signal
from scipy.ndimage import map_coordinates
from scipy.signal.windows import triang
from skimage.restoration import unwrap_phase

def gaussian_window(win_size):
    t_noy = np.ceil(4 * win_size)
    xs, ys = np.meshgrid(np.arange(-t_noy, t_noy + 1), np.arange(-t_noy, t_noy + 1))
    conv_matrix = np.exp(-(xs ** 2. + ys ** 2.) / (2. * (win_size ** 2.)))
    return conv_matrix / conv_matrix.sum()


def triangular_window(win_size):
    t_noy = np.floor(win_size)
    g1 = triang(2 * t_noy - 1)
    conv_matrix = np.outer(g1, g1)
    raise NotImplementedError("The triangular window does not yield valid results")
    return conv_matrix / conv_matrix.sum()


def detect_phase(img, grid_pitch, window="gaussian", boundary="symm"):
    """ Detect the phase modulation along the coordinate axes by convolution with a gaussian window or a triangular
    window, see [1] for more details.

    Parameters
    ----------
    img : ndarray
        The image of the grid
    grid_pitch : int
        The grid pitch in pixels
    window : str
        The type of window used for the convolution. "gaussian" and "triangular" are valid input.
    boundary : str
        The boundary conditions used at the edges of the image.

    Returns
    -------
    ndarray, ndarray : ndarray
        The phase modulation fields along the image axes

    References
    ----------
    .. [1] The grid method for in-plane displacement and strain measurement: a review and analysis Michel Grediac,
    Frédéric Sur, Benoît Blaysat
    """
    s_x, s_y = np.shape(img)
    fc = 2. * np.pi / float(grid_pitch)

    if window == "triangular":
        conv_matrix = triangular_window(grid_pitch)
    else:
        conv_matrix = gaussian_window(grid_pitch)

    xs, ys = np.meshgrid(np.arange(s_y), np.arange(s_x))
    # x-direction
    img_complex_x = img * np.exp(-1j * fc * xs)
    phase_x = signal.convolve2d(img_complex_x, conv_matrix, boundary=boundary, mode='valid') / float(grid_pitch)

    # y-direction
    img_complex_y = img * np.exp(-1j * fc * ys)
    phase_y = signal.convolve2d(img_complex_y, conv_matrix, boundary=boundary, mode='valid') / float(grid_pitch)

    return phase_x, phase_y


def angle_from_disp(disp, mirror_grid_dist):
    return np.arctan(disp / mirror_grid_dist) / 2.


def disp_from_phases_single_component(phase, phase0, grid_pitch, unwrap=True):
    """ Determine the displacement of every pixel based on the phase modulation in two configurations, see [1] for
    more details.

    Parameters
    ----------
    phase : ndarray
        The phase modulation field in the deformed configuration as complex numbers
    phase0 : ndarray
        The phase modulation field in the deformed configuration as complex numbers
    grid_pitch : int
        The grid pitch in pixels
    unwrap : bool
        Perform phase unwrapping using [2].

    Returns
    -------
    disp : ndarray
        The displacement field

    References
    ----------
    ..  [1] Michel Grediac, Frédéric Sur, Benoît Blaysat. The grid method for in-plane displacement and
    strain measurement: a review and analysis. Strain, Wiley-Blackwell, 2016, 52 (3), pp.205-243.
    ff10.1111/str.12182ff. ffhal-01317145f
        [2] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor, and Munther
    A. Gdeisat, "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a
    noncontinuous path", Journal Applied Optics, Vol. 41, No. 35 (2002) 7437,
    """
    if unwrap:
        return -grid_pitch * unwrap_phase(2 * np.angle(phase / phase0), wrap_around=True, seed=0) / 2. / 2. / np.pi
    else:
        return -grid_pitch * np.angle(phase / phase0) / 2. / np.pi


def disp_fields_from_phases(phase_x, phase_x_0, phase_y, phase_y_0, grid_pitch, correct_phase=True, maxit=10, tol=1e-5,
                            unwrap=True):
    """ Determine the displacements of every pixel based on the phase modulation along two axes in two configurations, see [1] for
    more details.

    Parameters
    ----------
    phase_x : ndarray
        The phase modulation field along the x-axis in the deformed configuration as complex numbers
    phase_x_0 : ndarray
        The phase modulation field along the x-axis in the undeformed configuration as complex numbers
    phase_y : ndarray
        The phase modulation field along the y-axis in the deformed configuration as complex numbers
    phase_y_0 : ndarray
        The phase modulation field along the y-axis in the undeformed configuration as complex numbers
    grid_pitch : int
        The grid pitch in pixels
    correct_phase : bool
        Correct the phases for finite displacements
    maxit : int
        The maximum number of iterations for the phase correction
    tol : float
        The maximum allowable residual for the phase correction
    unwrap : bool
        Perform phase unwrapping using [2].

    Returns
    -------
    disp_x,disp_y : ndarray
        The displacement field

    References
    ----------
    ..  [1] Michel Grediac, Frédéric Sur, Benoît Blaysat. The grid method for in-plane displacement and
    strain measurement: a review and analysis. Strain, Wiley-Blackwell, 2016, 52 (3), pp.205-243.
    ff10.1111/str.12182ff. ffhal-01317145f
        [2] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor, and Munther
    A. Gdeisat, "Fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a
    noncontinuous path", Journal Applied Optics, Vol. 41, No. 35 (2002) 7437,
    """

    logger = logging.getLogger(__name__)

    if not correct_phase:
        u_x = disp_from_phases_single_component(phase_x, phase_x_0, grid_pitch, unwrap)
        u_y = disp_from_phases_single_component(phase_y, phase_y_0, grid_pitch, unwrap)
        return u_x, u_y

    if correct_phase:
        n_x, n_y = phase_x.shape
        xs, ys = np.meshgrid(np.arange(n_y), np.arange(n_x))
        u_x = disp_from_phases_single_component(phase_x, phase_x_0, grid_pitch, unwrap)
        u_y = disp_from_phases_single_component(phase_y, phase_y_0, grid_pitch, unwrap)

        u_x_first = u_x
        u_y_first = u_y
        for i in range(maxit):
            phase_x_n = map_coordinates(phase_x, [ys + u_y, xs + u_x], order=4, mode="mirror")
            phase_y_n = map_coordinates(phase_y, [ys + u_y, xs + u_x], order=4, mode="mirror")

            u_x_last = u_x.copy()
            u_y_last = u_y.copy()
            u_x = disp_from_phases_single_component(phase_x_n, phase_x_0, grid_pitch, unwrap)
            u_y = disp_from_phases_single_component(phase_y_n, phase_y_0, grid_pitch, unwrap)
            if np.max(np.abs(u_x - u_x_last)) < tol and np.max(np.abs(u_y - u_y_last)) < tol:
                logger.info("Phase correction converged in %i iterations correcting %.6f and %.6f pixels" % (
                    i, np.max(np.abs(u_x_first - u_x)), np.max(np.abs(u_y_first - u_y))))
                return u_x, u_y

        logger.info("Large displacement correction diverged, returning uncorrected frame")
        return u_x_first, u_x_first


def disp_from_grids(grid_undeformed, grid_deformed, grid_pitch, correct_phase=True):
    """ Determine the displacements of every pixel based the image of an undeformed grid and a deformed grid. This is
    done by determining the displacement from the phase modulation along two axes in two configurations,
    see [1] for more details.

    Parameters
    ----------
    grid_undeformed : ndarray
        The phase modulation field along the x-axis in the deformed configuration as complex numbers
    grid_deformed : ndarray
        The phase modulation field along the x-axis in the undeformed configuration as complex numbers
    grid_pitch : float
        The grid pitch in pixels
    correct_phase : bool
        Correct the phases for finite displacements
    Returns
    -------
    disp_x,disp_y : ndarray
        The displacement field

    References
    ----------
    ..  [1] Michel Grediac, Frédéric Sur, Benoît Blaysat. The grid method for in-plane displacement and
    strain measurement: a review and analysis. Strain, Wiley-Blackwell, 2016, 52 (3), pp.205-243.
    ff10.1111/str.12182ff. ffhal-01317145f
    """
    phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)
    phase_x, phase_y = detect_phase(grid_deformed, grid_pitch)
    disp_x_from_phase, disp_y_from_phase = disp_fields_from_phases(phase_x, phase_x0, phase_y, phase_y0,
                                                                   grid_pitch, correct_phase=correct_phase)

    return disp_x_from_phase, disp_y_from_phase
