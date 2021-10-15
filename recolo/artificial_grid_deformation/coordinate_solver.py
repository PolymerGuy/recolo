import logging
import numpy as np
from scipy.ndimage import map_coordinates


def find_coords_in_undef_conf(xs, ys, disp_func, tol=1e-7, maxit=20):
    """
    Solves x = X + u(X) for X for a given u().
    This is done by solving  x - X - u(X) = 0 using a Newton scheme with numerically calculated gradients.


    Parameters
    ----------
    xs : ndarray
        The x-coordinate field in the deformed configuration
    ys : ndarray
        The y-coordinate field in the deformed configuration
    disp_func : func
        The displacement field as a function.
        The function must be on the form disp_x,disp_y = disp_func(xs,ys)
    tol : float
        The tolerance used for the residuals
    maxit : int
        The maximum number of iterations used
    Returns
    -------
    Xs, Ys: ndarray
        The coordinate components in the undeformed configurations

    """

    logger = logging.getLogger(__name__)

    # An initial guess for the coordinates in the undeformed configuration
    Xs = xs.copy()
    Ys = ys.copy()

    # Estimate the increment size along both axes
    dx = np.mean(np.gradient(xs, axis=1))
    dy = np.mean(np.gradient(ys, axis=0))

    # Define the function to be minimized
    def func(Xsi, Ysi):
        ux, uy = disp_func(Xsi, Ysi)
        return (xs - Xs - ux), (ys - Ys - uy)

    # Define the corresponding Jacobian matrix
    def jacobian(Xs, Ys, dx, dy):

        ux, uy = disp_func(Xs, Ys)
        duxdy, duxdx = np.gradient(ux, dx, edge_order=1)
        duydy, duydx = np.gradient(uy, dy, edge_order=1)

        jac = -np.array([[duxdx + 1, duxdy], [duydx, duydy + 1]])
        jac = np.moveaxis(jac, -1, 0)
        jac = np.moveaxis(jac, -1, 0)
        return jac

    # Full newton iterations
    for i in range(maxit):
        jac_inv = np.linalg.inv(jacobian(Xs, Ys, dx, dy))
        res = np.einsum('ijkl,kij->ijl', jac_inv, np.array(func(Xs, Ys)))
        dXs = res[:, :, 0]
        dYs = res[:, :, 1]

        Xs = Xs - dXs
        Ys = Ys - dYs

        if np.max(np.abs(dXs)) < tol and np.max(np.abs(dYs)) < tol:
            logger.info(
                "Converged in %i iterations with a final residual of: %f "%(i,np.max(np.abs(np.array(func(Xs, Ys))))))
            return Xs, Ys

    raise ValueError("Did not converge to %f in %i iterations" % (tol, i))


def interpolated_disp_field(u_x, u_y, dx, dy, order=3, mode="nearest"):
    """
    Interpolate the displacement fields given by u_x and u_y by means of B-splines
    Parameters
    ----------
    u_x : ndarray
        The displacement field values along the x-axis
    u_y : ndarray
        The displacement field values along the y-axis
    dx : float
        Step size along the x-axis
    dy : float
        Step size along the y-axis
    order : int
        Interpolation order
    mode : string
        Interpolation mode.

    Returns
    -------
    disp_func: func
        The B-spline interpolators for u_x and u_y
    """

    def interpolated_disp_func(xs, ys):
        int_u_x = map_coordinates(u_x, [ys / dy, xs / dx], mode=mode, order=order)
        int_u_y = map_coordinates(u_y, [ys / dy, xs / dx], mode=mode, order=order)
        return int_u_x, int_u_y

    return interpolated_disp_func
