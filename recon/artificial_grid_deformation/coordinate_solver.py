import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt


def find_coords_in_undef_conf(xs, ys, disp_func, tol=1e-7, maxit=20):
    """
    Solves x = X + u(X) for X.
    This is done by solving  x - X - u(X) = 0 using a Newton scheme with numerically calculated gradients.

    Parameters
    ----------
    xs
    ys
    disp_func
    tol
    maxit

    Returns
    -------
    Xs, Ys
    """
    #
    # Note that disp_func is now a smooth function
    Xs = xs.copy()
    Ys = ys.copy()

    dx = np.mean(np.gradient(xs, axis=1))
    dy = np.mean(np.gradient(ys, axis=0))

    def func(Xsi, Ysi):
        ux, uy = disp_func(Xsi, Ysi)
        return (xs - Xs - ux), (ys - Ys - uy)

    def jacobian(Xs, Ys, dx, dy):

        ux, uy = disp_func(Xs, Ys)
        duxdy, duxdx = np.gradient(ux, dx, edge_order=1)
        duydy, duydx = np.gradient(uy, dy, edge_order=1)

        jac = -np.array([[duxdx + 1, duxdy], [duydx, duydy + 1]])
        jac = np.moveaxis(jac, -1, 0)
        jac = np.moveaxis(jac, -1, 0)
        return jac

    for i in range(maxit):
        jac_inv = np.linalg.inv(jacobian(Xs, Ys, dx, dy))
        res = np.einsum('ijkl,kij->ijl', jac_inv, np.array(func(Xs, Ys)))
        u, _ = func(Xs, Ys)
        dXs = res[:, :, 0]
        dYs = res[:, :, 1]

        Xs = Xs - dXs
        Ys = Ys - dYs

        if np.max(np.abs(dXs)) < tol and np.max(np.abs(dYs)) < tol:
            print("Converged in %i iterations with a final residual of: " % i)
            print(np.max(np.abs(np.array(func(Xs, Ys)))))
            return Xs, Ys

    raise ValueError("Did not converge to %f in %i iterations" % (tol, i))



def interpolated_disp_field(u_x, u_y, dx, dy, order=3, mode="nearest"):
    """
    Interpolate fields given by u_x and u_y by means of B-splines
    Parameters
    ----------
    u_x : ndarray
        The displacement field along the x-axis
    u_y : ndarray
        The displacement field along the y-axis
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
    interp_u_x, interp_u_y : func
        The B-spline interpolators for u_x and u_y
    """
    def interpolated_disp_func(xs, ys):
        int_u_x = map_coordinates(u_x, [ys / dy, xs / dx], mode=mode, order=order)
        int_u_y = map_coordinates(u_y, [ys / dy, xs / dx], mode=mode, order=order)
        return int_u_x, int_u_y
    return interpolated_disp_func


