import numpy as np
from scipy.ndimage import map_coordinates
from scipy import interpolate
import matplotlib.pyplot as plt


# Solves x =X +u(X) for X


def solve_coordinates(xs, ys, disp_func, tol=1e-7, maxit=20):
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
            print("Converged in %i iterations with a final residual of: "%i)
            print(np.max(np.abs(np.array(func(Xs, Ys)))))
            return Xs, Ys

    raise ValueError("Did not converge to %f in %i iterations" % (tol, i))


def disp_func(x, y):
    return (0.5 * np.sin(np.pi * x / 50) + 0.7 * np.sin(np.pi * y / 50)), (
            0.5 * np.sin(np.pi * x / 50) + 0.7 * np.sin(np.pi * y / 50))


def interpolated_disp_func(u_x, u_y, dx,dy, order=3,mode="nearest"):
    def interpolated_disp_funcs(xs, ys):
        int_u_x = map_coordinates(u_x,[ys/dy,xs/dx],mode=mode,order=order)
        int_u_y = map_coordinates(u_y,[ys/dy,xs/dx],mode=mode,order=order)
        return int_u_x,int_u_y
    return interpolated_disp_funcs


xs, ys = np.meshgrid(np.arange(0, 80, 4), np.arange(0, 100, 4))
u_x,u_y = disp_func(xs,ys)
disp_func_int = interpolated_disp_func(u_x,u_y,4,4)
int_u_x,int_u_y = disp_func_int(xs,ys)
#plt.imshow(int_u_y-u_y)
#plt.show()

x_cor, y_cor = solve_coordinates(xs, ys, disp_func)
x, y = solve_coordinates(xs, ys, disp_func_int)

plt.figure()
plt.imshow(x_cor-x)
plt.figure()
plt.imshow(x-xs)
plt.show()
