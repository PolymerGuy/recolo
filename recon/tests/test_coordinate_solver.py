from unittest import TestCase
from recon.artificial_grid_deformation import *
import numpy as np


def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2) ** 2.))


def disp_func(x, y,amp_scale=0.5):
    return (amp_scale*0.4 * np.cos(np.pi * x / 30) + amp_scale*0.5 * np.sin(np.pi * y / 40)), (
            amp_scale*0.6 * np.cos(np.pi * x / 50) + amp_scale*0.7 * np.sin(np.pi * y / 60))


class TestFindCoordinatesInUndefConf(TestCase):
    # Check that x = X + u(X)
    def test_analytical_disp_field(self):
        tol = 1e-5
        xs, ys = np.meshgrid(np.arange(0, 80, 2), np.arange(0, 100, 4))
        X, Y = find_coords_in_undef_conf(xs, ys, disp_func, tol=1e-9)

        u_X, u_Y = disp_func(X, Y)

        residual_x = xs - X - u_X
        residual_y = ys - Y - u_Y
        max_res_x = np.max(np.abs(residual_x))
        max_res_y = np.max(np.abs(residual_y))

        if max_res_x > tol or max_res_y > tol:
            self.fail("Maximum error is %f and %f" % (max_res_x, max_res_y))

    def test_interpolated_disp_field(self):
        tol = 1e-5
        xs, ys = np.meshgrid(np.arange(0, 80, 2), np.arange(0, 100, 4))
        u_x, u_y = disp_func(xs, ys)
        disp_func_int = interpolated_disp_field(u_x, u_y, 2, 4)

        X, Y = find_coords_in_undef_conf(xs, ys, disp_func_int, tol=1e-9)

        u_X, u_Y = disp_func_int(X, Y)

        residual_x = xs - X - u_X
        residual_y = ys - Y - u_Y

        max_res_x = np.max(np.abs(residual_x))
        max_res_y = np.max(np.abs(residual_y))

        if max_res_x > tol or max_res_y > tol:
            self.fail("Maximum error is %f and %f" % (max_res_x, max_res_y))

    def test_compare_interpolated_and_analytical(self):
        # As there will always be minor error at the edges, we look at the RMS difference for the whole field
        tol = 1e-3
        xs, ys = np.meshgrid(np.arange(0, 80, 0.5), np.arange(0, 100, 0.7))
        u_x, u_y = disp_func(xs, ys)
        disp_func_int = interpolated_disp_field(u_x, u_y, 0.5, 0.7,order=3,mode="nearest")

        X, Y = find_coords_in_undef_conf(xs, ys, disp_func_int, tol=1e-9)

        # Check that x - X - u(X) = 0
        u_X_int, u_Y_int = disp_func_int(X, Y)
        u_X, u_Y = disp_func(X, Y)
        rms_diff_u_x = rms_diff(u_X,u_X_int)
        rms_diff_u_y = rms_diff(u_Y,u_Y_int)

        if rms_diff_u_x > tol or rms_diff_u_y > tol:
            self.fail("RMS error is %f and %f" % (rms_diff_u_x, rms_diff_u_y))
