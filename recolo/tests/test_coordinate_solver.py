from unittest import TestCase
from recolo.artificial_grid_deformation import find_coords_in_undef_conf, interpolated_disp_field
import numpy as np


def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2) ** 2.))


def biharmonic_disp_field(x, y, amp_scale=0.5):
    return (amp_scale * 0.4 * np.cos(np.pi * x / 30) + amp_scale * 0.5 * np.sin(np.pi * y / 40)), (
            amp_scale * 0.6 * np.cos(np.pi * x / 50) + amp_scale * 0.7 * np.sin(np.pi * y / 60))


class TestFindCoordinatesInUndefConf(TestCase):
    # As X is needed for other calculations, check that we can determine X from x = X + u(X)

    def test_analytical_disp_field(self):
        tol = 1e-5
        dx = 3.5
        dy = 2.7
        xs, ys = np.meshgrid(np.arange(0, 80, dx), np.arange(0, 100, dy))
        Xs, Ys = find_coords_in_undef_conf(xs, ys, biharmonic_disp_field, tol=1e-9)

        u_X, u_Y = biharmonic_disp_field(Xs, Ys)

        errors_x = xs - Xs - u_X
        errors_y = ys - Ys - u_Y
        peak_error_x = np.max(np.abs(errors_x))
        peak_error_y = np.max(np.abs(errors_y))

        if peak_error_x > tol or peak_error_y > tol:
            self.fail("Maximum error is %f and %f" % (peak_error_x, peak_error_y))

    def test_interpolated_disp_field(self):
        tol = 1e-5
        dx = 3.5
        dy = 2.7
        xs, ys = np.meshgrid(np.arange(0, 80, dx), np.arange(0, 100, dy))
        # Make an approximated displacement field
        u_x, u_y = biharmonic_disp_field(xs, ys)
        disp_func_interp = interpolated_disp_field(u_x, u_y, dx=2, dy=4, order=3)

        X, Y = find_coords_in_undef_conf(xs, ys, disp_func_interp, tol=1e-9)

        u_X, u_Y = disp_func_interp(X, Y)

        errors_x = xs - X - u_X
        errors_y = ys - Y - u_Y

        peak_error_x = np.max(np.abs(errors_x))
        peak_error_y = np.max(np.abs(errors_y))

        if peak_error_x > tol or peak_error_y > tol:
            self.fail("Maximum error is %f and %f" % (peak_error_x, peak_error_y))

    def test_compare_interpolated_and_analytical(self):
        # As there will always be minor error at the edges, we look at the mean error for the whole field
        tol = 1.e-3
        dx = 3.5
        dy = 2.7
        xs, ys = np.meshgrid(np.arange(0, 80, dx), np.arange(0, 100, dy))

        # Make an approximated displacement field0
        u_x, u_y = biharmonic_disp_field(xs, ys)
        disp_func_interp = interpolated_disp_field(u_x, u_y, dx=dx, dy=dy, order=3, mode="nearest")

        X_interp, Y_interp = find_coords_in_undef_conf(xs, ys, disp_func_interp, tol=1e-9)
        X, Y = find_coords_in_undef_conf(xs, ys, biharmonic_disp_field, tol=1e-9)

        rms_diff_X = rms_diff(X_interp, X)
        rms_diff_Y = rms_diff(Y_interp, Y)

        if rms_diff_X > tol or rms_diff_Y > tol:
            self.fail("RMS error is %f and %f" % (rms_diff_X, rms_diff_Y))

    def test_check_grid_sampling_independency(self):
        # Ensure that the sampling of u_x and u_y does not have a large impact on the final results
        tol = 1.e-3

        dxs = [0.1,0.5,1.0,3.2]

        for i,dx in enumerate(dxs):
            dy = dx + 0.12
            xs, ys = np.meshgrid(np.arange(0, 80, dx), np.arange(0, 100, dy))

            # Make an approximated displacement field0
            u_x, u_y = biharmonic_disp_field(xs, ys)
            disp_func_interp = interpolated_disp_field(u_x, u_y, dx=dx, dy=dy, order=3, mode="nearest")

            X_interp, Y_interp = find_coords_in_undef_conf(xs, ys, disp_func_interp, tol=1e-9)
            X, Y = find_coords_in_undef_conf(xs, ys, biharmonic_disp_field, tol=1e-9)

            rms_diff_X = rms_diff(X_interp, X)
            rms_diff_Y = rms_diff(Y_interp, Y)

            if rms_diff_X > tol or rms_diff_Y > tol:
                self.fail("RMS error is %f and %f for dx=%f and dy=%f" % (rms_diff_X, rms_diff_Y,dx,dy))