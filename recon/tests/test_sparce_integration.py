from unittest import TestCase
import numpy as np
from recon.slope_integration import int2D

def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2)) ** 2.)


def peak_diff(array1, array2):
    return np.max(np.abs(array1 - array2))


class Test_integration_accuracy(TestCase):

    def test_sinusoidal_disp_field(self):
        tol_rms = 1.e-6
        tol_peak = 1.e-3

        n_pts_x, n_pts_y = 31, 35
        amp = 1.
        plate_len_x, plate_len_y, = 1.5, 1.5
        dx = plate_len_x / n_pts_x
        dy = plate_len_y / n_pts_y

        int_const = 0.

        xs, ys = np.meshgrid(np.linspace(0., 1., n_pts_x), np.linspace(0., 1., n_pts_y))

        disp_field = amp * np.sin(np.pi * xs) * np.sin(np.pi * ys)

        gradient_x, gradient_y = np.gradient(disp_field, dx, dy)

        disp_field_from_slopes = int2D(gradient_x, gradient_y, int_const, dx, dy)

        rms_error = rms_diff(disp_field, disp_field_from_slopes)
        peak_error = peak_diff(disp_field, disp_field_from_slopes)

        if rms_error > tol_rms:
            self.fail("RMS error is %f" % rms_error)

        if peak_error > tol_peak:
            self.fail("Peak error is %f" % rms_error)

    def test_sinus_cosine_disp_field(self):
        tol_rms = 1.e-6
        tol_peak = 1.e-3

        n_pts_x, n_pts_y = 31, 35
        amp = 1.
        plate_len_x, plate_len_y, = 1.5, 1.5
        dx = plate_len_x / n_pts_x
        dy = plate_len_y / n_pts_y

        int_const = 0.

        xs, ys = np.meshgrid(np.linspace(0., 1., n_pts_x), np.linspace(0., 1., n_pts_y))

        disp_field = amp * np.sin(np.pi * xs) * np.cos(np.pi * ys)

        gradient_x, gradient_y = np.gradient(disp_field, dx, dy)

        disp_field_from_slopes = int2D(gradient_x, gradient_y, int_const, dx, dy)

        rms_error = rms_diff(disp_field, disp_field_from_slopes)
        peak_error = peak_diff(disp_field, disp_field_from_slopes)

        if rms_error > tol_rms:
            self.fail("RMS error is %f" % rms_error)

        if peak_error > tol_peak:
            self.fail("Peak error is %f" % rms_error)

    def test_high_frequency_harmonic(self):
        tol_rms = 1.e-6
        tol_peak = 1.e-3

        n_periods_x = 3
        n_periods_y = 2

        n_pts_x, n_pts_y = 31, 35
        amp = 1.
        plate_len_x, plate_len_y, = 1.5, 1.5
        dx = plate_len_x / n_pts_x
        dy = plate_len_y / n_pts_y

        int_const = 1.5

        xs, ys = np.meshgrid(np.linspace(0., 1., n_pts_x), np.linspace(0., 1., n_pts_y))

        disp_field = amp * np.sin(n_periods_x * np.pi * xs) * np.sin(n_periods_y* np.pi * ys) +int_const

        gradient_x, gradient_y = np.gradient(disp_field, dx, dy)

        disp_field_from_slopes = int2D(gradient_x, gradient_y, int_const, dx, dy)

        rms_error = rms_diff(disp_field, disp_field_from_slopes)
        peak_error = peak_diff(disp_field, disp_field_from_slopes)

        if rms_error > tol_rms:
            self.fail("RMS error is %f" % rms_error)

        if peak_error > tol_peak:
            self.fail("Peak error is %f" % rms_error)