from recon.deflectomerty.deflectometry import detect_phase, disp_from_phase

from unittest import TestCase
import numpy as np


def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2)) ** 2.)


def grid_grey_scales(xs, ys, pitch):
    return np.cos(2. * np.pi * xs / float(pitch)) * np.cos(2. * np.pi * ys / float(pitch))


class Test_PhaseDetection(TestCase):

    def test_rigid_body_motion(self):

        rel_error_tol = 1e-3

        grid_pitch = 5
        n_pitches = 20

        displacement_x = 1.e-3
        displacement_y = 1.e-3

        x = np.arange(grid_pitch * n_pitches, dtype=float)
        y = np.arange(grid_pitch * n_pitches, dtype=float)

        xs, ys = np.meshgrid(x, y)

        grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase = disp_from_phase(phase_x,phase_x0,grid_pitch)
        disp_y_from_phase = disp_from_phase(phase_y,phase_y0,grid_pitch)

        max_rel_error_x = np.max(np.abs(disp_x_from_phase - displacement_x)) / displacement_x
        max_rel_error_y = np.max(np.abs(disp_y_from_phase - displacement_y)) / displacement_y

        if max_rel_error_x > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_x)
        if max_rel_error_y > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_y)
