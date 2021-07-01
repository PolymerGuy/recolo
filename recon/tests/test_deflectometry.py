from recon.deflectomerty.deflectometry import detect_phase, disp_from_phase
from recon.artificial_grid_deformation import harmonic_disp_field, make_grid
from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt


def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2) ** 2.))




class Test_PhaseDetection(TestCase):

    def test_rigid_body_motion_small_disp(self):

        rel_error_tol = 1e-3

        grid_pitch = 5
        n_pitches = 20

        displacement_x = 1.e-2
        displacement_y = 1.e-2

        x = np.arange(grid_pitch * n_pitches, dtype=float)
        y = np.arange(grid_pitch * n_pitches, dtype=float)

        xs, ys = np.meshgrid(x, y)

        grid_undeformed = make_grid(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = make_grid(xs_disp, ys_disp, grid_pitch)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_from_phase(phase_x, phase_x0, phase_y, phase_y0, grid_pitch,
                                                               correct_phase=False)

        max_rel_error_x = np.max(np.abs(disp_x_from_phase - displacement_x)) / displacement_x
        max_rel_error_y = np.max(np.abs(disp_y_from_phase - displacement_y)) / displacement_y

        if max_rel_error_x > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_x)
        if max_rel_error_y > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_y)

    def test_rigid_body_motion_large_disp(self):

        rel_error_tol = 1e-3

        grid_pitch = 5
        n_pitches = 20

        displacement_x = 1.2
        displacement_y = 1.2

        x = np.arange(grid_pitch * n_pitches, dtype=float)
        y = np.arange(grid_pitch * n_pitches, dtype=float)

        xs, ys = np.meshgrid(x, y)

        grid_undeformed = make_grid(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = make_grid(xs_disp, ys_disp, grid_pitch)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_from_phase(phase_x, phase_x0, phase_y, phase_y0, grid_pitch,
                                                               correct_phase=True)

        max_rel_error_x = np.max(np.abs(disp_x_from_phase - displacement_x)) / displacement_x
        max_rel_error_y = np.max(np.abs(disp_y_from_phase - displacement_y)) / displacement_y

        if max_rel_error_x > rel_error_tol:
            plt.imshow(disp_x_from_phase)
            plt.show()
            self.fail("Maximum error was %f" % max_rel_error_x)
        if max_rel_error_y > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_y)

    def test_sine_small_disp(self):
        relative_error = 5e-3
        oversampling = 9

        grid_pitch = 5

        disp_amp = 0.01
        disp_period = 400
        disp_n_periodes = 1

        xs, ys, xs_disp, ys_disp, u_x, u_y = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                                 formulation="lagrangian")

        grid_undeformed = make_grid(xs, ys, grid_pitch, oversampling=oversampling)

        grid_displaced_eulr = make_grid(xs_disp, ys_disp, grid_pitch, oversampling=oversampling)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_from_phase(phase_x, phase_x0, phase_y, phase_y0, grid_pitch,
                                                               correct_phase=False)

        u_x = u_x[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]
        u_y = u_y[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]

        rms_error_u_x = rms_diff(u_x, disp_x_from_phase)
        rms_error_u_y = rms_diff(u_y, disp_y_from_phase)

        if rms_error_u_x / disp_amp > relative_error:
            self.fail("RMS error in u_x is %f" % (rms_error_u_x / disp_amp))
        if rms_error_u_y / disp_amp > relative_error:
            self.fail("RMS error in u_x is %f" % (rms_error_u_y / disp_amp))

    def test_sine_large_disp(self):
        relative_error = 5e-3

        grid_pitch = 5
        oversampling = 15

        disp_amp = 3.57
        disp_period = 400
        disp_n_periodes = 1

        xs, ys, xs_disp, ys_disp, u_x, u_y = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                                 formulation="lagrangian")

        grid_undeformed = make_grid(xs, ys, grid_pitch, oversampling=oversampling, pixel_size=1)

        grid_displaced_eulr = make_grid(xs_disp, ys_disp, grid_pitch, oversampling=oversampling,
                                               pixel_size=1)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_from_phase(phase_x, phase_x0, phase_y, phase_y0, grid_pitch,
                                                               correct_phase=True)

        u_x = u_x[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]
        u_y = u_y[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]

        # plt.imshow(disp_x_from_phase)
        # plt.show()
        # plt.imshow(u_x)
        # plt.show()

        rms_error_u_x = rms_diff(u_x, disp_x_from_phase)
        rms_error_u_y = rms_diff(u_y, disp_y_from_phase)

        if rms_error_u_x / disp_amp > relative_error:
            self.fail("RMS error in u_x is %f" % (rms_error_u_x / disp_amp))
        if rms_error_u_y / disp_amp > relative_error:
            self.fail("RMS error in u_x is %f" % (rms_error_u_y / disp_amp))

    def test_half_sine_large_disp(self):
        acceptable_amp_loss = 0.1

        grid_pitch = 5
        oversampling = 9

        disp_amp = 1.7
        disp_period = 500
        disp_n_periodes = 0.5

        xs, ys, xs_disp, ys_disp,_,_ = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes, formulation="lagrangian")

        grid_undeformed = make_grid(xs, ys, grid_pitch, oversampling=oversampling, pixel_size=1)

        grid_displaced_eulr = make_grid(xs_disp, ys_disp, grid_pitch, oversampling=oversampling,
                                               pixel_size=1)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_from_phase(phase_x, phase_x0, phase_y, phase_y0, grid_pitch,
                                                               correct_phase=True)

        peak_disp_x = np.max(np.abs(disp_x_from_phase))
        peak_disp_y = np.max(np.abs(disp_y_from_phase))

        if np.abs((peak_disp_x / disp_amp) - 1.) > acceptable_amp_loss:
            self.fail("Amplitude loss of  %f" % (1. - peak_disp_x / disp_amp))
        if np.abs((peak_disp_y / disp_amp) - 1.) > acceptable_amp_loss:
            self.fail("Amplitude loss of %f" % (1. - peak_disp_x / disp_amp))


