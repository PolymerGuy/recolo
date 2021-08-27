from recon.deflectomerty import disp_from_grids
from recon.artificial_grid_deformation import harmonic_disp_field, dotted_grid
from unittest import TestCase
import numpy as np
from skimage.restoration import estimate_sigma


def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2) ** 2.))


class Test_NoiseAddition(TestCase):

    def test_additive_gaussian_noise(self):

        # Relatively slack tolerances as there is an inherent uncertainty in the the estimate of the noise level.
        rel_error_tol = 5e-2

        # Large pitches to help the noise estimation algorithm
        grid_pitch = 20
        n_pitches = 20

        # 1%, 5% and 10% noise
        noise_levels = [0.01,0.05,0.1]

        x = np.arange(grid_pitch * n_pitches, dtype=float)
        y = np.arange(grid_pitch * n_pitches, dtype=float)

        xs, ys = np.meshgrid(x, y)

        for noise_level in noise_levels:
            grid_with_noise = dotted_grid(xs, ys, grid_pitch,noise_std=noise_level)
            estimated_sigma = estimate_sigma(grid_with_noise)

            # Normalize to the amplitude of two
            estimated_noise_level = estimated_sigma/2.

            if np.abs(estimated_noise_level-noise_level)/noise_level> rel_error_tol:
                self.fail("Correct noise amplitude was %f but the estimated was %f"%(noise_level,estimated_noise_level))



class Test_DeflectometryOnKnownDeformations(TestCase):

    def test_rigid_body_motion_small_disp(self):

        rel_error_tol = 1e-3

        grid_pitch = 5
        n_pitches = 20

        displacement_x = 1.e-2
        displacement_y = 1.e-2

        x = np.arange(grid_pitch * n_pitches, dtype=float)
        y = np.arange(grid_pitch * n_pitches, dtype=float)

        xs, ys = np.meshgrid(x, y)

        grid_undeformed = dotted_grid(xs, ys, grid_pitch)

        # x = X + u(X) can be solved directly as u(X) is a constant
        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_deformed = dotted_grid(xs_disp, ys_disp, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_from_grids(grid_undeformed,grid_deformed,grid_pitch,correct_phase=True)

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

        grid_undeformed = dotted_grid(xs, ys, grid_pitch)

        # x = X + u(X) can be solved directly as u(X) is a constant
        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_deformed = dotted_grid(xs_disp, ys_disp, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_from_grids(grid_undeformed,grid_deformed,grid_pitch,correct_phase=True)


        max_rel_error_x = np.max(np.abs(disp_x_from_phase - displacement_x)) / displacement_x
        max_rel_error_y = np.max(np.abs(disp_y_from_phase - displacement_y)) / displacement_y

        if max_rel_error_x > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_x)
        if max_rel_error_y > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_y)

    def test_sine_small_disp(self):
        relative_error = 1e-2
        oversampling = 9

        grid_pitch = 5

        disp_amp = 0.01
        disp_period = 400
        disp_n_periodes = 1

        x_undef, y_undef, Xs, Ys, u_X, u_Y = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                                 formulation="lagrangian")

        grid_undeformed = dotted_grid(x_undef, y_undef, grid_pitch, oversampling=oversampling)

        grid_deformed = dotted_grid(Xs, Ys, grid_pitch, oversampling=oversampling)

        disp_x_from_phase, disp_y_from_phase = disp_from_grids(grid_undeformed, grid_deformed, grid_pitch,
                                                               correct_phase=True)

        u_X = u_X[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]
        u_Y = u_Y[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]

        rms_error_u_x = rms_diff(u_X, disp_x_from_phase)
        rms_error_u_y = rms_diff(u_Y, disp_y_from_phase)

        if rms_error_u_x / disp_amp > relative_error:
            self.fail("RMS error in u_x is %f" % (rms_error_u_x / disp_amp))
        if rms_error_u_y / disp_amp > relative_error:
            self.fail("RMS error in u_x is %f" % (rms_error_u_y / disp_amp))

    def test_sine_large_disp(self):
        relative_error = 1e-2

        grid_pitch = 5
        oversampling = 9

        disp_amp = 1.57
        disp_period = 400
        disp_n_periodes = 1

        xs_undef, ys_undef, Xs, Ys, u_X, u_Y = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                                 formulation="lagrangian")

        grid_undeformed = dotted_grid(xs_undef, ys_undef, grid_pitch, oversampling=oversampling, pixel_size=1)

        grid_deformed = dotted_grid(Xs, Ys, grid_pitch, oversampling=oversampling,
                                    pixel_size=1)

        disp_x_from_phase, disp_y_from_phase = disp_from_grids(grid_undeformed,grid_deformed,grid_pitch,correct_phase=True)


        u_X = u_X[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]
        u_Y = u_Y[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]

        rms_error_u_x = rms_diff(u_X, disp_x_from_phase)
        rms_error_u_y = rms_diff(u_Y, disp_y_from_phase)

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

        xs_undef, ys_undef, Xs, Ys, _, _ = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                             formulation="lagrangian")

        grid_undeformed = dotted_grid(xs_undef, ys_undef, grid_pitch, oversampling=oversampling, pixel_size=1)

        grid_deformed = dotted_grid(Xs, Ys, grid_pitch, oversampling=oversampling,
                                    pixel_size=1)

        disp_x_from_phase, disp_y_from_phase = disp_from_grids(grid_undeformed,grid_deformed,grid_pitch,correct_phase=True)


        peak_disp_x = np.max(np.abs(disp_x_from_phase))
        peak_disp_y = np.max(np.abs(disp_y_from_phase))

        if np.abs((peak_disp_x / disp_amp) - 1.) > acceptable_amp_loss:
            self.fail("Amplitude loss of  %f" % (1. - peak_disp_x / disp_amp))
        if np.abs((peak_disp_y / disp_amp) - 1.) > acceptable_amp_loss:
            self.fail("Amplitude loss of %f" % (1. - peak_disp_x / disp_amp))
