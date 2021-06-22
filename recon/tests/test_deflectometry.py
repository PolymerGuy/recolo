from recon.deflectomerty.deflectometry import detect_phase, disp_from_phase

from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
import muDIC as dic
from skimage.restoration import unwrap_phase


def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2)) ** 2.)


def grid_grey_scales(xs, ys, pitch):
    return np.cos(2. * np.pi * xs / float(pitch)) * np.cos(2. * np.pi * ys / float(pitch))


def grid_grey_scales_oversampling(xs, ys, pitch,oversampling=5):
    return np.cos(2. * np.pi * xs / float(pitch)) * np.cos(2. * np.pi * ys / float(pitch))





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

        grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch)
        disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch)

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

        displacement_x = 0.5
        displacement_y = 0.5

        x = np.arange(grid_pitch * n_pitches, dtype=float)
        y = np.arange(grid_pitch * n_pitches, dtype=float)

        xs, ys = np.meshgrid(x, y)

        grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch)


        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch,unwrap=True)
        disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch,unwrap=True)

        max_rel_error_x = np.max(np.abs(disp_x_from_phase - displacement_x)) / displacement_x
        max_rel_error_y = np.max(np.abs(disp_y_from_phase - displacement_y)) / displacement_y

        if max_rel_error_x > rel_error_tol:
            plt.imshow(disp_x_from_phase)
            plt.show()
            self.fail("Maximum error was %f" % max_rel_error_x)
        if max_rel_error_y > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_y)

    def test_sine_small_disp(self):
        acceptable_amp_loss = 0.1

        grid_pitch = 5

        disp_amp = 0.01
        disp_period = 100
        disp_n_periodes = 2

        x = np.arange(disp_n_periodes * disp_period, dtype=float)
        y = np.arange(disp_n_periodes * disp_period, dtype=float)

        xs, ys = np.meshgrid(x, y)

        displacement_x = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * xs / xs.max())
        displacement_y = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * ys / ys.max())

        grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch, small_disp=True, axis=0)
        disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch, small_disp=True, axis=1)

        peak_disp_x = np.max(np.abs(disp_x_from_phase))
        peak_disp_y = np.max(np.abs(disp_y_from_phase))

        if np.abs((peak_disp_x/disp_amp)-1.) > acceptable_amp_loss:
            self.fail("Amplitude loss of  %f" % (1. - peak_disp_x / disp_amp))
        if np.abs((peak_disp_y/disp_amp)-1.) > acceptable_amp_loss:
            self.fail("Amplitude loss of %f" % (1. - peak_disp_x / disp_amp))


    def test_sine_large_disp(self):
        acceptable_amp_loss = 0.1

        grid_pitch = 5

        disp_amp = 0.5
        disp_period = 100
        disp_n_periodes = 2

        x = np.arange(disp_n_periodes * disp_period, dtype=float)
        y = np.arange(disp_n_periodes * disp_period, dtype=float)

        xs, ys = np.meshgrid(x, y)

        displacement_x = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * xs / xs.max())
        displacement_y = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * ys / ys.max())

        grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch, small_disp=True, axis=0,unwrap=True)
        disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch, small_disp=True, axis=1,unwrap=True)

        plt.imshow(disp_x_from_phase)
        plt.show()

        peak_disp_x = np.max(np.abs(disp_x_from_phase))
        peak_disp_y = np.max(np.abs(disp_y_from_phase))

        if np.abs((peak_disp_x/disp_amp)-1.) > acceptable_amp_loss:
            self.fail("Amplitude loss of  %f" % (1. - peak_disp_x / disp_amp))
        if np.abs((peak_disp_y/disp_amp)-1.) > acceptable_amp_loss:
            self.fail("Amplitude loss of %f" % (1. - peak_disp_x / disp_amp))

    def test_half_sine_disp_sweep(self):

        grid_pitches = [5, 7]
        for grid_pitch in grid_pitches:
            rel_error_tol = 1e-3
            peak_disp_x = []
            peak_disp_y = []
            disp_periodes = np.arange(10, 200, 10)
            for disp_period in disp_periodes:
                disp_amp = 0.01
                disp_n_periodes = 2

                x = np.arange(disp_n_periodes * disp_period, dtype=float)
                y = np.arange(disp_n_periodes * disp_period, dtype=float)

                xs, ys = np.meshgrid(x, y)

                displacement_x = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * xs / xs.max())
                displacement_y = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * ys / ys.max())

                grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

                xs_disp = xs - displacement_x
                ys_disp = ys - displacement_y

                grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch)

                phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
                phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

                disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch, small_disp=True, axis=0)
                disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch, small_disp=True, axis=1)

                peak_disp_x.append(np.max(disp_x_from_phase))
                peak_disp_y.append(np.max(disp_y_from_phase))

            peak_disp_x = np.array(peak_disp_x)
            peak_disp_y = np.array(peak_disp_y)

            plt.plot(disp_periodes, peak_disp_x / disp_amp, label="Grid pitch: %i" % grid_pitch)
            plt.xlabel("Displacement periode [pix]")
            plt.ylabel("Normalized displacement amplitude [-]")

        plt.hlines(0.9, np.min(disp_periodes), np.max(disp_periodes), linestyles="--")
        plt.legend()
        plt.show()

    def test_half_sine_disp_sweepX(self):
        # Tests that the performance is OK in terms of bandwidth.
        grid_pitches = [5, 7]
        for grid_pitch in grid_pitches:
            rel_error_tol = 1e-3
            peak_disp_x = []
            peak_disp_y = []
            disp_periodes = np.arange(10, 200, 10)
            for disp_periode in disp_periodes:

                disp_amp = 0.01
                disp_n_periodes = 2

                x = np.arange(disp_n_periodes * disp_periode, dtype=float)
                y = np.arange(disp_n_periodes * disp_periode, dtype=float)

                xs, ys = np.meshgrid(x, y)
                Xs, Ys = np.meshgrid(x, y)

                displacement_x = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * xs / xs.max())
                displacement_y = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * ys / ys.max())

                tol = 1.e-12
                for i in range(20):
                    Xs = Xs - (xs - Xs - disp_amp * np.sin(disp_n_periodes * np.pi * Xs / xs.max())) / (
                            -1 - disp_amp * np.cos(
                        disp_n_periodes * np.pi * Xs / xs.max()) / xs.max() / disp_n_periodes)
                    Ys = Ys - (ys - Ys - disp_amp * np.sin(disp_n_periodes * np.pi * Ys / ys.max())) / (
                            -1 - disp_amp * np.cos(
                        disp_n_periodes * np.pi * Ys / ys.max()) / ys.max() / disp_n_periodes)

                    errors_x = np.max(np.abs(Xs + disp_amp * np.sin(disp_n_periodes * np.pi * Xs / xs.max()) - xs))
                    errors_y = np.max(np.abs(Ys + disp_amp * np.sin(disp_n_periodes * np.pi * Ys / xs.max()) - ys))
                    # print(errors_x)
                    # print(errors_y)

                    if errors_x < tol and errors_y < tol:
                        print("Coordinate correction converged in %i iterations" % i)
                        break

                grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

                xs_disp = xs - displacement_x
                ys_disp = ys - displacement_y

                # xs_disp = Xs
                # ys_disp = Ys

                grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch)

                phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
                phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

                disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch, small_disp=True, axis=0)
                disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch, small_disp=True, axis=1)

                peak_disp_x.append(np.max(disp_x_from_phase))
                peak_disp_y.append(np.max(disp_y_from_phase))

            peak_disp_x = np.array(peak_disp_x)
            peak_disp_y = np.array(peak_disp_y)

            plt.plot(disp_periodes, peak_disp_x / disp_amp, label="Grid pitch: %i" % grid_pitch)
            plt.xlabel("Displacement periode [pix]")
            plt.ylabel("Normalized displacement amplitude [-]")

        plt.hlines(0.9, np.min(disp_periodes), np.max(disp_periodes), linestyles="--")
        plt.legend()
        plt.show()

    def test_half_sine_disp_muDIC(self):

        rel_error_tol = 1e-3

        grid_pitch = 5

        disp_amp = 0.5
        disp_period = 100
        disp_n_periodes = 2

        x = np.arange(disp_n_periodes * disp_period, dtype=float)
        y = np.arange(disp_n_periodes * disp_period, dtype=float)

        xs, ys = np.meshgrid(x, y)

        displacement_x = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * xs / xs.max())
        displacement_y = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * ys / ys.max())

        grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch)


        plt.imshow(grid_displaced_eulr)
        plt.show()

        imgs = dic.image_stack_from_list([grid_undeformed, grid_displaced_eulr])
        mesher = dic.Mesher()
        mesh = mesher.mesh(imgs, Xc1=10, Xc2=190, Yc1=10, Yc2=190, n_elx=20, n_ely=20, GUI=False)
        input = dic.DICInput(mesh, imgs)
        anal = dic.DICAnalysis(input)
        res = anal.run()
        fields = dic.Fields(res)

        print("sdfsdfsdfsdf", fields.disp().shape)
        plt.imshow(fields.disp()[0, 1, :, :, -1])
        plt.show()

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch, small_disp=False, axis=0)
        disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch, small_disp=False, axis=1)

        plt.imshow(disp_x_from_phase)
        plt.show()

        max_rel_error_x = np.max(np.abs(disp_x_from_phase - displacement_x[4 * grid_pitch:-4 * grid_pitch,
                                                            4 * grid_pitch:-4 * grid_pitch]) / displacement_x[
                                                                                               4 * grid_pitch:-4 * grid_pitch,
                                                                                               4 * grid_pitch:-4 * grid_pitch])
        max_rel_error_y = np.max(np.abs(disp_y_from_phase - displacement_y[4 * grid_pitch:-4 * grid_pitch,
                                                            4 * grid_pitch:-4 * grid_pitch]) / displacement_y[
                                                                                               4 * grid_pitch:-4 * grid_pitch,
                                                                                               4 * grid_pitch:-4 * grid_pitch])

        if max_rel_error_x > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_x)
        if max_rel_error_y > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_y)

    def test_grid_pitch_insensitivity(self):

        # The grid pitch on the actual image is not exactly known and we need to make sure that
        # the phase detection works even when we don't hit the right value exactly.

        rel_error_tol = 1e-3

        grid_pitch_assumed = 5.0
        grid_pitch_real = 5.08
        n_pitches = 20

        displacement_x = 1.e-2
        displacement_y = 1.e-2

        x = np.arange(grid_pitch_assumed * n_pitches, dtype=float)
        y = np.arange(grid_pitch_assumed * n_pitches, dtype=float)

        xs, ys = np.meshgrid(x, y)

        grid_undeformed = grid_grey_scales(xs, ys, grid_pitch_real)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = grid_grey_scales(xs_disp, ys_disp, grid_pitch_real)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch_assumed)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch_assumed)

        disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch_assumed)
        disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch_assumed)

        plt.imshow(disp_x_from_phase)
        plt.show()

        max_rel_error_x = np.max(np.abs(disp_x_from_phase - displacement_x)) / displacement_x
        max_rel_error_y = np.max(np.abs(disp_y_from_phase - displacement_y)) / displacement_y

        if max_rel_error_x > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_x)
        if max_rel_error_y > rel_error_tol:
            self.fail("Maximum error was %f" % max_rel_error_y)
