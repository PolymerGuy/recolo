from recon.deflectomerty import slopes_from_images
from recon.slope_integration import disp_from_slopes
from unittest import TestCase
import numpy as np


class Test_DeflectometryOnImages(TestCase):

    def test_half_sine_deflection_no_upscale(self):
        rel_tol = 0.01
        path_to_imgs = "./ExampleGridImages/"

        pixel_size = 1
        mirror_grid_dist = 500.
        grid_pitch = 5

        deflection_amp = 0.1
        n_pts_x = 400
        n_pts_y = 400

        xs, ys = np.meshgrid(np.linspace(0, 1, n_pts_x), np.linspace(0, 1, n_pts_y))
        deflection_field = deflection_amp * np.sin(np.pi * xs) * np.sin(np.pi * ys)

        angle_x, angle_y = slopes_from_images(path_to_imgs, grid_pitch, mirror_grid_dist)

        reconstucted_defl = disp_from_slopes(angle_x, angle_y, pixel_size, zero_at="bottom corners")

        # As a reduced field of view causes the a shift of the whole field, a manual correction is performed
        reconstucted_defl = reconstucted_defl + deflection_field[4 * grid_pitch, 4 * grid_pitch]
        cropped_deflection = deflection_field[4 * grid_pitch:-4 * grid_pitch, 4 * grid_pitch:-4 * grid_pitch]

        rel_peak_error = np.max(np.abs(reconstucted_defl[1, :, :] - cropped_deflection)) / deflection_amp

        if rel_peak_error > rel_tol:
            self.fail("The peak error of %f is larger than the tolerance of %f" % (rel_peak_error, rel_tol))
