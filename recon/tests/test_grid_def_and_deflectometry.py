from recon.deflectomerty import disp_from_grids, angle_from_disp
from recon.artificial_grid_deformation import deform_grid_from_deflection
from recon.slope_integration import disp_from_slopes
from unittest import TestCase
import numpy as np


class Test_DeformGridNoise(TestCase):

    def test_noise_with_different_std(self):
        rel_tol = 0.05

        pixel_size = 1
        mirror_grid_dist = 500.
        grid_pitch = 5
        upscale = 1
        oversampling = 5

        n_pts_x = 100
        n_pts_y = 100
        n_imgs = 50

        noise_stds = np.arange(0.001,0.01,0.001)

        for noise_std in noise_stds:
            undeformed_field = np.zeros((n_pts_x,n_pts_y))

            grids = [deform_grid_from_deflection(undeformed_field,pixel_size,mirror_grid_dist,grid_pitch,upscale,oversampling,img_noise_std=noise_std) for _ in range(n_imgs)]
            noise_meas_std = np.mean(np.std(grids,axis=0))
            norm_noise_std = noise_meas_std/(np.max(grids)-np.min(grids))

            if np.abs(norm_noise_std-noise_std)/noise_std > rel_tol:
                self.fail("The noise level is not correct. Correct is %f, estimated is %f"%(noise_std,norm_noise_std))





class Test_DeformGridAndRunDeflectometry(TestCase):

    def test_half_sine_deflection_no_upscale(self):
        rel_tol = 0.01

        pixel_size = 1
        mirror_grid_dist = 500.
        grid_pitch = 5
        upscale = 1
        oversampling = 5

        deflection_amp = 0.1
        n_pts_x = 200
        n_pts_y = 200

        xs,ys = np.meshgrid(np.linspace(0,1,n_pts_x),np.linspace(0,1,n_pts_y))
        deflection_field = deflection_amp * np.sin(np.pi*xs)*np.sin(np.pi*ys)
        undeformed_field = np.zeros_like(deflection_field)

        undeformed_grid = deform_grid_from_deflection(undeformed_field,pixel_size,mirror_grid_dist,grid_pitch,upscale,oversampling)
        deformed_grid = deform_grid_from_deflection(deflection_field,pixel_size,mirror_grid_dist,grid_pitch,upscale,oversampling)

        disp_x,disp_y = disp_from_grids(undeformed_grid,deformed_grid,grid_pitch,correct_phase=True)
        angle_x = angle_from_disp(disp_x,mirror_grid_dist)
        angle_y = angle_from_disp(disp_y,mirror_grid_dist)

        # Add axis
        angle_x = angle_x[np.newaxis,:,:]
        angle_y = angle_y[np.newaxis,:,:]

        reconstucted_defl = disp_from_slopes(angle_x,angle_y,pixel_size,zero_at="bottom corners")
        # As a reduced field of view causes the a shift of the whole field, a manual correction is performed

        reconstucted_defl = reconstucted_defl + deflection_field[4*grid_pitch,4*grid_pitch]
        cropped_deflection = deflection_field[4*grid_pitch:-4*grid_pitch,4*grid_pitch:-4*grid_pitch]

        rel_peak_error = np.max(np.abs(reconstucted_defl[0,:,:]-cropped_deflection))/deflection_amp

        if rel_peak_error > rel_tol:
            self.fail("The peak error of %f is larger than the tolerance of %f"%(rel_peak_error,rel_tol))



