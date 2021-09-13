# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath

sys.path.extend([abspath(".")])

import recon
import numpy as np
import matplotlib.pyplot as plt
import os

cwd = os.path.dirname(os.path.realpath(__file__))

# Minimal example of pressure load reconstruction based on input from Abaqus. The deflection field from Abaqus is
# used to generate images used for deflectometry. The images used for deflectometry has to be interpolated from the
# displacement fields to produce a high resolution grid image.
#
# This example looks at the effect of limited field of view. This is done by cropping the displacement fields from
# Abaqus before generation of the grid images.
#
# Minor deviations from the correct pressure field occurs due to the use of central differences to determine the acceleration fields,
# introduction significant errors when the temporal resolution is as low as here.


# Crop the displacement field by removing a frame around the field with a given width.
remove_n_pixels = [0, 1, 2, 4]

# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.33  # Poisson's ratio []
density = 7700
plate_thick = 5e-3
plate = recon.make_plate(mat_E, mat_nu, density, plate_thick)

# Reconstruction settings
win_size = 30

# Deflectometry settings
abq_to_img_scale = 8
mirror_grid_dist = 500.
grid_pitch = 5.  # pixels

for remove_pixel in remove_n_pixels:
    # Load Abaqus data
    abq_sim_fields = recon.load_abaqus_rpts(os.path.join(cwd, "AbaqusExampleData/"))
    if remove_pixel == 0:
        cropped_disp_field = abq_sim_fields.disp_fields
    else:
        # Remove the data points
        cropped_disp_field = abq_sim_fields.disp_fields[:, remove_pixel:-remove_pixel, remove_pixel:-remove_pixel]

    # The deflectometry return the slopes of the plate which has to be integrated in order to determine the deflection
    slopes_x = []
    slopes_y = []
    undeformed_grid = recon.artificial_grid_deformation.deform_grid_from_deflection(cropped_disp_field[0, :, :],
                                                                                    pixel_size=abq_sim_fields.pixel_size_x,
                                                                                    mirror_grid_dist=mirror_grid_dist,
                                                                                    grid_pitch=grid_pitch,
                                                                                    img_upscale=abq_to_img_scale)
    for disp_field in cropped_disp_field:
        deformed_grid = recon.artificial_grid_deformation.deform_grid_from_deflection(disp_field,
                                                                                      pixel_size=abq_sim_fields.pixel_size_x,
                                                                                      mirror_grid_dist=mirror_grid_dist,
                                                                                      grid_pitch=grid_pitch,
                                                                                      img_upscale=abq_to_img_scale)

        disp_x, disp_y = recon.deflectomerty.disp_from_grids(undeformed_grid, deformed_grid, grid_pitch)
        slope_x = recon.deflectomerty.angle_from_disp(disp_x, mirror_grid_dist)
        slope_y = recon.deflectomerty.angle_from_disp(disp_y, mirror_grid_dist)
        slopes_x.append(slope_x)
        slopes_y.append(slope_y)

    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)
    pixel_size = abq_sim_fields.pixel_size_x / abq_to_img_scale

    # Integrate slopes to get deflection fields
    disp_fields = recon.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size,
                                                           zero_at="bottom corners", zero_at_size=1,
                                                           extrapolate_edge=0, filter_sigma=0, downsample=1)

    # Kinematic fields from deflection field
    kin_fields = recon.kinematic_fields_from_deflections(disp_fields,
                                                         pixel_size=pixel_size,
                                                         sampling_rate=abq_sim_fields.sampling_rate)

    # Reconstruct pressure using the virtual fields method
    virtual_field = recon.virtual_fields.Hermite16(win_size, pixel_size)
    pressure_fields = np.array(
        [recon.solver_VFM.calc_pressure_thin_elastic_plate(field, plate, virtual_field) for field in kin_fields])

    # Reconstructed pressure from VFM
    center = int(pressure_fields.shape[1] / 2)
    plt.plot(abq_sim_fields.times * 1000., pressure_fields[:, center, center], "-o",
             label="Cropped by %i pixels" % remove_pixel)

# Plot the results
# Correct pressure history used in the Abaqus simulation
times = np.array([0.0, 0.00005, 0.00010, 0.0003, 0.001]) * 1000
pressures = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) * 1e5
plt.plot(times, pressures, '-', label="Correct pressure")

plt.xlim(left=0.000, right=0.3)
plt.ylim(top=110000, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
