# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath

sys.path.extend([abspath("../AbaqusExperiments")])

import recolo
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('science')


def read_exp_press_data():
    start = 25550 + 38
    end = 26200

    data = np.genfromtxt("/home/sindreno/Downloads/Rene/Valid_0.125_3.txt", skip_header=20)

    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, :] / 10.
    return press - press[0, :], time


# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.33  # Poisson's ratio []
density = 7700
plate_thick = 5e-3
plate = recolo.make_plate(mat_E, mat_nu, density, plate_thick)

# Image noise
noise_std = 0.008*0

# recolostruction settings
win_size = 30  # Should be increased when deflectometry is used

# Deflectometry settings
upscale = 8
mirror_grid_dist = 1.385
grid_pitch = 5.  # pixels

crops = [10]
pressure_fields_for_crops = []

for crop in crops:
    # Load Abaqus data
    abq_sim_fields = recolo.load_abaqus_rpts("/home/sindreno/Rene/testfolder/fields/")
    print("Field shape is",abq_sim_fields.disp_fields.shape)
    #if crop>0:
    #    cropped_disp_field = abq_sim_fields.disp_fields[:,crop:-crop,crop:-crop]
    #else:
    cropped_disp_field = abq_sim_fields.disp_fields
    # The deflectometry return the slopes of the plate which has to be integrated in order to determine the deflection
    slopes_x = []
    slopes_y = []
    undeformed_grid = recolo.artificial_grid_deformation.deform_grid_from_deflection(cropped_disp_field[0, :, :],
                                                                                    abq_sim_fields.pixel_size_x,
                                                                                    mirror_grid_dist,
                                                                                    grid_pitch,
                                                                                    img_upscale=upscale,
                                                                                    img_noise_std=0)
    if crop>0:
        undeformed_grid = undeformed_grid[crop:-crop,crop:-crop]

    for disp_field in cropped_disp_field:
        deformed_grid = recolo.artificial_grid_deformation.deform_grid_from_deflection(disp_field,
                                                                                      abq_sim_fields.pixel_size_x,
                                                                                      mirror_grid_dist,
                                                                                      grid_pitch,
                                                                                      img_upscale=upscale,
                                                                                      img_noise_std=noise_std)

        if crop>0:
            deformed_grid = deformed_grid[crop:-crop, crop:-crop]

        disp_x, disp_y = recolo.deflectomerty.disp_from_grids(undeformed_grid, deformed_grid, grid_pitch,window="triangular")
        slope_x = recolo.deflectomerty.angle_from_disp(disp_x, mirror_grid_dist)
        slope_y = recolo.deflectomerty.angle_from_disp(disp_y, mirror_grid_dist)
        slopes_x.append(slope_x)
        slopes_y.append(slope_y)

    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)
    pixel_size = abq_sim_fields.pixel_size_x / upscale


    # Integrate slopes to get deflection fields
    disp_fields = recolo.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size,
                                                           zero_at="bottom corners", zero_at_size=10,
                                                           extrapolate_edge=0, downsample=1)

    # Kinematic fields from deflection field
    kin_fields = recolo.kinematic_fields_from_deflections(disp_fields, pixel_size,
                                                         abq_sim_fields.sampling_rate, filter_space_sigma=20,
                                                         filter_time_sigma=2)

    # recolostruct pressure using the virtual fields method
    virtual_field = recolo.virtual_fields.Hermite16(win_size, pixel_size)
    pressure_fields = np.array(
        [recolo.solver_VFM.calc_pressure_thin_elastic_plate(field, plate, virtual_field) for field in kin_fields])

    # recolostructed
    center = int(pressure_fields.shape[1] / 2)
    plt.plot(abq_sim_fields.times * 1000., pressure_fields[:, center, center], "-o", label="recolostruction, cropped by %i pixels"%(crop))
    pressure_fields_for_crops.append(pressure_fields)
# Plot the results
# Correct
pressures, times = read_exp_press_data()
plt.plot(times * 1e3, pressures[:, 8] * 1e6, '-', label="Correct pressure")


plt.xlim(left=0.000, right=0.6)
plt.ylim(top=110000, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
