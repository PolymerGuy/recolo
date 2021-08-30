# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import recon
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
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
plate = recon.make_plate(mat_E, mat_nu, density, plate_thick)

mirror_grid_dist = 500.

# crops = np.arange(0,5)
crops = [0]
peak_presses = []
for crop in crops:

    abq_sim_fields = recon.load_abaqus_rpts("/home/sindreno/Rene/testfolder/fields/")

    upscale = 8

    pixel_size = abq_sim_fields.pixel_size_x / upscale

    disp_fields = abq_sim_fields.disp_fields

    if crop > 0:
        disp_fields = disp_fields[:, crop:-crop, crop:-crop]

    # pressure reconstruction parameters
    win_size = 30
    sampling_rate = 1. / (abq_sim_fields.times[1] - abq_sim_fields.times[0])

    # Load slope fields and calculate displacement fields
    grid_pitch = 5.  # pixels

    # Deflectometry
    undeformed_grid = recon.artificial_grid_deformation.deform_grid_from_deflection(disp_fields[0, :, :], abq_sim_fields.pixel_size_x, mirror_grid_dist,
                                                                                    grid_pitch,
                                                                                    img_upscale=upscale)
    sloppes_x = []
    sloppes_y = []
    for disp_field in disp_fields:
        deformed_grid = recon.artificial_grid_deformation.deform_grid_from_deflection(disp_field, abq_sim_fields.pixel_size_x, mirror_grid_dist,
                                                                                      grid_pitch,
                                                                                      img_upscale=upscale)
        slopes_x, slopes_y = recon.deflectomerty.disp_from_grids(undeformed_grid, deformed_grid, grid_pitch)
        sloppes_x.append(slopes_x)
        sloppes_y.append(slopes_y)

    slopes_x = np.array(sloppes_x)
    slopes_y = np.array(sloppes_y)

    # Integrate slopes from deflectometry to get deflection fields
    disp_fields = recon.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size,
                                                           zero_at="bottom corners", zero_at_size=5,
                                                           extrapolate_edge=0, filter_sigma=0, downsample=1)

    # Kinematic fields from deflection field
    fields = recon.kinematic_fields_from_deflections(disp_fields, pixel_size, sampling_rate,
                                                     filter_time_sigma=2,
                                                     filter_space_sigma=0)

    virtual_field = recon.virtual_fields.Hermite16(win_size, pixel_size)

    # Results are stored in these lists
    times = []
    pressure_fields = []

    for i, field in enumerate(fields):
        print("Processing frame %i" % i)
        recon_press = recon.solver_VFM.pressure_elastic_thin_plate(field, plate, virtual_field)

        # Store results
        pressure_fields.append(recon_press)
        times.append(field.time)

    pressure_fields = np.array(pressure_fields)
    center = int(pressure_fields.shape[1] / 2)
    peak_presses.append(np.max(pressure_fields[:, center, center]))
    # Plot the results
    plt.plot((np.array(times)) * 1000., pressure_fields[:, center, center], '-',
             label="Crop factor=%f" % (crop * upscale / disp_fields.shape[-1]))

real_press, real_time = read_exp_press_data()

plt.plot(real_time[:] * 1000., real_press[:, 8] * 1.e6, '--', label="Transducer", alpha=0.7)

plt.plot(real_time * 1000, gaussian_filter(real_press[:, 8] * 1.e6, sigma=2. * 500. / 75.), '--',
         label="We should get this curve for sigma=2")

plt.xlim(left=0.000, right=0.9)
plt.ylim(top=80000, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
