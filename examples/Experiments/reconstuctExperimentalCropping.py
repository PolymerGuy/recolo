# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath

sys.path.extend([abspath(".")])

import recolo
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('science')

# plate and model parameters
mat_E = 190.e9  # Young's modulus [Pa]
mat_nu = 0.3  # Poisson's ratio []
plate_thick = 4.95e-3
rho = 7934.  # Density [kg/m^3]

plate = recolo.make_plate(mat_E, mat_nu, rho, plate_thick)

# pressure reconstruction parameters
win_size = 20
sampling_rate = 75000.
downsampling_factor = 5
filter_time_sigma = 6
filter_space_sigma = 2

# Load slope fields and calculate displacement fields
exp_data = recolo.demoData.ImpactHammerExperiment()

ref_img_ids = list(range(30))  # We use the first 30 images to produce a low-noise reference
use_imgs = range(30, 280)  # Skip the first 30 images

# Grid description
grid_pitch = 5.0  # pixels
grid_pitch_len = 5.88433e-3  # m

# Deflectometry setup
mirror_grid_distance = 1.385  # m

# Calculations stars below
pixel_size_on_grid_plane = grid_pitch_len / grid_pitch
pixel_size_on_mirror = 1.02189781 * (grid_pitch_len / grid_pitch) * 0.5

slopes_y, slopes_x = recolo.deflectomerty.slopes_from_images(exp_data.path_to_img_folder, grid_pitch,
                                                             mirror_grid_distance, pixel_size_on_grid_plane,
                                                             ref_img_ids=ref_img_ids,
                                                             only_img_ids=use_imgs,
                                                             crop=(76, -35, 10, -10), window="triangular",
                                                             correct_phase=False)

disp_fields = recolo.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size_on_mirror,
                                                        zero_at="bottom corners", zero_at_size=5,
                                                        filter_sigma=filter_space_sigma, downsample=downsampling_factor)

# Results are stored in these lists
times = []
presses = []

kin_fields = recolo.kinematic_fields_from_deflections(disp_fields, downsampling_factor * pixel_size_on_mirror, sampling_rate,
                                                      filter_time_sigma=filter_time_sigma)
virtual_field = recolo.virtual_fields.Hermite16(win_size, downsampling_factor * pixel_size_on_mirror)

for i, field in enumerate(kin_fields):
    print("Processing frame %i" % i)
    recon_press = recolo.solver_VFM.calc_pressure_thin_elastic_plate(field, plate, virtual_field)
    presses.append(recon_press)
    times.append(field.time)

presses = np.array(presses)
center = int(presses.shape[1] / 2)

# Load impact hammer data
hammer_force, hammer_time = exp_data.hammer_data()

# Plot the results
plt.figure(figsize=(7,5))
plt.plot(times, np.sum(presses, axis=(1, 2)) * ((pixel_size_on_mirror * downsampling_factor) ** 2.), label="VFM force from whole plate")
plt.plot(times, np.sum(presses[:,20:50,20:50], axis=(1, 2)) * ((pixel_size_on_mirror * downsampling_factor) ** 2.), label="VFM force from subsection of plate")
plt.plot(hammer_time, hammer_force, label="Impact hammer")
plt.xlim(left=0.0008, right=0.003)
plt.ylim(top=500, bottom=-100)
plt.xlabel("Time [ms]")
plt.ylabel(r"Force [N]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
