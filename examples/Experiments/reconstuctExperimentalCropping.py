# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import recolo
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('science')

# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.3  # Poisson's ratio []
plate_thick = 2.98e-3
rho = 7934.

plate = recolo.make_plate(mat_E, mat_nu, rho, plate_thick)

# pressure reconstruction parameters
win_size = 30
sampling_rate = 10000.

# Load slope fields and calculate displacement fields


#path = "/home/sindreno/gridmethod_Rene/images_full_2"
path = "/home/sindreno/PressureRecon/impact_hammer/deflectometry"

grid_pitch = 7.0  # pixels
grid_pitch_len = 2.5 / 1000.  # m

mirror_grid_distance = 1.63  # m

pixel_size_on_grid_plane = grid_pitch_len / grid_pitch
pixel_size_on_mirror = grid_pitch_len / grid_pitch * 0.5

ref_img_ids = range(50, 60)
use_imgs = range(80, 130)

slopes_y, slopes_x = recolo.deflectomerty.slopes_from_images(path, grid_pitch, mirror_grid_distance,
                                                            ref_img_ids=ref_img_ids, only_img_ids=use_imgs,
                                                            crop=(45, 757,0,-1),window="gaussian",correct_phase=False)

disp_fields = recolo.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size_on_mirror,
                                                       zero_at="bottom corners",zero_at_size=20,
                                                       extrapolate_edge=0, filter_sigma=10, downsample=1)

# Results are stored in these lists
times = []
presses = []

fields = recolo.kinematic_fields_from_deflections(disp_fields, pixel_size_on_mirror, sampling_rate, filter_time_sigma=0,
                                                 filter_space_sigma=10)
virtual_field = recolo.virtual_fields.Hermite16(win_size, pixel_size_on_mirror)

for i, field in enumerate(fields):
    print("Processing frame %i" % i)
    recon_press = recolo.solver_VFM.calc_pressure_thin_elastic_plate(field, plate, virtual_field)
    presses.append(recon_press)
    times.append(field.time)

presses = np.array(presses)
center = int(presses.shape[1] / 2)

# Plot the results
plt.plot((np.array(times[:]) - 0.000500) * 1000., presses[:, center, center] / 1000., '-',
         label="Reconstruction")

#real_press, real_time = read_exp_press_data(experiment="open channel")

#plt.plot(real_time * 1000., real_press[:, 7] * 1.e3, '--', label="Transducer", alpha=0.7)

#plt.plot(real_time * 1000, gaussian_filter(real_press[:, 7] * 1.e3, sigma=2. * 500. / 75.), '--',
#         label="We should get this curve for sigma=2")

plt.xlim(left=0.000, right=0.9)
plt.ylim(top=80, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()

import wget