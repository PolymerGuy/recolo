import recon
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from experimentalPressures.experimental_data import read_exp_press_data

plt.style.use('science')

# Parameters for parameter study
crop_pixels = -2

# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.3  # Poisson's ratio []
plate_thick = 4.95e-3
rho = 7934.

plate = recon.calculate_plate_stiffness(mat_E, mat_nu, rho, plate_thick)

# pressure reconstruction parameters
win_size = 12
sampling_rate = 75000.

# Load slope fields and calculate displacement fields


path = "/home/sindreno/gridmethod_Rene/images_full_2"
img_paths = recon.utils.list_files_in_folder(path, file_type=".tif", abs_path=True)

grid_pitch = 5.08  # pixels
grid_pitch_len = 5.88 / 1000.  # m

mirror_grid_distance = 1.37  # m

pixel_size_on_grid_plane = grid_pitch_len / grid_pitch
pixel_size_on_mirror = grid_pitch_len/grid_pitch * 0.5


ref_img_ids = range(50,60)
use_imgs = range(50, 150)

slopes_y, slopes_x = recon.deflectomerty.slopes_from_grid_imgs_dic(path, grid_pitch, pixel_size_on_grid_plane,
                                                               mirror_grid_distance, ref_img_ids=ref_img_ids,
                                                               only_img_ids=use_imgs)
pixel_size_on_mirror = pixel_size_on_mirror * 12.3

disp_fields = recon.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size_on_mirror, zero_at="top corners",
                                                       extrapolate_edge=0, filter_sigma=2, downsample=1)

# Results are stored in these lists
times = []
presses = []

fields = recon.kinematic_fields_from_experiments(disp_fields, pixel_size_on_mirror, sampling_rate, filter_time_sigma=0,
                                                 filter_space_sigma=0)
virtual_field = recon.virtual_fields.Hermite16(win_size, pixel_size_on_mirror)

for i, field in enumerate(fields):
    print("Processing frame %i" % i)
    recon_press = recon.solver.plate_iso_qs_lin(field, plate, virtual_field)
    presses.append(recon_press)
    times.append(field.time)

presses = np.array(presses)
center = int(presses.shape[1] / 2)

# Plot the results
plt.plot((np.array(times[:]) - 0.000500) * 1000., presses[:, center, center] / 1000., '-',
         label="Reconstruction")

real_press, real_time = read_exp_press_data(experiment="open channel")

plt.plot(real_time * 1000., real_press[:, 7] * 1.e3, '--', label="Transducer", alpha=0.7)

plt.plot(real_time * 1000, gaussian_filter(real_press[:, 7] * 1.e3, sigma=2. * 500. / 75.), '--',
         label="We should get this curve for sigma=2")

plt.xlim(left=0.000, right=0.9)
plt.ylim(top=80, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
