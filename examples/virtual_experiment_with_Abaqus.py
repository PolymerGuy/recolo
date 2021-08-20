import recon
import numpy as np
from scipy.ndimage import gaussian_filter, zoom
import matplotlib.pyplot as plt

plt.style.use('science')


def read_exp_press_data(experiment="open channel"):
    start = 25550 + 38
    end = 26200

    data = np.genfromtxt("/home/sindreno/Downloads/Rene/Valid_0.125_3.txt", skip_header=20)

    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, :] / 10.
    return press - press[0, :], time


def deform_grid_from_deflection(deflection_field, pixel_size, mirror_grid_dist, grid_pitch, upscale=4, oversampling=5):
    if upscale > 1:
        disp_fields = zoom(deflection_field, upscale, prefilter=True, order=3)
    else:
        disp_fields = deflection_field

    slopes_x, slopes_y = np.gradient(disp_fields, pixel_size)
    u_x = slopes_x * mirror_grid_dist * 2.
    u_y = slopes_y * mirror_grid_dist * 2.

    n_pix_x, n_pix_y = disp_fields.shape
    xs, ys = np.meshgrid(np.arange(n_pix_x), np.arange(n_pix_y))

    interp_u = recon.artificial_grid_deformation.interpolated_disp_field(u_x, u_y, dx=1, dy=1, order=3, mode="nearest")

    Xs, Ys = recon.artificial_grid_deformation.find_coords_in_undef_conf(xs, ys, interp_u, tol=1e-9)

    grid_deformed = recon.artificial_grid_deformation.make_dotted_grid(Xs, Ys, grid_pitch, oversampling=oversampling)

    return grid_deformed


def deflectometry_from_grid(grid_undeformed, grid_deformed, mirror_grid_dist, grid_pitch):
    phase_x, phase_y = recon.deflectomerty.detect_phase(grid_deformed, grid_pitch)
    phase_x0, phase_y0 = recon.deflectomerty.detect_phase(grid_undeformed, grid_pitch)

    disp_x_from_phase, disp_y_from_phase = recon.deflectomerty.disp_from_phase(phase_x, phase_x0, phase_y, phase_y0,
                                                                               grid_pitch, correct_phase=True)

    slopes_x = recon.deflectomerty.angle_from_disp(disp_x_from_phase, mirror_grid_dist)
    slopes_y = recon.deflectomerty.angle_from_disp(disp_y_from_phase, mirror_grid_dist)

    return slopes_x, slopes_y


deflectometry = True

# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.33  # Poisson's ratio []
density = 7700
plate_thick = 5e-3
plate = recon.calculate_plate_stiffness(mat_E, mat_nu, density, plate_thick)

mirror_grid_dist = 500.

use_img_ids = np.arange(0, 300)
abq_sim_fields = recon.load_abaqus_rpts("/home/sindreno/Rene/testfolder/fields/", use_only_img_ids=use_img_ids)

upscale = 4
n_frames, n_pix_x, n_pix_y = abq_sim_fields.disp_fields.shape
n_pix_x *= upscale
n_pix_y *= upscale

disp_fields = abq_sim_fields.disp_fields

#disp_fields = disp_fields[:,5:-5,5:-5]


# pressure reconstruction parameters
win_size = 30
sampling_rate = 1. / (abq_sim_fields.times[1] - abq_sim_fields.times[0])

# Load slope fields and calculate displacement fields
grid_pitch = 5.  # pixels
pixel_size_x = abq_sim_fields.plate_len_x / n_pix_x  # m

#slopes_x, slopes_y = np.gradient(disp_fields, pixel_size_x, axis=(1, 2))

#sloppes_x = []
#for i in range(n_frames):
#    sloppes_x.append(zoom(slopes_x[i, :, :], upscale, prefilter=True, order=3))
#slopes_x = np.array(sloppes_x)

#sloppes_y = []
#for i in range(n_frames):
#    sloppes_y.append(zoom(slopes_y[i, :, :], upscale, prefilter=True, order=3))
#slopes_y = np.array(sloppes_y)

# Stuff

if deflectometry:
    undeformed_grid = deform_grid_from_deflection(disp_fields[0, :, :], pixel_size_x, mirror_grid_dist, grid_pitch,
                                                  upscale=upscale)
    sloppes_x = []
    sloppes_y = []
    for i in range(n_frames):
        print(i)
        deformed_grid = deform_grid_from_deflection(disp_fields[i, :, :], pixel_size_x, mirror_grid_dist, grid_pitch,
                                                    upscale=upscale)
        slopes_x, slopes_y = deflectometry_from_grid(undeformed_grid, deformed_grid, mirror_grid_dist, grid_pitch)
        sloppes_x.append(slopes_x)
        sloppes_y.append(slopes_y)
    slopes_x = np.array(sloppes_x)
    slopes_y = np.array(sloppes_y)

slopes_x = np.moveaxis(slopes_x, 0, -1)
slopes_y = np.moveaxis(slopes_y, 0, -1)

disp_fields = recon.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size_x,
                                                       zero_at="bottom corners", zero_at_size=5,
                                                       extrapolate_edge=0, filter_sigma=0, downsample=1)

# Results are stored in these lists
times = []
presses = []

fields = recon.kinematic_fields_from_experiments(disp_fields, pixel_size_x, sampling_rate,
                                                 filter_time_sigma=0 * 2. * 500. / 75.,
                                                 filter_space_sigma=0)

virtual_field = recon.virtual_fields.Hermite16(win_size, pixel_size_x)

for i, field in enumerate(fields):
    print("Processing frame %i" % i)
    recon_press = recon.solver.plate_iso_qs_lin(field, plate, virtual_field)
    presses.append(recon_press)
    times.append(field.time)

presses = np.array(presses)
center = int(presses.shape[1] / 2)

# Plot the results
plt.plot((np.array(times[:])) * 1000., presses[:, center, center], '-', )

real_press, real_time = read_exp_press_data(experiment="open channel")

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
