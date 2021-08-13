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

def deform_grid_from_deflection(deflection_field, pixel_size, mirror_grid_dist, grid_pitch, upscale=1,oversampling=5):
    disp_fields = zoom(deflection_field, upscale, prefilter=True, order=3)

    slopes_x, slopes_y = np.gradient(disp_fields, pixel_size)
    u_x = slopes_x * mirror_grid_dist * 2.
    u_y = slopes_y * mirror_grid_dist * 2.

    xs, ys = np.meshgrid(np.arange(n_pix_x), np.arange(n_pix_y))

    interp_u = recon.artificial_grid_deformation.interpolated_disp_field(u_x, u_y, dx=1, dy=1, order=3, mode="nearest")

    Xs, Ys = recon.artificial_grid_deformation.find_coords_in_undef_conf(xs, ys, interp_u, tol=1e-9)

    grid_deformed = recon.artificial_grid_deformation.make_dotted_grid(Xs, Ys, grid_pitch, oversampling=oversampling)

    return grid_deformed

def deflectometry_from_grid(grid_undeformed,grid_deformed,mirror_grid_dist,grid_pitch):
    phase_x, phase_y = recon.deflectomerty.detect_phase(grid_deformed, grid_pitch)
    phase_x0, phase_y0 = recon.deflectomerty.detect_phase(grid_undeformed, grid_pitch)

    disp_x_from_phase, disp_y_from_phase = recon.deflectomerty.disp_from_phase(phase_x, phase_x0, phase_y, phase_y0,
                                                                               grid_pitch, correct_phase=True)

    slopes_x = recon.deflectomerty.angle_from_disp(disp_x_from_phase, mirror_grid_dist)
    slopes_y = recon.deflectomerty.angle_from_disp(disp_y_from_phase, mirror_grid_dist)

    return slopes_x,slopes_y


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

upscale = 8
n_frames, n_pix_x, n_pix_y = abq_sim_fields.disp_fields.shape

disp_fields = []
for i in range(n_frames):
    disp_fields.append(zoom(abq_sim_fields.disp_fields[i, :, :], upscale, prefilter=True, order=3))

disp_fields = np.array(disp_fields)

#disp_fields = disp_fields[:, 20:-20, 20:-20]

n_frames, n_pix_x, n_pix_y = disp_fields.shape

print(disp_fields.shape)

print(n_pix_x, n_pix_y)

# pressure reconstruction parameters
win_size = 30
sampling_rate = 1. / (abq_sim_fields.times[1] - abq_sim_fields.times[0])

# Load slope fields and calculate displacement fields
grid_pitch = 5.  # pixels
pixel_size_x = abq_sim_fields.plate_len_x / n_pix_x  # m

pixel_size_on_mirror = 1

slopes_x, slopes_y = np.gradient(disp_fields, pixel_size_x, axis=(1, 2))

slopes_x_cp = slopes_x

print("shape of slopes is ", slopes_x.shape)

# Stuff
if deflectometry:
    xs, ys = np.meshgrid(np.arange(n_pix_x), np.arange(n_pix_y))

    u_x = slopes_x * mirror_grid_dist * 2.
    u_y = slopes_y * mirror_grid_dist * 2.
    dx = 1

    grid_undeformed = recon.artificial_grid_deformation.make_dotted_grid(xs, ys, grid_pitch, oversampling=5)

    sloppes_x = []
    sloppes_y = []
    for i in range(n_frames):
        print(i)
        print("U_x peak is %f" % u_x[i, :, :].max())

        interp_u = recon.artificial_grid_deformation.interpolated_disp_field(u_x[i, :, :], u_y[i, :, :], dx=dx, dy=dx,
                                                                             order=3, mode="nearest")

        Xs, Ys = recon.artificial_grid_deformation.find_coords_in_undef_conf(xs, ys, interp_u, tol=1e-9)

        grid_deformed = recon.artificial_grid_deformation.make_dotted_grid(Xs, Ys, grid_pitch, oversampling=5)

        phase_x, phase_y = recon.deflectomerty.detect_phase(grid_deformed, grid_pitch)
        phase_x0, phase_y0 = recon.deflectomerty.detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = recon.deflectomerty.disp_from_phase(phase_x, phase_x0, phase_y, phase_y0,
                                                                                   grid_pitch, correct_phase=True)

        slopes_x = recon.deflectomerty.angle_from_disp(disp_x_from_phase, mirror_grid_dist)
        slopes_y = recon.deflectomerty.angle_from_disp(disp_y_from_phase, mirror_grid_dist)

        sloppes_x.append(slopes_x)
        sloppes_y.append(slopes_y)

    slopes_x = np.array(sloppes_x)
    slopes_y = np.array(sloppes_y)
    print("shape of slopes is ", slopes_x.shape)

slopes_x = np.moveaxis(slopes_x, 0, -1)
slopes_y = np.moveaxis(slopes_y, 0, -1)

disp_fields = recon.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size_x,
                                                       zero_at="bottom corners", zero_at_size=5,
                                                       extrapolate_edge=0, filter_sigma=0, downsample=1)

# plt.imshow(disp_fields[4])
# plt.show()

# disp_fields = abq_sim_fields.disp_fields


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

plt.plot(real_time[::3] * 1000., real_press[::3, 8] * 1.e6, '--', label="Transducer", alpha=0.7)

plt.plot(real_time * 1000, gaussian_filter(real_press[:, 8] * 1.e6, sigma=2. * 500. / 75.), '--',
         label="We should get this curve for sigma=2")

plt.xlim(left=0.000, right=0.9)
plt.ylim(top=80000, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
