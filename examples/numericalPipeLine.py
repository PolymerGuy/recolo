import recon
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

plt.style.use('science')


def read_exp_press_data(experiment="open channel"):
    start = 25550# + 38
    end = 26200

    if experiment == "open channel":
        data = np.genfromtxt("/home/sindreno/Downloads/Rene/Valid_0.125_3.txt", skip_header=20)
    else:
        data = np.genfromtxt("./experimentalPressures/trans_half_1.txt", skip_header=20)
    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, :] / 10.
    return press - press[0, :], time


# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.3  # Poisson's ratio []
plate_thick = 5.e-3
rho = 7700

abq_sim_fields = recon.load_abaqus_rpts("/home/sindreno/Rene/testfolder/fields/")
n_frames, n_pix_x, n_pix_y = abq_sim_fields.disp_fields.shape

plate = recon.calculate_plate_stiffness(mat_E, mat_nu, rho, plate_thick)

# pressure reconstruction parameters
win_size = 30
sampling_rate = 1. / (abq_sim_fields.times[1] - abq_sim_fields.times[0])

# Load slope fields and calculate displacement fields
grid_pitch = 1.  # pixels
grid_pitch_len = abq_sim_fields.plate_len_x / n_pix_x  # m

pixel_size_on_mirror = grid_pitch_len / grid_pitch
crops = range(1,21,5)
#crops = [0]
for crop in crops:
    slopes_x, slopes_y = np.gradient(abq_sim_fields.disp_fields, pixel_size_on_mirror, axis=(1, 2))
    slopes_x = np.moveaxis(slopes_x, 0, -1)[crop:-crop,crop:-crop,:]
    slopes_y = np.moveaxis(slopes_y, 0, -1)[crop:-crop,crop:-crop,:]



    disp_fields = recon.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size_on_mirror,
                                                           zero_at="bottom corners",
                                                           extrapolate_edge=0, filter_sigma=0, downsample=1)

    # Results are stored in these lists
    times = []
    presses = []

    fields = recon.kinematic_fields_from_deflections(disp_fields, pixel_size_on_mirror, sampling_rate, filter_time_sigma=0 * 2. * 500. / 75.,
                                                     filter_space_sigma=0)
    virtual_field = recon.virtual_fields.Hermite16(win_size, pixel_size_on_mirror)

    for i, field in enumerate(fields):
        print("Processing frame %i" % i)
        recon_press = recon.solver.pressure_elastic_thin_plate(field, plate, virtual_field)
        presses.append(recon_press)
        times.append(field.time)

    presses = np.array(presses)
    center = int(presses.shape[1] / 2)

    # Plot the results
    plt.plot((np.array(times[:])) * 1000., presses[:, center, center], '-',
             label="Reconstruction cropped by %i pixels"%crop)

real_press, real_time = read_exp_press_data(experiment="open channel")

plt.plot(real_time * 1000., real_press[:, 8] * 1.e6, '--', label="Transducer", alpha=0.7)

plt.plot(real_time * 1000, gaussian_filter(real_press[:, 8] * 1.e6, sigma=2. * 500. / 75.), '--',
         label="We should get this curve for sigma=2")

plt.xlim(left=0.000, right=0.9)
plt.ylim(top=80000, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
