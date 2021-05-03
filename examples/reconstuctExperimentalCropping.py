import recon
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

plt.style.use('science')


def read_exp_press_data():
    start = 25500
    end = 26200

    data = np.genfromtxt("./experimentalPressures/trans_open_1.txt", skip_header=20)
    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, :] / 10.
    return press - press[0, :], time


def crop_and_integrate_exp_data(crop_factor):
    from recon.slope_integration import sparce_integration
    from scipy.io import loadmat

    # Integrate slopes to obtain displacement fields
    # data = loadmat("/home/sindreno/Downloads/Rene/slopes_5_0_set1_full3.mat")
    data = loadmat("/home/sindreno/Downloads/Rene/slopes_1_0_set1_full2.mat")
    # data = loadmat("/home/sindreno/Downloads/Rene/slopes_5_0_set1_half1.mat")

    slopes_x = data["slope_x"]
    slopes_y = data["slope_y"]

    print(slopes_x.shape)

    slopes_x = slopes_x - np.mean(slopes_x[:, :, 0])
    slopes_y = slopes_y - np.mean(slopes_y[:, :, 0])

    pixel_size = 3.0 / 1000./5.

    disp_fields = []

    for i in np.arange(0, 100):
        print("Integrating frame %i cropped by %i pixels" % (i, crop_factor))
        slope_y = slopes_x[:, :, i]
        slope_x = slopes_y[:, :, i]

        slope_y = gaussian_filter(slope_y, sigma=10)
        slope_x = gaussian_filter(slope_x, sigma=10)

        slope_y = slope_y[:, :]
        slope_x = slope_x[:, :]

        if crop_factor > 0:
            slope_y = slope_y[crop_factor:-crop_factor, crop_factor:-crop_factor]
            slope_x = slope_x[crop_factor:-crop_factor, crop_factor:-crop_factor]
        elif crop_factor < 0:
            slope_x = np.pad(slope_x, pad_width=(-crop_factor, -crop_factor), mode="edge")
            slope_y = np.pad(slope_y, pad_width=(-crop_factor, -crop_factor), mode="edge")

        disp_field = sparce_integration.int2D(slope_x, slope_y, pixel_size, pixel_size, const_at_edge="bottom_corners")

        disp_fields.append(disp_field)

    return np.array(disp_fields)


# Parameters for parameter study
crop_pixels = [-2]

# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.3  # Poisson's ratio []
plate_thick = 4.95e-3
rho = 7934.

plate = recon.calculate_plate_stiffness(mat_E, mat_nu, rho, plate_thick)

# pressure reconstruction parameters
win_size = 30
pixel_size = 3.0 / 1000./5.
sampling_rate = 75000.

for crop_pixel in crop_pixels:

    raw_disp_field = -crop_and_integrate_exp_data(crop_pixel)
    # Results are stored in these lists
    times = []
    presses = []

    fields = recon.fields_from_experiments(raw_disp_field, pixel_size, sampling_rate, filter_time_sigma=0,
                                           filter_space_sigma=0)
    virtual_field = recon.virtual_fields.Hermite16(win_size, pixel_size)

    for i, field in enumerate(fields):
        print("Processing frame %i" % i)
        recon_press = recon.solver.plate_iso_qs_lin(field, plate, virtual_field)

        presses.append(recon_press)

        times.append(field.time)

    presses = np.array(presses)
    center = int(presses.shape[1] / 2)

    # Plot the results
    plt.plot((np.array(times[:]) + 0.000025) * 1000., presses[:, center, center] / 1000., '-',
             label="Reconstruction")

real_press, real_time = read_exp_press_data()

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
