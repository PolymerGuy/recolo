import matplotlib.pyplot as plt
from recon import plate_iso_qs_lin, fields_from_experiments
import recon
import numpy as np
from scipy.ndimage import gaussian_filter


def read_exp_press_data():
    import numpy as np

    start = 25500  # + 38
    end = 26200
    data = np.genfromtxt("./experimentalPressures/trans_open_1.txt", skip_header=20)

    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, 8] / 10.
    return press - press[0], time


data = -np.moveaxis(np.load("disps_sindre.npy"), 0, -1)

# plate and model parameters
mat_E = 190.e9  # Young's modulus [Pa]
mat_nu = 0.3  # Poisson's ratio []
plate_thick = 5e-3
rho = 8000

plate = recon.calculate_plate_stiffness(mat_E, mat_nu, rho, plate_thick)

# pressure reconstruction parameters
win_size = 30
pixel_size_x = 2.94 / 1000.

# Results are stored in these lists
times = []
presses = []

fields = fields_from_experiments(data, filter_time_sigma=0, filter_space_sigma=0)

virtual_field = recon.virtual_fields.Hermite16(win_size, pixel_size_x)

for i, field in enumerate(fields):
    print("Processing frame %i" % i)
    recon_press = plate_iso_qs_lin(field, plate, virtual_field)

    presses.append(recon_press)
    times.append(field.time)

presses = np.array(presses)
center = int(presses.shape[1] / 2)

# Plot the results
plt.plot(np.array(times[:]) + 0.00007, presses[:,center-2,center+1], '-o', label="Reconstructed pressure")
plt.ylim(bottom=-1000, top=80000)
real_press, real_time = read_exp_press_data()

plt.plot(real_time, real_press * 1.e6, '--', label="Transducer")
plt.plot(real_time, gaussian_filter(real_press, sigma=2. * 500. / 75.) * 1.e6, '--',
         label="We should get this curve for sigma=2")
plt.xlim(left=0.000, right=0.0006)
plt.ylim(top=80000, bottom=0)
plt.xlabel("Time [Sec]")
plt.ylabel("Pressure [Pa]")

plt.legend(frameon=False)
plt.show()
