import matplotlib.pyplot as plt
import recon
import numpy as np

def read_exp_press_data():
    import numpy as np

    start = 25550
    end = 26200
    data = np.genfromtxt("./examples/experimentalPressures/trans_open_1.txt", skip_header=20)

    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, 8] / 10.
    return press - press[0], time


# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.33  # Poisson's ratio []
density = 7700
plate_thick = 5e-3
plate = recon.calculate_plate_stiffness(mat_E, mat_nu, density, plate_thick)

# pressure reconstruction parameters
win_size = 30

# Load data from abaqus
abq_sim_fields = recon.load_abaqus_rpts("/home/sindreno/Rene/testfolder/fields/")


fields = recon.fields_from_abaqus_rpts(abq_sim_fields, downsample=5,downsample_space=2, accel_from_disp=True, filter_time_sigma=0,
                                       noise_amp_sigma=None)


# Define a piece-wise virtual field
n_pts_x, n_pts_y = fields(0).deflection.shape
dx = abq_sim_fields.plate_len_x / float(n_pts_x)
virtual_field = recon.virtual_fields.Hermite16(win_size, dx)

press_avgs = []
press_stds = []
times = []
presses = []
for i, field in enumerate(fields):
    print("Processing frame %i" % i)
    recon_press = recon.solver.plate_iso_qs_lin(field, plate, virtual_field)
    presses.append(recon_press)
    times.append(field.time)

# Plot results at the center of the fields
presses = np.array(presses)
center = int(presses.shape[1] / 2)

plt.plot(times, presses[:,center,center], '-o', label="Reconstructed pressure")
real_press, real_time = read_exp_press_data()
plt.plot(real_time, real_press * 1.e6, '--', label="Pressure applied to FEA")
plt.xlim(left=0, right=0.0007)
plt.ylim(top=80000, bottom=-15000)
plt.xlabel("Time [Sec]")
plt.ylabel("Pressure [Pa]")

plt.legend(frameon=False)
plt.show()
