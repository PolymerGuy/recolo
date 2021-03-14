import matplotlib.pyplot as plt
from recon import plate_iso_qs_lin, load_abaqus_rpts, fields_from_abaqus_rpts
import recon
import numpy as np

from scipy.signal import butter, filtfilt


def read_exp_press_data():
    import numpy as np

    start = 25600 + 38
    end = 26200
    data = np.genfromtxt("/home/sindreno/Downloads/Rene/Valid_0.125_3.txt", skip_header=20)

    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, 8] / 10.
    return press - press[0], time


def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.33  # Poisson's ratio []
plate_thick = 5e-3

mat_D = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu))  # flexural rigidity [N m]
mat_D11 = (plate_thick ** 3.) / 12. * mat_E / (1. - mat_nu ** 2.)
mat_D12 = (plate_thick ** 3.) / 12. * mat_E * mat_nu / (1. - mat_nu ** 2.)
# pressure reconstruction parameters

win_size = 24

press_avgs = []
press_stds = []
times = []

abq_simulation = load_abaqus_rpts("/home/sindreno/Rene/testfolder/fields/")

fields = fields_from_abaqus_rpts(abq_simulation, downsample=6, accel_from_disp=True, filter_time_sigma=2)

fields_filtered = fields#.down_sampled(6)#.filtered_time(2).filtered_space(2)

presses = []
for i,field in enumerate(fields_filtered):
    field.deflection = field.deflection-field.deflection[0,0]
    print("Processing frame %i"%i)
    error = []
    win_sizes = []
    n_pts_x, n_pts_y = field.deflection.shape
    dx = abq_simulation.plate_len_x / float(abq_simulation.npts_x)
    dy = abq_simulation.plate_len_y / float(abq_simulation.npts_y)
    # for win_size in np.arange(4,50,2):
    bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]

    # define piecewise virtual fields
    virtual_field = recon.virtual_fields.Hermite16(win_size, dx)

    recon_press, internal_energy = plate_iso_qs_lin(win_size, field, mat_D11, mat_D12, virtual_field,
                                                    shift_res=True, return_valid=True)

    presses.append(recon_press)
    press_avgs.append(np.mean(recon_press))
    press_stds.append(np.std(recon_press))
    times.append(field.time)

presses = np.array(presses)
# plt.plot( times[::1], press_avgs,
#         label="Reconstructed pressure, S_space=%i, S_time=%i" % (sigma_spatial, sigma_time))
plt.plot(times,press_avgs, label="Reconstructed pressure")
real_press, real_time = read_exp_press_data()
plt.plot(real_time, real_press * 1.e6, '--', label="Pressure applied to FEA")
plt.xlim(left=0, right=0.0004)
plt.ylim(top=80000, bottom=-10000)
plt.xlabel("Time [Sec]")
plt.ylabel("Pressure [Pa]")

plt.legend(frameon=False)
plt.show()
