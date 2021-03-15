import matplotlib.pyplot as plt
from recon import plate_iso_qs_lin, load_abaqus_rpts, fields_from_abaqus_rpts, fields_from_experiments
import recon
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt


def read_exp_press_data():
    import numpy as np

    start = 25500# + 38
    end = 26200
    data = np.genfromtxt("/home/sindreno/Downloads/Rene/Valid_0.125_3.txt", skip_header=20)

    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, 8] / 10.
    return press - press[0], time


data = loadmat("/home/sindreno/Rene/dataset/w_5_2_set1.mat")["w"] #x,y,frame


# plate and model parameters
#mat_E = 210.e9  # Young's modulus [Pa]
mat_E = 190.e9  # Young's modulus [Pa]
mat_nu = 0.3  # Poisson's ratio []
plate_thick = 5e-3

mat_D = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu))  # flexural rigidity [N m]
mat_D11 = (plate_thick ** 3.) / 12. * mat_E / (1. - mat_nu ** 2.)
mat_D12 = (plate_thick ** 3.) / 12. * mat_E * mat_nu / (1. - mat_nu ** 2.)
# pressure reconstruction parameters

win_size = 24

press_avgs = []
press_stds = []
times = []
presses = []


fields = fields_from_experiments(data)

fields_filtered = fields#.down_sampled(6)#.filtered_time(2).filtered_space(2)




_,npts_x, npts_y = fields.shape()
dx = 0.15 / float(npts_x)
dy = 0.15 / float(npts_y)

bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]

virtual_field = recon.virtual_fields.Hermite16(win_size, dx)

for i,field in enumerate(fields_filtered):
    print("Processing frame %i"%i)
    recon_press, internal_energy = plate_iso_qs_lin(win_size, field, mat_D11, mat_D12, virtual_field,
                                                    shift_res=True, return_valid=True)

    presses.append(recon_press)
    press_avgs.append(recon_press[16,16])
    press_stds.append(np.std(recon_press))
    times.append(field.time)

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
