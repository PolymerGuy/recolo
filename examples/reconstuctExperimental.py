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
    data = np.genfromtxt("./experimentalPressures/trans_open_1.txt", skip_header=20)

    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, 8] / 10.
    return press - press[0], time


#data = loadmat("/home/sindreno/Rene/dataset/w_5_2_set1.mat")["w"] #x,y,frame
data =-np.moveaxis(np.load("disp_sindre.npy"),0,-1)#  * 2.94/1000.


# plate and model parameters
#mat_E = 210.e9  # Young's modulus [Pa]
mat_E = 190.e9  # Young's modulus [Pa]
mat_nu = 0.3  # Poisson's ratio []
plate_thick = 5e-3
rho = 8000

mat_D = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu))  # flexural rigidity [N m]
mat_D11 = (plate_thick ** 3.) / 12. * mat_E / (1. - mat_nu ** 2.)
mat_D12 = (plate_thick ** 3.) / 12. * mat_E * mat_nu / (1. - mat_nu ** 2.)
# pressure reconstruction parameters

win_size = 24

press_avgs = []
press_stds = []
times = []
presses = []


fields = fields_from_experiments(data,filter_time_sigma=1,filter_space_sigma=2)

fields_filtered = fields#.down_sampled(6)#.filtered_time(2).filtered_space(2)




_,npts_x, npts_y = fields.shape()
dx = 2.94/1000.
dy = 2.94/1000.

bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]

virtual_field = recon.virtual_fields.Hermite16(win_size, dx)

for i,field in enumerate(fields_filtered):
    print("Processing frame %i"%i)
    recon_press, internal_energy = plate_iso_qs_lin(field, mat_D11, mat_D12, virtual_field,
                                                    rho=rho,thickness=plate_thick)

    center = int(recon_press.shape[0]/2)
    presses.append(recon_press)
    press_avgs.append(recon_press[center,center])
    press_stds.append(np.std(recon_press))
    times.append(field.time)

# plt.plot( times[::1], press_avgs,
#         label="Reconstructed pressure, S_space=%i, S_time=%i" % (sigma_spatial, sigma_time))
#plt.plot(times[100:140]-times[100],press_avgs[100:140], label="Reconstructed pressure")
#plt.plot(times[:],press_avgs[:], label="Reconstructed pressure")
plt.plot(press_avgs[:], label="Reconstructed pressure")
plt.legend(frameon=False)
plt.ylim(bottom=-1000, top=80000)
real_press, real_time = read_exp_press_data()
plt.figure()
from scipy.ndimage import gaussian_filter
plt.plot(real_time, real_press * 1.e6, '--', label="Pressure applied to FEA")
plt.plot(real_time, gaussian_filter(real_press,sigma=2.*500/75) * 1.e6, '--', label="We should get this curve")
#plt.plot(real_time, gaussian_filter(real_press,sigma=4.*500/75) * 1.e6, '--', label="Pressure applied to FEA")
plt.xlim(left=0.00025, right=0.0006)
plt.ylim(top=80000, bottom=0)
plt.xlabel("Time [Sec]")
plt.ylabel("Pressure [Pa]")

plt.legend(frameon=False)
plt.show()
