import matplotlib.pyplot as plt
from recon import plate_iso_qs_lin, field_from_disp_func, analydisp, Hermite16, field_from_abaqus
import numpy as np
from scipy.ndimage import shift, gaussian_filter, gaussian_filter1d
import natsort
import os
from scipy.interpolate import griddata

from scipy.signal import butter,filtfilt

def read_exp_press_data():
    import numpy as np

    start = 25600+38
    end = 26200
    data = np.genfromtxt("/home/sindreno/Downloads/Rene/Valid_0.125_3.txt",skip_header=20)

    time = data[start:end,0] * 1.e-3
    time = time-time[0]
    press = data[start:end,8]/10.
    return press-press[0], time


def butter_lowpass_filter(data, cutoff, nyq,order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def list_files_in_folder(path, file_type=".rpt"):
    """ List all files with a given extension for a given path. The output is sorted
        Parameters
        ----------
        path : str
            Path to the folder containing the files
        file_type : str
            The file extension ex. ".rpt"
        Returns
        -------
        list
            A list of sorted file names
        """
    return natsort.natsorted([file for file in os.listdir(path) if file.endswith(file_type)])


def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2)) ** 2.)


def mea_diff(array1, array2):
    return np.nanmean((array1 - array2))


def load_abaqus_rpts(path_to_rpts):

    rpt_file_paths = list_files_in_folder(path_to_rpts,file_type=".rpt")
    print("Reading %i files"%len(rpt_file_paths))

    disp_fields = []
    accel_fields = []
    times = []
    for file_name in rpt_file_paths:
        path_to_rpt = os.path.join(path_to_rpts,file_name)
        field_data = np.genfromtxt(path_to_rpt, dtype=float,
                                   skip_header=19)

        time = np.genfromtxt(path_to_rpt,dtype=str,skip_header=8,max_rows=1)[-1]

        node_label = field_data[:, 0]
        node_coord_x = field_data[:, 1]
        node_coord_y = field_data[:, 2]
        node_disp_z = field_data[:, 3]
        node_acceleration_z = field_data[:, 4]

        # All data is assumed to be sampled on a square grid
        seed = int(node_disp_z.size ** 0.5)

        plate_len_x = (node_coord_x.max() - node_coord_x.min()) * 1e-3
        plate_len_y = (node_coord_y.max() - node_coord_y.min()) * 1e-3

        disp_field = -node_disp_z.reshape((seed, seed)) * 1e-3
        accel_field = -node_acceleration_z.reshape((seed, seed)) * 1e-3

        disp_fields.append(disp_field)
        accel_fields.append(accel_field)
        times.append(float(time))

    return np.array(disp_fields), np.array(accel_fields), times, plate_len_x, plate_len_y


###
###
# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.33  # Poisson's ratio []
plate_len_x = 0.3
plate_len_y = 0.3
plate_thick = 5e-3

mat_D = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu))  # flexural rigidity [N m]
mat_D11 = (plate_thick ** 3.) / 12. * mat_E / (1. - mat_nu ** 2.)
mat_D12 = (plate_thick ** 3.) / 12. * mat_E * mat_nu / (1. - mat_nu ** 2.)
# pressure reconstruction parameters

sigmas_spatial = [2]
sigma_times = [2]

for sigma_time in sigma_times:
    for sigma_spatial in sigmas_spatial:
        avgs = []
        stds = []



        disp_fields, accel_fields, times, plate_len_x, plate_len_y = load_abaqus_rpts("/home/sindreno/Rene/testfolder/fields/")

        disp_fields = disp_fields
        disp_fields = disp_fields#-disp_fields[:,0,0][:,np.newaxis,np.newaxis]
        #accel_fields = accel_fields

        analysis_time = times[-1]
        n_time_frames = len(times)
        time_step_size = float(analysis_time)/float(n_time_frames)

        disp_fields = np.array(disp_fields)

        down_sample_factor = 1
        disp_fields = disp_fields[::down_sample_factor]
        time_step_size = time_step_size * down_sample_factor
        print("Sampling rate of %f fps"%(1./time_step_size))


        #disp_fields = gaussian_filter(disp_fields,sigma=2)
        disp_fields = gaussian_filter1d(disp_fields,sigma=sigma_time,axis=0)

        for i in range(len(disp_fields)):
            #print("Filtering frame %i"%i)
            disp_fields[i,:,:] = gaussian_filter(disp_fields[i,:,:], sigma=sigma_spatial)

        #time_steps = np.gradient(np.array(times))
        #time_steps = np.array(times)[1:]-np.array(times)[:-1]
        #time_steps = np.concatenate(([time_steps[0]],time_steps))
        #disp_fields = gaussian_filter(disp_fields,sigma=2)



        #accel_fields = np.array(accel_fields)[::3]
        vel_fields = np.gradient(disp_fields,axis=0)/time_step_size
        accel_fields = np.gradient(vel_fields,axis=0)/time_step_size



        print(disp_fields.shape)
        print(accel_fields.shape)

        for i in range(disp_fields.shape[0]):
            print("Reconstructing frame %i" % i)

            #disp_field, accel_field, plate_len_x, plate_len_y = load_abaqus_rpt("/home/sindreno/Rene/testfolder/fields/fields_frame%i.rpt" % i)

            disp_field = disp_fields[i,:,:]
            accel_field = accel_fields[i,:,:]


            error = []
            win_sizes = []
            n_pts_x, n_pts_y =  disp_field.shape
            dx = plate_len_x / float(n_pts_x)
            dy = plate_len_y / float(n_pts_y)
            # for win_size in np.arange(4,50,2):
            for win_size in [24]:
                # for win_size in np.arange(4,40,4):
                # win_size = 24
                bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]

                # deflection = analydisp.sinusoidal_load(press, plate_len_x, plate_len_y, bend_stiff)

                # fields = field_from_disp_func(deflection, n_pts_x, n_pts_y, plate_len_x, plate_len_y)
                fields = field_from_abaqus(disp_field, accel_field, plate_len_x, plate_len_y)

                # define piecewise virtual fields
                virtual_fields = Hermite16(win_size, float(dx * win_size))

                recon_press, internal_energy = plate_iso_qs_lin(win_size, fields, mat_D11, mat_D12, virtual_fields)
                recon_press = shift(recon_press, shift=0.5, order=3)
                avg = np.mean(recon_press[win_size:-win_size, win_size:-win_size])
                std = np.std(recon_press[win_size:-win_size, win_size:-win_size])

                avgs.append(avg)
                stds.append(std)

                # print("Average recon pressure %f"%avg)

                external_work = np.sum(recon_press * fields.deflection * dx * dy)
                # print("Internal energy %f"%internal_energy)
                # print("External energy %f"%external_work)

                # rms_error = mea_diff(recon_press[win_size:-win_size,win_size:-win_size],fields.press[win_size:-win_size,win_size:-win_size])
                # error.append(internal_energy)
                # win_sizes.append(win_size)
        plt.plot(times[::down_sample_factor], avgs, label="Reconstructed pressure, S_space=%i, S_time=%i" % (sigma_spatial, sigma_time))

real_press,real_time = read_exp_press_data()
plt.plot(real_time,real_press * 1.e6,'--', label="Pressure applied to FEA")
plt.xlim(left=0,right=0.0004)
plt.ylim(top=80000,bottom=-10000)
plt.xlabel("Time [Sec]")
plt.ylabel("Pressure [Pa]")

plt.legend(frameon=False)
plt.show()
# print(recon_press.shape)
# print(fields.press.shape)
# plt.figure()
# plt.imshow(recon_press[win_size:-win_size,win_size:-win_size])
# plt.title("Recon - Truth")
# plt.colorbar()
# plt.legend(frameon=False)
# plt.show()

# plt.plot(recon_press[win_size:-win_size,win_size:-win_size][15,:])
# plt.show()
