import matplotlib.pyplot as plt
from recon import plate_iso_qs_lin, load_abaqus_rpts, fields_from_abaqus_rpts
import recon
import numpy as np

"""
Parameter study which chekcs the influence of spatial filtering, temporal filtering and down-sampling on the
reconstructed pressure. The input in generated using Abaqus where the experimental pressure-time profile
used below was applied to the plate assuming a homogenous pressure distribution.
"""


def read_exp_press_data():
    import numpy as np

    start = 25550# + 38
    end = 26200
    data = np.genfromtxt("./experimentalPressures/Valid_0.125_3.txt", skip_header=20)

    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, 8] / 10.  # Convert from Bars to MPa
    return press - press[0], time


# plate and model parameters
mat_E = 210.e9
mat_nu = 0.33
plate_thick = 5e-3

mat_D = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu))  # flexural rigidity [N m]
mat_D11 = (plate_thick ** 3.) / 12. * mat_E / (1. - mat_nu ** 2.)
mat_D12 = (plate_thick ** 3.) / 12. * mat_E * mat_nu / (1. - mat_nu ** 2.)
bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]
# pressure reconstruction parameters

win_size = 24

abq_simulation = load_abaqus_rpts("./AbaqusRPTs/")

spatial_sigmas = [0, 2, 4]
temporal_sigmas = [0, 2, 4]
downsampling_factors = [0, 2, 4, 6]

for downsampling_factor in downsampling_factors:
    for spatial_sigma in spatial_sigmas:
        for temporal_sigma in temporal_sigmas:

            fields = fields_from_abaqus_rpts(abq_simulation, downsample=downsampling_factor,bin_downsamples=False, accel_from_disp=True,
                                             filter_time_sigma=temporal_sigma, filter_space_sigma=spatial_sigma)

            presses = []
            press_avgs = []
            press_stds = []
            times = []

            for i, field in enumerate(fields):
                field.deflection = field.deflection - field.deflection[0, 0]
                print("Processing frame %i" % i)
                n_pts_x, n_pts_y = field.deflection.shape
                dx = abq_simulation.plate_len_x / float(abq_simulation.npts_x)
                dy = abq_simulation.plate_len_y / float(abq_simulation.npts_y)

                # define piecewise virtual fields
                virtual_field = recon.virtual_fields.Hermite16(win_size, dx)

                recon_press, internal_energy = plate_iso_qs_lin(win_size, field, mat_D11, mat_D12, virtual_field,
                                                                shift_res=True, return_valid=True)

                presses.append(recon_press)
                press_avgs.append(np.mean(recon_press))
                press_stds.append(np.std(recon_press))
                times.append(field.time)

            presses = np.array(presses)
            plt.plot(times, press_avgs, label="Temporal=%i Spatial=%i" % (temporal_sigma, spatial_sigma))


    real_press, real_time = read_exp_press_data()
    plt.plot(real_time, real_press * 1.e6, '--', label="Pressure applied to FEA", color="black")
    plt.xlim(left=0, right=0.0004)
    plt.ylim(top=80000, bottom=-10000)
    plt.xlabel("Time [Sec]")
    plt.ylabel("Pressure [Pa]")
    plt.title("Downsampling by %i" % downsampling_factor)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("./Studies/Filter_study_dwnsmpl%i.png" % downsampling_factor)
    # plt.show()
    plt.close()
