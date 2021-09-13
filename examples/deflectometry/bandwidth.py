# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])


import recon
import numpy as np
import matplotlib.pyplot as plt

# Test of the bandwidth of the grid method. This is done by deforming a grid using a harmonic displacement field with
# increasing frequency and determining the peak amplitude. The peak amplitude measured by the method decreases as
# the frequency of the displacement field increases, see [1] for further discussion.
#
# [1] Michel Grediac, Frédéric Sur, Benoît Blaysat. The grid method for in-plane displacement and
# strain measurement: a review and analysis. Strain, Wiley-Blackwell, 2016, 52 (3), pp.205-243.
# ff10.1111/str.12182ff. ffhal-01317145f

# Grid image settings
grid_pitch = 5

# Displacement field description
disp_amp = 0.01
disp_n_periodes = 2
disp_periodes = np.arange(25, 200, 5)

peak_disp_x = []
peak_disp_y = []
for disp_period in disp_periodes:
    # Generate the coordinates fields corresponding to the harmonic displacement field
    xs, ys, Xs_lagr, Ys_disp_lagr, u_x, u_y = recon.artificial_grid_deformation.harmonic_disp_field(disp_amp,
                                                                                                    disp_period,
                                                                                                    disp_n_periodes,
                                                                                                    formulation="lagrangian")
    # Generate the grid images
    grid_undeformed = recon.artificial_grid_deformation.dotted_grid(xs, ys, grid_pitch)
    grid_displaced_eulr = recon.artificial_grid_deformation.dotted_grid(Xs_lagr, Ys_disp_lagr, grid_pitch)

    # Calculate the phase fields
    phase_x, phase_y = recon.deflectomerty.detect_phase(grid_displaced_eulr, grid_pitch)
    phase_x0, phase_y0 = recon.deflectomerty.detect_phase(grid_undeformed, grid_pitch)

    # Calculate the displacements from the phase fields
    disp_x_from_phase, disp_y_from_phase = recon.deflectomerty.disp_fields_from_phases(phase_x, phase_x0, phase_y,
                                                                                       phase_y0,
                                                                                       grid_pitch,
                                                                                       correct_phase=True)

    peak_disp_x.append(np.max(disp_x_from_phase))
    peak_disp_y.append(np.max(disp_y_from_phase))

peak_disp_x = np.array(peak_disp_x)
peak_disp_y = np.array(peak_disp_y)

# Theoretical bias caused by limited bandwidth. See [1].
theoretical_bias = np.exp(-2. * np.pi ** 2. * grid_pitch ** 2. * (1. / np.array(disp_periodes)) ** 2.)

# Plot the results
plt.plot(disp_periodes, peak_disp_x / disp_amp, label="Grid pitch: %i" % grid_pitch)
plt.plot(disp_periodes, theoretical_bias, label="Theoretical bias")

plt.xlabel("Displacement periode [pix]")
plt.ylabel("Normalized displacement amplitude [-]")
plt.title("Displacement bandwidth for an amplitude of %.4f pixels " % disp_amp)

plt.hlines(0.9, np.min(disp_periodes), np.max(disp_periodes), linestyles="--")
plt.legend()
plt.show()
