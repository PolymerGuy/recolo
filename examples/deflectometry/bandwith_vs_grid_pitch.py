# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

from recon.deflectomerty.grid_method import detect_phase, disp_fields_from_phases
from recon.artificial_grid_deformation import dotted_grid
import numpy as np
import matplotlib.pyplot as plt

# Test of the bandwith of the grid method. This is done by deforming a grid with a harmonic displacement field with
# increasing frequency and determining the peak amplitude. This is here done for different grid pitches.


grid_pitches = [5, 7, 9, 11]
disp_amp = 0.01

for grid_pitch in grid_pitches:
    rel_error_tol = 1e-3
    peak_disp_x = []
    peak_disp_y = []
    disp_periodes = np.arange(25, 200, 10)
    for disp_period in disp_periodes:
        disp_n_periodes = 2

        x = np.arange(disp_n_periodes * disp_period, dtype=float)
        y = np.arange(disp_n_periodes * disp_period, dtype=float)

        xs, ys = np.meshgrid(x, y)

        displacement_x = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * xs / xs.max())
        displacement_y = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * ys / ys.max())

        grid_undeformed = dotted_grid(xs, ys, grid_pitch)

        xs_disp = xs - displacement_x
        ys_disp = ys - displacement_y

        grid_displaced_eulr = dotted_grid(xs_disp, ys_disp, grid_pitch)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
        phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_fields_from_phases(phase_x, phase_x0, phase_y, phase_y0,
                                                                       grid_pitch,
                                                                       correct_phase=True)

        peak_disp_x.append(np.max(disp_x_from_phase))
        peak_disp_y.append(np.max(disp_y_from_phase))

    peak_disp_x = np.array(peak_disp_x)
    peak_disp_y = np.array(peak_disp_y)

    plt.plot(disp_periodes, peak_disp_x / disp_amp, label="Grid pitch: %i" % grid_pitch)
plt.xlabel("Displacement periode [pix]")
plt.ylabel("Normalized displacement amplitude [-]")
plt.title("Displacement bandwidth for an amplitude of %.4f pixels " % disp_amp)

plt.hlines(0.9, np.min(disp_periodes), np.max(disp_periodes), linestyles="--")
plt.legend(frameon=False)
plt.show()
