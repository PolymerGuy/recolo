# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

from recon.deflectomerty.grid_method import detect_phase, disp_fields_from_phases
from recon.artificial_grid_deformation import harmonic_disp_field, dotted_grid
import numpy as np
import matplotlib.pyplot as plt


# Comparison of a Eulerian and Lagrangian formulation for the displacements used in artificial grid deformation. The
# current coordinates x = X + u(X) is here referred to as the Lagrangian formulation the current coordinates x = X +
# u(x) is here referred to as the Eulerian formulation. When the grid is to be deformed the grey scale values are
# sampled at X and the sensor is fixed in space with coordinates x. Thisnecessitates solving x = X + u(X) for X when
# the Lagrangian formulation is used.


grid_pitch = 5
oversampling = 9

disp_amp = 2.
disp_period = 500
disp_n_periodes = 1.0


xs, ys, xs_disp_eulr, ys_disp_eulr, _, _ = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                               formulation="eulerian")

_, _, xs_disp_lagr, ys_disp_lagr, u_x, u_y = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                                   formulation="lagrangian")

grid_undeformed = dotted_grid(xs, ys, grid_pitch, oversampling=oversampling, pixel_size=1)

grid_displaced_eulr = dotted_grid(xs_disp_eulr, ys_disp_eulr, grid_pitch, oversampling=oversampling,
                                  pixel_size=1)
grid_displaced_lagr = dotted_grid(xs_disp_lagr, ys_disp_lagr, grid_pitch, oversampling=oversampling,
                                  pixel_size=1)

phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)
phase_x_eulr, phase_y_eulr = detect_phase(grid_displaced_eulr, grid_pitch)
phase_x_lagr, phase_y_lagr = detect_phase(grid_displaced_lagr, grid_pitch)

disp_x_eulr, _ = disp_fields_from_phases(phase_x_eulr, phase_x0, phase_y_eulr, phase_y0, grid_pitch,
                                         correct_phase=True)

disp_x_lagr, _ = disp_fields_from_phases(phase_x_lagr, phase_x0, phase_y_eulr, phase_y0, grid_pitch,
                                         correct_phase=True)

peak_disp_x_eulr = np.max(np.abs(disp_x_eulr))

peak_disp_x_lagr = np.max(np.abs(disp_x_lagr))

u_x = u_x[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]

plt.plot(disp_x_lagr[80, :], "--",label="Measured Lagrangian")
plt.plot(disp_x_eulr[80, :], label="Measured Eulerian")
plt.plot(u_x[80, :], label="Correct")
plt.ylabel("Displacement [pix]")
plt.xlabel("Position [pix]")
plt.legend(frameon=False)
plt.twinx()
plt.plot(np.abs(disp_x_lagr[80, :]-u_x[80,:]),"--",color="red")
plt.plot(np.abs(disp_x_eulr[80, :]-u_x[80,:]),color="red")
plt.ylabel("Absolute difference [pix]",color="red")
plt.title("Comparison of Lagrangian u(X) and Eulerian u(x) displacements")
plt.tight_layout()
plt.show()



