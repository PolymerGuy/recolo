# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath

sys.path.extend([abspath(".")])

import recon
import numpy as np
import matplotlib.pyplot as plt

# Comparison of a Eulerian and Lagrangian formulation for the displacements used in artificial grid deformation. The
# current coordinates x = X + u(X) is here referred to as the Lagrangian formulation the current coordinates x = X +
# u(x) is here referred to as the Eulerian formulation. When the grid is to be deformed the grey scale values are
# sampled at X and the sensor is fixed in space with coordinates x. This requires solving x = X + u(X) for X when
# the Lagrangian formulation is used.

# Grid image settings
grid_pitch = 5
oversampling = 9

# Displacement field description
disp_amp = 2.
disp_period = 500
disp_n_periodes = 1.0


# Generate the coordinates fields corresponding to the harmonic displacement field
xs, ys, Xs_eulr, Ys_eulr, _, _ = recon.artificial_grid_deformation.harmonic_disp_field(disp_amp, disp_period,
                                                                                           disp_n_periodes,
                                                                                           formulation="eulerian")

_, _, Xs_lagr, Ys_disp_lagr, u_x, u_y = recon.artificial_grid_deformation.harmonic_disp_field(disp_amp,
                                                                                              disp_period,
                                                                                              disp_n_periodes,
                                                                                              formulation="lagrangian")

# Generate the grid images
grid_undeformed = recon.artificial_grid_deformation.dotted_grid(xs, ys, grid_pitch, oversampling=oversampling,
                                                                pixel_size=1)

grid_deformed_eulr = recon.artificial_grid_deformation.dotted_grid(Xs_eulr, Ys_eulr, grid_pitch,
                                                                   oversampling=oversampling,
                                                                   pixel_size=1)
grid_deformed_lagr = recon.artificial_grid_deformation.dotted_grid(Xs_lagr, Ys_disp_lagr, grid_pitch,
                                                                   oversampling=oversampling,
                                                                   pixel_size=1)
# Calculate the phase fields
phase_x0, phase_y0 = recon.deflectomerty.detect_phase(grid_undeformed, grid_pitch)
phase_x_eulr, phase_y_eulr = recon.deflectomerty.detect_phase(grid_deformed_eulr, grid_pitch)
phase_x_lagr, phase_y_lagr = recon.deflectomerty.detect_phase(grid_deformed_lagr, grid_pitch)

# Calculate the displacements from the phase fields
disp_x_eulr, _ = recon.deflectomerty.disp_fields_from_phases(phase_x_eulr, phase_x0, phase_y_eulr, phase_y0, grid_pitch,
                                         correct_phase=False)

disp_x_lagr, _ = recon.deflectomerty.disp_fields_from_phases(phase_x_lagr, phase_x0, phase_y_eulr, phase_y0, grid_pitch,
                                         correct_phase=False)

peak_disp_x_eulr = np.max(np.abs(disp_x_eulr))

peak_disp_x_lagr = np.max(np.abs(disp_x_lagr))

# Crop correct displacement field to account for the reduced field size due to the phase detection.
u_x = u_x[grid_pitch * 4:-grid_pitch * 4, grid_pitch * 4:-grid_pitch * 4]

# Plot the results
center_pixel = 80
plt.plot(disp_x_lagr[center_pixel, :], "--", label="Measured Lagrangian")
plt.plot(disp_x_eulr[center_pixel, :], label="Measured Eulerian")
plt.plot(u_x[center_pixel, :], label="Correct")
plt.ylabel("Displacement [pix]")
plt.xlabel("Position [pix]")
plt.legend(frameon=False)
plt.twinx()
plt.plot(np.abs(disp_x_lagr[center_pixel, :] - u_x[center_pixel, :]), "--", color="red")
plt.plot(np.abs(disp_x_eulr[center_pixel, :] - u_x[center_pixel, :]), color="red")
plt.ylabel("Absolute difference [pix]", color="red")
plt.title("Comparison of Lagrangian u(X) and Eulerian u(x) displacements")
plt.tight_layout()
plt.show()
