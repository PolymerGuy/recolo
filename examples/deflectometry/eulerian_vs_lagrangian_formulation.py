from recon.deflectomerty.deflectometry import detect_phase, disp_from_phase
from recon.artificial_grid_deformation import harmonic_disp_field, make_dotted_grid
import numpy as np
import matplotlib.pyplot as plt

grid_pitch = 5
oversampling = 9

disp_amp = 2.
disp_period = 500
disp_n_periodes = 1.0


xs, ys, xs_disp_eulr, ys_disp_eulr, _, _ = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                               formulation="eulerian")

_, _, xs_disp_lagr, ys_disp_lagr, u_x, u_y = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                                   formulation="lagrangian")

grid_undeformed = make_dotted_grid(xs, ys, grid_pitch, oversampling=oversampling, pixel_size=1)

grid_displaced_eulr = make_dotted_grid(xs_disp_eulr, ys_disp_eulr, grid_pitch, oversampling=oversampling,
                                       pixel_size=1)
grid_displaced_lagr = make_dotted_grid(xs_disp_lagr, ys_disp_lagr, grid_pitch, oversampling=oversampling,
                                       pixel_size=1)

phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)
phase_x_eulr, phase_y_eulr = detect_phase(grid_displaced_eulr, grid_pitch)
phase_x_lagr, phase_y_lagr = detect_phase(grid_displaced_lagr, grid_pitch)

disp_x_eulr, _ = disp_from_phase(phase_x_eulr, phase_x0, phase_y_eulr, phase_y0, grid_pitch,
                                 correct_phase=True)

disp_x_lagr, _ = disp_from_phase(phase_x_lagr, phase_x0, phase_y_eulr, phase_y0, grid_pitch,
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



