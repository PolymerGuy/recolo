# Comparison of Eularian displacements and Lagrangian displacements
import numpy as np
import matplotlib.pyplot as plt
from recon.deflectomerty.deflectometry import detect_phase, disp_from_phase
from skimage.restoration import unwrap_phase

plt.style.use('science')


def grid_grey_scales(xs, ys, pitch):
    return np.cos(2. * np.pi * xs / float(pitch)) * np.cos(2. * np.pi * ys / float(pitch))


def deformed_grid(xs, ys, disp_x, disp_y, grid_pitch):
    # Assumes x = X + u(x) which gives X = x - u(x)
    X = xs - disp_x
    Y = ys - disp_y
    return grid_grey_scales(X, Y, grid_pitch)


dev = []
amps = np.linspace(0., 2., 10)
amps = [0.1]
for disp_amp in amps:

    tol = 1e-12

    grid_pitch = 15  #
    n_pitches = 40
    n_periodes = 5

    x = np.arange(grid_pitch * n_pitches, dtype=float)
    y = np.arange(grid_pitch * n_pitches, dtype=float)

    xs, ys = np.meshgrid(x, y)
    Xs, Ys = np.meshgrid(x, y)

    # disp_amp = 1.0

    displacement_x = disp_amp * np.sin(n_periodes * np.pi * xs / xs.max())
    displacement_y = disp_amp * np.sin(n_periodes * np.pi * ys / ys.max())

    # Solve x = X + u(X) for X by solving X + u(X)-x = 0 using newtons method
    for i in range(20):
        Xs = Xs - (xs - Xs - disp_amp * np.sin(n_periodes * np.pi * Xs / xs.max())) / (
                -1 - disp_amp * np.cos(n_periodes * np.pi * Xs / xs.max()) / xs.max() / n_periodes)
        Ys = Ys - (ys - Ys - disp_amp * np.sin(n_periodes * np.pi * Ys / ys.max())) / (
                -1 - disp_amp * np.cos(n_periodes * np.pi * Ys / ys.max()) / ys.max() / n_periodes)

        errors_x = np.max(np.abs(Xs + disp_amp * np.sin(n_periodes * np.pi * Xs / xs.max()) - xs))
        errors_y = np.max(np.abs(Ys + disp_amp * np.sin(n_periodes * np.pi * Ys / xs.max()) - ys))
        # print(errors_x)
        # print(errors_y)

        if errors_x < tol and errors_y < tol:
            print("Coordinate correction converged in %i iterations" % i)
            break

    # plt.imshow(displacement_x)
    # plt.show()

    grid_undeformed = grid_grey_scales(xs, ys, grid_pitch)

    xs_disp = xs - displacement_x
    ys_disp = ys - displacement_y

    dev.append(np.max(np.abs(Xs - xs_disp)))

    print("Largest correction is: %f" % np.max(np.abs(Xs - xs_disp)))

grid_displaced_eulr_x = grid_grey_scales(xs_disp, ys_disp, grid_pitch)
grid_displaced_eulr_X = grid_grey_scales(Xs, Ys, grid_pitch)

phase_x, phase_y = detect_phase(grid_displaced_eulr_x, grid_pitch)
phase_X, phase_Y = detect_phase(grid_displaced_eulr_X, grid_pitch)
phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)



disp_x_from_phase = disp_from_phase(phase_x, phase_x0, grid_pitch, correct_phase=True, axis=0)
disp_X_from_phase = disp_from_phase(phase_X, phase_x0, grid_pitch, correct_phase=True, axis=0)
disp_y_from_phase = disp_from_phase(phase_y, phase_y0, grid_pitch, correct_phase=True, axis=1)

plt.plot(disp_x_from_phase[50,:],label="x")
plt.plot(disp_X_from_phase[50,:],label="X")
plt.legend()
plt.show()
#
# max_rel_error_x = np.max(np.abs(disp_x_from_phase - displacement_x[4 * grid_pitch:-4 * grid_pitch,
#                                                     4 * grid_pitch:-4 * grid_pitch]) / displacement_x[
#                                                                                        4 * grid_pitch:-4 * grid_pitch,
#                                                                                        4 * grid_pitch:-4 * grid_pitch])
# max_rel_error_y = np.max(np.abs(disp_y_from_phase - displacement_y[4 * grid_pitch:-4 * grid_pitch,
#                                                     4 * grid_pitch:-4 * grid_pitch]) / displacement_y[
#                                                                                        4 * grid_pitch:-4 * grid_pitch,
#                                                                                        4 * grid_pitch:-4 * grid_pitch])
#
# plt.figure()
# plt.title("Comparison of Eulerian and Lagrangian displacements")
# plt.plot(amps, dev)
# plt.ylabel("U(x)-U(X) [pix]")
# plt.xlabel("Displacement amplitude [pix]")
# plt.tight_layout()
# plt.show()
#
# grid_displaced_eulr = grid_grey_scales(Xs, Ys, grid_pitch)
# grid_displaced_lagrangian = deformed_grid(xs, ys, displacement_x, displacement_y, grid_pitch)
#
# plt.figure()
# plt.imshow(Xs - xs)
# plt.title("Lagrangian")
#
# plt.figure()
# plt.imshow(xs_disp - xs)
# plt.title("Eulerian")
#
# plt.figure()
# plt.imshow((Xs - xs) - (xs_disp - xs))
# plt.title("Difference")
# plt.show()
#
# plt.figure()
# plt.plot(displacement_x[100, :])
# plt.ylabel("U(X) [pix]")
# plt.xlabel("X [pix]")
# plt.twinx()
# plt.plot(((Xs - xs) - (xs_disp - xs))[100, :], color="red")
# plt.ylabel("U(X)-U(x) [pix]", color="red")
# plt.show()
#
# print("Peak error is:", np.max(np.abs((Xs - xs) - (xs_disp - xs))) / disp_amp)
