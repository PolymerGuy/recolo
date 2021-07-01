from recon.deflectomerty.deflectometry import detect_phase, disp_from_phase
from recon.artificial_grid_deformation import harmonic_disp_field, make_grid
import numpy as np
import matplotlib.pyplot as plt

grid_pitch = 5
oversampling = 15

disp_amps = np.arange(0.1, 2.5, 0.2)
amp = []
amp_uncor = []
diff = []
for disp_amp in disp_amps:
    disp_period = 200
    disp_n_periodes = 1

    xs, ys, xs_disp, ys_disp, _, _ = harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                                         formulation="lagrangian")

    grid_undeformed = make_grid(xs, ys, grid_pitch, oversampling=oversampling, pixel_size=1)

    grid_displaced_eulr = make_grid(xs_disp, ys_disp, grid_pitch, oversampling=oversampling,
                                    pixel_size=1)

    phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)
    phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

    disp_x_from_phase, disp_y_from_phase = disp_from_phase(phase_x, phase_x0, phase_y, phase_y0,
                                                           grid_pitch, correct_phase=True)

    disp_x_from_phase_uncor, disp_y_from_phase_uncor = disp_from_phase(phase_x, phase_x0, phase_y,
                                                                       phase_y0,
                                                                       grid_pitch, correct_phase=False)

    peak_disp_x = np.max(np.abs(disp_x_from_phase))
    peak_disp_x_uncor = np.max(np.abs(disp_x_from_phase_uncor))
    peak_disp_y = np.max(np.abs(disp_y_from_phase))

    diff.append(np.max(np.abs(disp_x_from_phase_uncor - disp_x_from_phase)))
    amp.append(peak_disp_x)
    amp_uncor.append(peak_disp_x_uncor)


plt.plot(disp_amps, diff, label="Corrected")
plt.ylabel("Pixel correction [pix]")
plt.xlabel("Amplitude of sinusoidal displacement [pix]")
plt.title("Change in displacement magnitude due to phase-correction")
plt.tight_layout()
plt.show()
