from recon.deflectomerty.deflectometry import detect_phase, disp_from_phase
from recon.artificial_grid_deformation import make_dotted_grid
import numpy as np
import matplotlib.pyplot as plt

def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2) ** 2.))



# The grid pitch on the actual image is not exactly known and we need to make sure that
# the phase detection works even when we don't hit the right value exactly.
rel_error_tol = 1e-2

oversampling = 9

grid_pitch_assumed = 5.0

real_grid_pitches = np.arange(5.0,5.1,0.05)
peak_error_x = []
rms_error_x = []
peak_error_y = []
rms_error_y = []
for grid_pitch_real in real_grid_pitches:
    n_pitches = 20

    displacement_x = 1.e-2
    displacement_y = 1.e-2

    x = np.arange(grid_pitch_assumed * n_pitches, dtype=float)
    y = np.arange(grid_pitch_assumed * n_pitches, dtype=float)

    xs, ys = np.meshgrid(x, y)

    grid_undeformed = make_dotted_grid(xs, ys, grid_pitch_real, 1, oversampling)

    xs_disp = xs - displacement_x
    ys_disp = ys - displacement_y

    grid_displaced_eulr = make_dotted_grid(xs_disp, ys_disp, grid_pitch_real, 1, oversampling)

    phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch_assumed)
    phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch_assumed)

    disp_x_from_phase, disp_y_from_phase = disp_from_phase(phase_x, phase_x0, phase_y, phase_y0,
                                                           grid_pitch_assumed,
                                                           correct_phase=True)


    peak_error_x.append(np.max(np.abs(disp_x_from_phase-displacement_x))/displacement_x)
    peak_error_y.append(np.max(np.abs(disp_y_from_phase-displacement_y))/displacement_y)

    rms_error_x.append(rms_diff(disp_x_from_phase,displacement_x)/displacement_x)
    rms_error_y.append(rms_diff(disp_y_from_phase,displacement_y)/displacement_y)

plt.plot(real_grid_pitches,peak_error_x)
plt.plot(real_grid_pitches,peak_error_y)
plt.ylabel("Peak relative displacement error [pix]")
plt.xlabel("Real grid pitch [pix]")
plt.twinx()
plt.plot(real_grid_pitches,rms_error_x,color="red")
plt.plot(real_grid_pitches,rms_error_y,color="red")
plt.ylabel("RMS relative displacement error [pix]",color="red")
plt.tight_layout()
plt.title("Displacement error caused by grid-pitch mismatch")
plt.show()