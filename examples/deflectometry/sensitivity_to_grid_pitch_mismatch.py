# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath

sys.path.extend([abspath(".")])

import recon
import numpy as np
import matplotlib.pyplot as plt


def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1 - array2) ** 2.))


# The grid pitch on an image is not neccesarily known exactly which may introduce bias in the calculated displacement
# field. This example generates grid images with different grid pitches and quantifies the consequence in terms of
# displacement bias. The displacement field is here rigid body motion.

# Grid image settings
oversampling = 9
grid_pitch_assumed = 5.0
real_grid_pitches = np.arange(5.0, 5.1, 0.05)
grid_image_size = int(20 * grid_pitch_assumed)

# Displacement field description
displacement_x = 1.e-2
displacement_y = 1.e-2

peak_error_x = []
rms_error_x = []
peak_error_y = []
rms_error_y = []
for grid_pitch_real in real_grid_pitches:
    # Generate the coordinates fields corresponding to the harmonic displacement field
    xs, ys, Xs, Ys, _, _ = recon.artificial_grid_deformation.rigid_body_disp_field(displacement_x, displacement_y,
                                                                                   grid_image_size, grid_image_size)
    # Generate the grid images
    grid_undeformed = recon.artificial_grid_deformation.dotted_grid(xs, ys, grid_pitch_real, 1, oversampling)
    grid_displaced_eulr = recon.artificial_grid_deformation.dotted_grid(Xs, Ys, grid_pitch_real, 1, oversampling)

    # Calculate the phase fields
    phase_x, phase_y = recon.deflectomerty.detect_phase(grid_displaced_eulr, grid_pitch_assumed)
    phase_x0, phase_y0 = recon.deflectomerty.detect_phase(grid_undeformed, grid_pitch_assumed)

    # Calculate the displacements from the phase fields
    disp_x_from_phase, disp_y_from_phase = recon.deflectomerty.disp_fields_from_phases(phase_x, phase_x0, phase_y,
                                                                                       phase_y0,
                                                                                       grid_pitch=grid_pitch_assumed,
                                                                                       correct_phase=True)

    peak_error_x.append(np.max(np.abs(disp_x_from_phase - displacement_x)) / displacement_x)
    peak_error_y.append(np.max(np.abs(disp_y_from_phase - displacement_y)) / displacement_y)

    rms_error_x.append(rms_diff(disp_x_from_phase, displacement_x) / displacement_x)
    rms_error_y.append(rms_diff(disp_y_from_phase, displacement_y) / displacement_y)

# Plot the results
plt.plot(real_grid_pitches, peak_error_x)
plt.plot(real_grid_pitches, peak_error_y)
plt.ylabel("Peak relative displacement error [pix]")
plt.xlabel("Real grid pitch [pix]")
plt.twinx()
plt.plot(real_grid_pitches, rms_error_x, color="red")
plt.plot(real_grid_pitches, rms_error_y, color="red")
plt.ylabel("RMS relative displacement error [pix]", color="red")
plt.tight_layout()
plt.title("Displacement error caused by grid-pitch mismatch")
plt.show()
