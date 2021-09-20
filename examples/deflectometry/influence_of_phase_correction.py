# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import recolo
import numpy as np
import matplotlib.pyplot as plt

# The displacement field is determined from the phase fields but has to be corrected for finite displacements,
# see equation (4.15) in [1]  for reference. This script compares the displacement results with and without this
# correction.
#
# [1] Michel Grediac, Frédéric Sur, Benoît Blaysat. The grid method for in-plane displacement and
# strain measurement: a review and analysis. Strain, Wiley-Blackwell, 2016, 52 (3), pp.205-243.
# ff10.1111/str.12182ff. ffhal-01317145f

# Grid image settings
grid_pitch = 5
oversampling = 15

# Displacement field description
disp_amps = np.arange(0.1, 2.5, 0.2)
disp_period = 200
disp_n_periodes = 1

amp = []
amp_uncor = []
diff = []
for disp_amp in disp_amps:
    # Generate the coordinates fields corresponding to the harmonic displacement field
    xs, ys, Xs, Ys, _, _ = recolo.artificial_grid_deformation.harmonic_disp_field(disp_amp, disp_period, disp_n_periodes,
                                               formulation="lagrangian")

    # Generate the grid images
    grid_undeformed = recolo.artificial_grid_deformation.dotted_grid(xs, ys, grid_pitch, oversampling=oversampling, pixel_size=1)

    grid_displaced_eulr = recolo.artificial_grid_deformation.dotted_grid(Xs, Ys, grid_pitch, oversampling=oversampling,
                                      pixel_size=1)
    # Calculate the phase fields
    phase_x, phase_y = recolo.deflectomerty.detect_phase(grid_displaced_eulr, grid_pitch)
    phase_x0, phase_y0 = recolo.deflectomerty.detect_phase(grid_undeformed, grid_pitch)

    # Calculate the displacements from the phase fields
    disp_x_from_phase, disp_y_from_phase = recolo.deflectomerty.disp_fields_from_phases(phase_x, phase_x0, phase_y, phase_y0,
                                                                   grid_pitch, correct_phase=True)

    disp_x_from_phase_uncor, disp_y_from_phase_uncor = recolo.deflectomerty.disp_fields_from_phases(phase_x, phase_x0, phase_y,
                                                                               phase_y0,
                                                                               grid_pitch, correct_phase=False)

    peak_disp_x = np.max(np.abs(disp_x_from_phase))
    peak_disp_x_uncor = np.max(np.abs(disp_x_from_phase_uncor))
    peak_disp_y = np.max(np.abs(disp_y_from_phase))

    diff.append(np.max(np.abs(disp_x_from_phase_uncor - disp_x_from_phase)))
    amp.append(peak_disp_x)
    amp_uncor.append(peak_disp_x_uncor)

# Plot the results
plt.plot(disp_amps, diff, label="Corrected")
plt.ylabel("Pixel correction [pix]")
plt.xlabel("Amplitude of sinusoidal displacement [pix]")
plt.title("Change in displacement magnitude due to phase-correction")
plt.tight_layout()
plt.show()
