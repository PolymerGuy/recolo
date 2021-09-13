# This allows for running the example when the repo has been cloned
import sys
from os.path import abspath
sys.path.extend([abspath(".")])

import recon
import numpy as np
import matplotlib.pyplot as plt
import os
cwd = os.path.dirname(os.path.realpath(__file__))
# Minimal example of pressure load reconstruction based on input from Abaqus. The mesh is very coarse 61x61 elements,
# which requires the use of a small window size for the pressure reconstruction.

# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.33  # Poisson's ratio []
density = 7700
plate_thick = 5e-3
plate = recon.make_plate(mat_E, mat_nu, density, plate_thick)

# Reconstruction settings
win_size = 6

# Load Abaqus data
abq_sim_fields = recon.load_abaqus_rpts(os.path.join(cwd,"AbaqusExampleData/"))

# Kinematic fields from deflection field
kin_fields = recon.kinematic_fields_from_deflections(abq_sim_fields.disp_fields,
                                                     pixel_size=abq_sim_fields.pixel_size_x,
                                                     sampling_rate=abq_sim_fields.sampling_rate,
                                                     acceleration_field=abq_sim_fields.accel_fields)

# Reconstruct pressure using the virtual fields method
virtual_field = recon.virtual_fields.Hermite16(win_size, abq_sim_fields.pixel_size_x)
pressure_fields = np.array([recon.solver_VFM.calc_pressure_thin_elastic_plate(field, plate, virtual_field) for field in kin_fields])

# Plot the results
# Correct pressure history used in the Abaqus simulation
times = np.array([0.0, 0.00005, 0.00010, 0.0003, 0.001]) * 1000
pressures = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) * 1e5
plt.plot(times, pressures, '-', label="Correct pressure")

# Reconstructed pressure from VFM
center_pixel = int(pressure_fields.shape[1] / 2)
plt.plot(abq_sim_fields.times * 1000., pressure_fields[:, center_pixel, center_pixel], "-o", label="Reconstructed pressure")

plt.xlim(left=0.000, right=0.3)
plt.ylim(top=110000, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
