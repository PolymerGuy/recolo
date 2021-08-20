import recon
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('science')

# plate and model parameters
mat_E = 210.e9  # Young's modulus [Pa]
mat_nu = 0.33  # Poisson's ratio []
density = 7700
plate_thick = 5e-3
plate = recon.calculate_plate_stiffness(mat_E, mat_nu, density, plate_thick)

deflecto_upscale = 4
mirror_grid_dist = 500.
win_size = 10
grid_pitch = 5.  # pixels

# Load Abaqus data
abq_sim_fields = recon.load_abaqus_rpts("/home/sindreno/Rene/testfolder/fields/")
pixel_size = abq_sim_fields.pixel_size_x# / deflecto_upscale


if False:
# Deflectometry
    slopes_x = []
    slopes_y = []
    undeformed_grid = recon.artificial_grid_deformation.deform_grid_from_deflection(abq_sim_fields.disp_fields[0, :, :],
                                                                                    abq_sim_fields.pixel_size_x,
                                                                                    mirror_grid_dist,
                                                                                    grid_pitch,
                                                                                    upscale=deflecto_upscale)
    for disp_field in abq_sim_fields.disp_fields:
        deformed_grid = recon.artificial_grid_deformation.deform_grid_from_deflection(disp_field,
                                                                                      abq_sim_fields.pixel_size_x,
                                                                                      mirror_grid_dist,
                                                                                      grid_pitch,
                                                                                      upscale=deflecto_upscale)
        slope_x, slope_y = recon.deflectomerty.deflectometry_from_grid(undeformed_grid, deformed_grid, mirror_grid_dist,
                                                                       grid_pitch)
        slopes_x.append(slope_x)
        slopes_y.append(slope_y)

    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)

slopes_x,slopes_y = np.gradient(abq_sim_fields.disp_fields,pixel_size,axis=(1,2))
# Integrate slopes from deflectometry to get deflection fields
disp_fields = recon.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size,
                                                       zero_at="bottom corners", zero_at_size=1,
                                                       extrapolate_edge=0, filter_sigma=0, downsample=1)


# Kinematic fields from deflection field
sampling_rate = 1. / (abq_sim_fields.times[1] - abq_sim_fields.times[0])
fields = recon.kinematic_fields_from_deflections(disp_fields, pixel_size, sampling_rate)

# Reconstruct pressure using the virtual fields method
virtual_field = recon.virtual_fields.Hermite16(win_size, pixel_size)
pressure_fields = [recon.solver.pressure_elastic_thin_plate(field, plate, virtual_field) for field in fields]

pressure_fields = np.array(pressure_fields)
center = int(pressure_fields.shape[1] / 2)
# Plot the results
# Correct
times = np.array([0.0,0.00005,0.000051,0.0003,0.001])*1000
pressures = np.array([0.0,0.0,1.0,0.0,0.0]) * 1e5
plt.plot(times,pressures,'--')

plt.plot(abq_sim_fields.times * 1000., pressure_fields[:, center, center])

plt.xlim(left=0.000, right=0.3)
plt.ylim(top=110000, bottom=-15)
plt.xlabel("Time [ms]")
plt.ylabel(r"Overpressure [kPa]")

plt.legend(frameon=False)
plt.tight_layout()
plt.show()
