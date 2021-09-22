Experiment with impact hammer
=============================

This example presents the force reconstruction from an experiment where an impact hammer is used to knock on a steel plate.
The force from the hammer onto the plate is reconstructed from the kinematics fields  measured by deflectometry.

Due to the large size of the data set, the data is hosted on https://dataverse.no/ and can either be downloaded manually
or using this toolkit.

Let's now go through the necessary steps for force reconstruction based on deflectometry.
First, we need to import the Recolo toolkit::

    import recolo

The experimental data is conveniently downloaded and accessed via the ImpactHammerExperiment class::

    exp_data = recolo.demoData.ImpactHammer()

After the download has completed, the force measurements can be accessed as::

    hammer_force, hammer_time = exp_data.hammer_data()

The experiment was performed on a 300 x 300 mm rectangular plate with the following properties::

     mat_E = 210.e9  # Young's modulus [Pa]
     mat_nu = 0.33  # Poisson's ratio []
     density = 7700
     plate_thick = 5e-3

The stiffness of the plate is calculated as::

     plate = recolo.make_plate(mat_E, mat_nu, density, plate_thick)

The experimental setup has to be described by::

    grid_pitch = 7.0  # pixels
    grid_pitch_len = 2.5 / 1000.  # m
    mirror_grid_distance = 1.63  # m
    pixel_size_on_mirror = grid_pitch_len / grid_pitch * 0.5

We now set the pressure reconstruction window size::

     win_size = 30

In this case, the deflection fields from Abaqus are used to generate grid images with the corresponding distortion.
The grid images are then used as input to deflectomerty and the slope fields of the plate are determined::

    slopes_y, slopes_x = recolo.deflectomerty.slopes_from_images(exp_data.path_to_imgs, grid_pitch, mirror_grid_distance,
                                                                ref_img_ids=ref_img_ids, only_img_ids=use_imgs,
                                                                crop=(45, 757,0,-1),window="gaussian",correct_phase=False)

The slope fields are then integrated to determine the deflection fields::

     # Integrate slopes to get deflection fields
     disp_fields = recolo.slope_integration.disp_from_slopes(slopes_x, slopes_y, pixel_size,
                                                            zero_at="bottom corners", zero_at_size=5,
                                                            extrapolate_edge=0, downsample=1)

Based on these fields, the kinematic fields (slopes and curvatures) are calculated::

     kin_fields = recolo.kinematic_fields_from_deflections(disp_fields, pixel_size_on_mirror,
                                                     abq_sim_fields.sampling_rate,filter_space_sigma=10)

Now, the pressure reconstuction can be initiated. First we define the Hermite16 virtual fields::

     virtual_field = recolo.virtual_fields.Hermite16(win_size, pixel_size_on_mirror)

and initialize the pressure reconstruction::

     pressure_fields = np.array(
     [recolo.solver_VFM.pressure_elastic_thin_plate(field, plate, virtual_field)
                                                      for field in kin_fields])


The results can then be visualized::

     import matplotlib.pyplot as plt
     # Plot the correct pressure in the center of the plate
     times = np.array([0.0, 0.00005, 0.00010, 0.0003, 0.001]) * 1000
     pressures = np.array([0.0, 0.0, 1.0, 0.0, 0.0]) * 1e5
     plt.plot(times, pressures, '-', label="Correct pressure")

     # Plot the coreconstructed pressure in the center of the plate
     center = int(pressure_fields.shape[1] / 2)
     plt.plot(abq_sim_fields.times * 1000., pressure_fields[:, center, center], "-o",label="Reconstructed pressure")

     plt.xlim(left=0.000, right=0.3)
     plt.ylim(top=110000, bottom=-15)
     plt.xlabel("Time [ms]")
     plt.ylabel(r"Overpressure [kPa]")

     plt.legend(frameon=False)
     plt.tight_layout()
     plt.show()

The resulting plot looks like this:

.. image:: ./figures/minimalExamplePressure.png
   :scale: 80 %
   :alt: The results
   :align: center

