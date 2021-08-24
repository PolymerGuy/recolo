from unittest import TestCase
from recon import kinematic_fields_from_deflections
import numpy as np


class TestDetermineKinematicFieldsFromDeflection(TestCase):
    def setUp(self):
        self.tol = 1e-6

        self.deflection_amp = 0.1
        n_pts_x = 400
        n_pts_y = 400
        pixel_size = 1.5
        time_ramp = np.arange(0, 1, 0.1) ** 2.
        sampling_rate = 1
        deflection_field = self.deflection_amp * np.ones((n_pts_x, n_pts_y))
        deflection_fields = deflection_field[np.newaxis, :, :] * time_ramp[:, np.newaxis, np.newaxis]

        self.field_stack = kinematic_fields_from_deflections(deflection_fields, pixel_size, sampling_rate=sampling_rate)

    def test_acceleration(self):



        correct_accelerations = self.deflection_amp * 0.2

        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.acceleration - correct_accelerations)) > self.tol:
                self.fail("For frame %i, the acceleration calculation is wrong by %f" % (
                i, np.max(np.abs(field.acceleration - correct_accelerations))))


    def test_curvature_xx_flat_field(self):


        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.curv_xx)) > self.tol:
                self.fail("Curvature was not zero")
    def test_curvature_xy_flat_field(self):


        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.curv_xy)) > self.tol:
                self.fail("Curvature was not zero")

    def test_curvature_yy_flat_field(self):


        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.curv_yy)) > self.tol:
                self.fail("Curvature was not zero")
    def test_slope_x_flat_field(self):
        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.slope_x)) > self.tol:
                self.fail("Curvature was not zero")

    def test_slope_y_flat_field(self):
        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.slope_y)) > self.tol:
                self.fail("Curvature was not zero")