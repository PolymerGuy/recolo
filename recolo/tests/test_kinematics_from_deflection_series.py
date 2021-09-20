from unittest import TestCase
from recolo import kinematic_fields_from_deflections
import numpy as np


class TestConstantDeflection(TestCase):
    def setUp(self):
        self.tol = 1e-6

        self.acceleration = 1.7
        n_pts_x = 400
        n_pts_y = 400
        pixel_size = 1.5
        time_ramp = 0.5 * self.acceleration * np.arange(0, 100, 1) ** 2.
        sampling_rate = 1
        deflection_field = np.ones((n_pts_x, n_pts_y))
        deflection_fields = deflection_field[np.newaxis, :, :] * time_ramp[:, np.newaxis, np.newaxis]

        self.field_stack = kinematic_fields_from_deflections(deflection_fields, pixel_size, sampling_rate=sampling_rate)

    def test_acceleration(self):
        for i, field in enumerate(self.field_stack):
            # We need to skip the first and last values as they are affected by the edges
            if i > 5 and i < 95:
                if np.max(np.abs(field.acceleration - self.acceleration)) > self.tol:
                    self.fail("For frame %i, the acceleration calculation is wrong by %f" % (
                        i, np.max(np.abs(field.acceleration - self.acceleration))))

    def test_curvature_xx(self):

        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.curv_xx)) > self.tol:
                self.fail("Curvature was not zero")

    def test_curvature_xy(self):

        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.curv_xy)) > self.tol:
                self.fail("Curvature was not zero")

    def test_curvature_yy(self):

        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.curv_yy)) > self.tol:
                self.fail("Curvature was not zero")

    def test_slope_x(self):
        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.slope_x)) > self.tol:
                self.fail("Curvature was not zero")

    def test_slope_y(self):
        for i, field in enumerate(self.field_stack):
            if np.max(np.abs(field.slope_y)) > self.tol:
                self.fail("Curvature was not zero")


class TestHalfSineDeflection(TestCase):
    def setUp(self):
        self.tol = 1e-6
        self.curv_rel_tol = 1e-2
        self.acceleration = 1.7
        n_pts_x = 200
        n_pts_y = 200
        pixel_size = 1
        time_ramp = 0.5 * self.acceleration * np.arange(0, 100, 1) ** 2.
        sampling_rate = 1
        xs,ys = np.meshgrid(np.linspace(0,1,n_pts_x),np.linspace(0,1,n_pts_y))

        deflection_field = np.sin(np.pi * xs) * np.sin(np.pi * ys)

        deflection_fields = deflection_field[np.newaxis, :, :] * time_ramp[:, np.newaxis, np.newaxis]

        self.acceleration_field = self.acceleration * deflection_field
        # Note the minus sign which is used for compliance with the formulation of the rotational degrees of freedom
        self.slope_x_fields = -(np.cos(np.pi * xs) * np.sin(np.pi * ys) * (np.pi)/(n_pts_x))[np.newaxis, :, :] * time_ramp[:, np.newaxis, np.newaxis]
        self.slope_y_fields = -(np.sin(np.pi * xs) * np.cos(np.pi * ys) * (np.pi)/(n_pts_x))[np.newaxis, :, :] * time_ramp[:, np.newaxis, np.newaxis]

        # Note the minus sign which is used for compliance with the formulation of the rotational degrees of freedom
        self.curv_xx_fields = -(- np.sin(np.pi * xs) * np.sin(np.pi * ys) * (np.pi**2.)/(n_pts_x**2.))[np.newaxis, :, :] * time_ramp[:, np.newaxis, np.newaxis]
        self.curv_yy_fields = -(- np.sin(np.pi * xs) * np.sin(np.pi * ys) * (np.pi**2.)/(n_pts_y**2.))[np.newaxis, :, :] * time_ramp[:, np.newaxis, np.newaxis]
        self.curv_xy_fields = -(np.cos(np.pi * xs) * np.cos(np.pi * ys) * (np.pi**2.)/(n_pts_y**2.))[np.newaxis, :, :] * time_ramp[:, np.newaxis, np.newaxis]

        self.field_stack = kinematic_fields_from_deflections(deflection_fields, pixel_size, sampling_rate=sampling_rate)

    def test_acceleration(self):
        for i, field in enumerate(self.field_stack):
            # We need to skip the first and last values as they are affected by the edges
            if i > 5 and i < 95:
                if np.max(np.abs(field.acceleration - self.acceleration_field)) > self.tol:
                    self.fail("For frame %i, the acceleration calculation is wrong by %f" % (
                        i, np.max(np.abs(field.acceleration - self.acceleration))))

    def test_curvature_xx(self):

        for i, field in enumerate(self.field_stack):
            correct_curv = self.curv_xx_fields[i,:,:]
            correct_curv_peak_amp = np.abs(np.min(correct_curv))
            calculated_curv = field.curv_yy

            peak_relative_error = np.max(np.abs(correct_curv-calculated_curv)/correct_curv_peak_amp)
            if peak_relative_error > self.curv_rel_tol:
                self.fail("Relative error of %f was found for curv_xx"%peak_relative_error)

    def test_curvature_xy_(self):

        for i, field in enumerate(self.field_stack):
            correct_curv = self.curv_xy_fields[i,:,:]
            correct_curv_peak_amp = np.abs(np.min(correct_curv))
            calculated_curv = field.curv_xy

            peak_relative_error = np.max(np.abs(correct_curv-calculated_curv)/correct_curv_peak_amp)
            if peak_relative_error > self.curv_rel_tol:
                self.fail("Relative error of %f was found for curv_xy"%peak_relative_error)

    def test_curvature_yy(self):

        for i, field in enumerate(self.field_stack):
            correct_curv = self.curv_yy_fields[i,:,:]
            correct_curv_peak_amp = np.abs(np.min(correct_curv))
            calculated_curv = field.curv_xx

            peak_relative_error = np.max(np.abs(correct_curv-calculated_curv)/correct_curv_peak_amp)
            if peak_relative_error > self.curv_rel_tol:
                self.fail("Relative error of %f was found for curv_yy"%peak_relative_error)

    def test_slope_x(self):
        # Note: slope_x is the gradient along y
        for i, field in enumerate(self.field_stack):
            correct_slope = self.slope_x_fields[i,:,:]
            correct_slope_peak_amp = np.abs(np.max(correct_slope))
            calculated_slope = field.slope_y
            peak_relative_error = np.max(np.abs(correct_slope-calculated_slope)/correct_slope_peak_amp)
            if peak_relative_error > self.curv_rel_tol:
                self.fail("Relative error of %f was found for slope_x"%peak_relative_error)

    def test_slope_y(self):
        # Note: slope_y is the gradient along x
        for i, field in enumerate(self.field_stack):
            correct_slope = self.slope_y_fields[i,:,:]
            correct_slope_peak_amp = np.abs(np.max(correct_slope))
            calculated_slope = field.slope_x
            peak_relative_error = np.max(np.abs(correct_slope-calculated_slope)/correct_slope_peak_amp)
            if peak_relative_error > self.curv_rel_tol:
                self.fail("Relative error of %f was found for slope_y"%peak_relative_error)