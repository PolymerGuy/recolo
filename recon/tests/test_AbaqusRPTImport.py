from unittest import TestCase
from recon.data_import import load_abaqus_rpts


class TestReadAbaqusField(TestCase):
    def setUp(self):
        self.abaqus_fields = load_abaqus_rpts("./ExampleAbaqusRPT")

    def test_check_shapes(self):
        shape_deflection = self.abaqus_fields.disp_fields.shape
        shape_accelerations = self.abaqus_fields.accel_fields.shape
        shape_time = self.abaqus_fields.times.shape


        if shape_deflection != shape_accelerations:
            self.fail("The deflection and acceleration field has different shapes")
        if shape_deflection[0] != shape_time[0]:
            self.fail(("The time axis is not of the same length (%i) as the number of frames (%i)"%(shape_deflection[0],shape_time[0])))

