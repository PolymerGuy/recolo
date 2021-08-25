from unittest import TestCase
from recon.data_import import load_abaqus_rpts
import pathlib
import os
cwd = pathlib.Path(__file__).parent.resolve()


class TestReadAbaqusField(TestCase):
    def setUp(self):
        path_to_rpts = os.path.join(cwd,"ExampleAbaqusRPT/")

        self.abaqus_fields = load_abaqus_rpts(path_to_rpts)

    def test_check_shapes(self):
        shape_deflection = self.abaqus_fields.disp_fields.shape
        shape_accelerations = self.abaqus_fields.accel_fields.shape
        shape_time = self.abaqus_fields.times.shape


        if shape_deflection != shape_accelerations:
            self.fail("The deflection and acceleration field has different shapes")
        if shape_deflection[0] != shape_time[0]:
            self.fail(("The time axis is not of the same length (%i) as the number of frames (%i)"%(shape_deflection[0],shape_time[0])))

