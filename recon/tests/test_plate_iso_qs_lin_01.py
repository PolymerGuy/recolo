from unittest import TestCase
from recon  import field_from_disp_func
from recon.analydisp import sinusoidal_load
from recon import plate_iso_qs_lin
from recon import Hermite16
import numpy as np
import matplotlib.pyplot as plt

def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1-array2))**2.)

class Test_FullStaticReconstruction(TestCase):




    def test_analytical_sinusoidal(self):
        # Tollerance set to 1 percent
        tol = 1e-2

        mat_E = 70.e9  # Young's modulus [Pa]
        mat_nu = 0.23  # Poisson's ratio []
        n_pts_x = 101
        n_pts_y = 101
        plate_len_x = 0.2
        plate_len_y = 0.2
        plate_thick = 1e-3
        press = 100.
        dx = plate_len_x / float(n_pts_x)
        dy = plate_len_y / float(n_pts_y)
        ###
        mat_D = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu))  # flexural rigidity [N m]
        mat_D11 = (plate_thick ** 3.) / 12. * mat_E / (1. - mat_nu ** 2.)
        mat_D12 = (plate_thick ** 3.) / 12. * mat_E * mat_nu / (1. - mat_nu ** 2.)



        win_size = 8
        bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]

        deflection = sinusoidal_load(press, plate_len_x, plate_len_y, bend_stiff)

        fields = field_from_disp_func(deflection, n_pts_x, n_pts_y, plate_len_x, plate_len_y)

        # define piecewise virtual fields
        virtual_fields = Hermite16(win_size, float(dx * win_size))

        recon_press,_ = plate_iso_qs_lin(win_size, fields, mat_D11, mat_D12, virtual_fields)
        error = rms_diff(recon_press, fields._press_)
        if error/press >tol:
            self.fail("Reconstruction had a normalized RMS error of %f"%(error/press))




