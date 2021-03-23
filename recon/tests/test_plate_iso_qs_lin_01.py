from unittest import TestCase
import recon
import numpy as np

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

        plate = recon.calculate_plate_stiffness(mat_E,mat_nu,0.0,plate_thick)


        win_size = 8
        bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]

        deflection = recon.analydisp.sinusoidal_load(press, plate_len_x, plate_len_y, bend_stiff)


        fields = recon.fieldStack_from_disp_func(deflection, n_pts_x, n_pts_y, plate_len_x, plate_len_y)
        # define piecewise virtual fields
        virtual_fields = recon.virtual_fields.Hermite16(win_size, dx)

        field = fields(0)

        recon_press = recon.solver.plate_iso_qs_lin(field,plate, virtual_fields,shift=True)
        correct_press = recon.analydisp.pressure_sinusoidal(press,n_pts_x,n_pts_y)[3:-4:,3:-4]
        error = rms_diff(recon_press, correct_press)
        if error/press >tol:
            self.fail("Reconstruction had a normalized RMS error of %f"%(error/press))




