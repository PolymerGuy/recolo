from unittest import TestCase
import numpy as np
import recolo

def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1-array2))**2.)

def deflection_due_to_sinus_load(peak_press, plate_x, plate_y, D_p):
    def field(norm_x, norm_y):
        return peak_press / (np.pi ** 4. * D_p) / (((1. / plate_x) ** 2. + (1. / plate_y) ** 2.) ** 2.) * np.sin(
            np.pi * norm_x) * np.sin(np.pi * norm_y)
    return field

def pressure_sinusoidal(peak_press,n_pts_x,n_pts_y):
    norm_x,norm_y = np.meshgrid(np.linspace(0,1,n_pts_x),np.linspace(0,1,n_pts_y))

    return peak_press * np.sin(np.pi * norm_x) * np.sin(np.pi * norm_y)



class Test_FullStaticReconstruction(TestCase):

    def test_analytical_sinusoidal(self):
        # Tolerance set to 1 percent
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

        plate = recolo.make_plate(mat_E, mat_nu, 0.0, plate_thick)


        win_size = 8
        bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]

        deflection = deflection_due_to_sinus_load(press, plate_len_x, plate_len_y, bend_stiff)


        fields = recolo.fieldStack_from_disp_func(deflection, n_pts_x, n_pts_y, plate_len_x, plate_len_y)
        # define piecewise virtual fields
        virtual_fields = recolo.virtual_fields.Hermite16(win_size, dx)

        field = fields(0)

        recon_press = recolo.solver_VFM.calc_pressure_thin_elastic_plate(field, plate, virtual_fields, shift=True)
        correct_press = pressure_sinusoidal(press,n_pts_x,n_pts_y)[3:-4:,3:-4]
        error = rms_diff(recon_press, correct_press)
        if error/press >tol:
            self.fail("Reconstruction had a normalized RMS error of %f"%(error/press))




