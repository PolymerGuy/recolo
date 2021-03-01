import numpy as np


def sinusoidal_load(peak_press, plate_x, plate_y, D_p):
    def field(norm_x, norm_y):
        return peak_press / (np.pi ** 4. * D_p) / (((1. / plate_x) ** 2. + (1. / plate_y) ** 2.) ** 2.) * np.sin(
            np.pi * norm_x) * np.sin(np.pi * norm_y)
    return field

def pressure_sinusoidal(peak_press, norm_x, norm_y):
    return peak_press * np.sin(np.pi * norm_x) * np.sin(np.pi * norm_y)


def deflection_constant(peak_press, plate_x, plate_y, D_p):
    def field(norm_x, norm_y):
        ow_p = np.zeros_like(norm_x)
        for i in range(1, 40):
            for j in range(1, 40):
                ow_p = ow_p + (-4. * peak_press / (np.pi ** 6. * D_p * i * j) * (1. - np.cos(i * np.pi)) \
                               * (1. - np.cos(j * np.pi)) / (((i / plate_x) ** 2. + (j / plate_y) ** 2.) ** 2.) \
                               * np.sin(np.pi * i * norm_x) * np.sin(np.pi * j * norm_y))
        return ow_p
    return field


def deflection_constant2(peak_press, plate_x, plate_y, D_p):
    def field(norm_x,norm_y):
        n_terms = 20
        sol = np.zeros_like(norm_x)
        # Only odd components
        for m in np.arange(1., n_terms * 2., 2.):
            for n in np.arange(1., n_terms * 2., 2.):
                sol = sol + (1. / (m * n * ((m ** 2. / plate_x ** 2.) + (n ** 2. / plate_y ** 2.)) ** 2.)) * np.sin(
                    m * np.pi * norm_x) * np.sin(n * np.pi * norm_y)
        return 16. * peak_press / (np.pi ** 6. * D_p) * sol
    return field



