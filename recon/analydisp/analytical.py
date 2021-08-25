import numpy as np


def sinusoidal_load(peak_press, plate_x, plate_y, D_p):
    def field(norm_x, norm_y):
        return peak_press / (np.pi ** 4. * D_p) / (((1. / plate_x) ** 2. + (1. / plate_y) ** 2.) ** 2.) * np.sin(
            np.pi * norm_x) * np.sin(np.pi * norm_y)
    return field

def pressure_sinusoidal(peak_press,n_pts_x,n_pts_y):
    norm_x,norm_y = np.meshgrid(np.linspace(0,1,n_pts_x),np.linspace(0,1,n_pts_y))

    return peak_press * np.sin(np.pi * norm_x) * np.sin(np.pi * norm_y)
