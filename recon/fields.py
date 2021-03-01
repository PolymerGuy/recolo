from recon.diff_tools import dF_complex_x, dF_complex_y, ddF_complex_x, ddF_complex_xy, ddF_complex_y
import numpy as np
from .analydisp import pressure_sinusoidal

class  Fields(object):
    def __init__(self,deflection,press,slopes,curvatures):
        self.deflection = deflection
        self.press = press
        self.slope_x,self.slope_y = slopes
        self.curv_xx,self.curv_yy,self.curv_xy = curvatures

def field_from_disp_func(disp_func,npts_x, npts_y, plate_x, plate_y):
    # calculate out-of-plane displacements
    xs, ys = np.meshgrid(np.linspace(0., 1., npts_x), np.linspace(0., 1., npts_y))
    deflection = disp_func(xs, ys)

    press = pressure_sinusoidal(100, xs, ys)
    #press = np.zeros_like(deflection)

    # Calculate slopes
    slope_x = -dF_complex_y(disp_func, xs, ys) / plate_x
    slope_y = -dF_complex_x(disp_func, xs, ys) / plate_y

    # calculate curvatures
    curv_yy = (-ddF_complex_x(disp_func, xs, ys ) / (plate_x ** 2.))
    curv_xx = (-ddF_complex_y(disp_func, xs, ys ) / (plate_x ** 2.))
    curv_xy = (-ddF_complex_xy(disp_func, xs, ys) / (plate_x ** 2.))
    return Fields(deflection, press, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy))

