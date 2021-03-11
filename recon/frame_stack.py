from recon.diff_tools import dF_complex_x, dF_complex_y, ddF_complex_x, ddF_complex_xy, ddF_complex_y
import numpy as np
from .analydisp import pressure_sinusoidal

class  Frame_stack(object):
    def __init__(self,deflection,press,slopes,curvatures,acceleration,times):
        self.deflection = deflection
        self.press = press
        self.slope_x,self.slope_y = slopes
        self.curv_xx,self.curv_yy,self.curv_xy = curvatures
        self.acceleration = acceleration

    def __len__(self):
        return self.deflection.shape[0]
    def __call__(self,frame_id, *args, **kwargs):
        return Frame(self.deflection[frame_id],self.press[frame_id],(self.slope_x[frame_id],self.slope_y[frame_id]),(self.curv_xx[frame_id],self.curv_yy[frame_id],self.curv_xy[frame_id]),self.acceleration[frame_id])

    def filter_time(self,sigma):
        pass
    def filter_space(self,sigma):
        pass

class  Frame(object):
    def __init__(self,deflection,press,slopes,curvatures,acceleration):
        self.deflection = deflection
        self.press = press
        self.slope_x,self.slope_y = slopes
        self.curv_xx,self.curv_yy,self.curv_xy = curvatures
        self.acceleration = acceleration
        self.time = None





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
    return Frame_stack(deflection, press, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), np.zeros_like(deflection))


def field_from_displacement(disp_field, acceleration_field,times, plate_x, plate_y):

    npts_x,npts_y = disp_field.shape
    odx = plate_x/npts_x
    ody = plate_y/npts_y

    # calculate out-of-plane displacements
    xs, ys = np.meshgrid(np.linspace(0., 1., npts_x), np.linspace(0., 1., npts_y))
    deflection = disp_field


    #press = pressure_sinusoidal(100, xs, ys)
    press = np.zeros_like(deflection)

    # calculate slopes
    slope_x,slope_y = np.gradient(-deflection, odx, ody)

    ###
    # calculate curvatures
    aux_k_xx, aux_k_s12 = np.gradient(slope_x, odx, ody)
    aux_k_s21, aux_k_yy = np.gradient(slope_y, odx, ody)
    aux_k_xy = .5*(aux_k_s12 + aux_k_s21)

    slope_x = slope_x[1:-1, 1:-1]
    slope_y = slope_y[1:-1, 1:-1]

    curv_xx = aux_k_xx[1:-1, 1:-1]
    curv_yy = aux_k_yy[1:-1, 1:-1]
    curv_xy = aux_k_xy[1:-1, 1:-1]

    deflection = disp_field[1:-1, 1:-1]
    accel_field= acceleration_field[1:-1, 1:-1]





    return Frame_stack(deflection, press, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), accel_field, times)
