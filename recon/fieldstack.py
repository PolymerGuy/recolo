from recon.math_tools.complex_step_diff import dF_complex_x, dF_complex_y, ddF_complex_x, ddF_complex_xy, ddF_complex_y
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from copy import copy
from collections import namedtuple
import logging


class FieldStack(object):
    def __init__(self, deflection, slopes, curvatures, acceleration, times):
        self._deflection_ = deflection
        self._slope_x_, self._slope_y_ = slopes
        self._curv_xx_, self._curv_yy_, self._curv_xy_ = curvatures
        self._acceleration_ = acceleration
        self._times_ = times

        self._iter_counter_ = 0

    def __len__(self):
        return len(self._deflection_)

    def __call__(self, frame_id, *args, **kwargs):
        return Fields(self._deflection_[frame_id], self._slope_x_[frame_id], self._slope_y_[frame_id],
                      self._curv_xx_[frame_id], self._curv_yy_[frame_id], self._curv_xy_[frame_id],
                      self._acceleration_[frame_id], self._times_[frame_id])

    def __iter__(self):
        return copy(self)

    def __next__(self):
        num = self._iter_counter_
        self._iter_counter_ += 1
        if self._iter_counter_ > self.__len__():
            raise StopIteration
        else:
            return self.__call__(num)


    def shape(self):
        return np.shape(self._deflection_)


Fields = namedtuple("Fields",
                    ["deflection", "slope_x", "slope_y", "curv_xx", "curv_yy", "curv_xy", "acceleration", "time"])


def kinematic_fields_from_deflections(defl_fields, pixel_size, sampling_rate, acceleration_field=None, filter_space_sigma=None,
                                      filter_time_sigma=None):
    """
    Calculate kinematic fields from a series of deflection fields.
    The following fields are calculated are:
        * Slopes
        * Curvatures
        * Out of plane acceleration

    Parameters
    ----------
    defl_fields : ndarray
        The deflection fields with shape [frame,x,y]
    pixel_size : float
        The physical pixel size
    sampling_rate : float
        The sampling rate at which the fields are acquired
    acceleration_field : ndarray (Optional)
        The deflection fields with shape [frame,x,y]
        If given, the acceleration fields are not determined by differentiation
        of the deflection fields along the time axis.
    filter_space_sigma : float
        The standard deviation of the gaussian low-pass filter used to filter the deflection fields
        spatially prior to differentiation.
    filter_time_sigma : float
        The standard deviation of the gaussian low-pass filter used to filter the deflection fields
        temporally prior to differentiation.
    Returns
    -------
    fieldstack : FieldStack
        The kinematic fields
    """
    logger = logging.getLogger(__name__)
    # Copy to make in-place operations safe
    disp_fields = copy(defl_fields)

    n_times, n_pts_x, n_pts_y = disp_fields.shape

    times = np.arange(n_times) * 1. / sampling_rate
    field_len_x = n_pts_x * pixel_size
    field_len_y = n_pts_y * pixel_size

    if filter_time_sigma:
        logger.info("Filtering in time with sigma=%f" % float(filter_time_sigma))
        disp_fields = gaussian_filter1d(disp_fields, sigma=filter_time_sigma, axis=0, mode="nearest")

    if filter_space_sigma:
        for i in range(len(disp_fields)):
            logger.info("Filtering frame %i with a gussian filter with a standard deviation of %f" %(i,filter_space_sigma))
            disp_fields[i, :, :] = gaussian_filter(disp_fields[i, :, :], sigma=filter_space_sigma)

    if acceleration_field is not None:
        logger.info("Acceleration fields were given by the user and does not correspond to filtered displacements")
        return fieldStack_from_disp_fields(disp_fields, acceleration_field, times, field_len_x, field_len_y)
    else:
        return fieldStack_from_disp_fields(disp_fields, None, times, field_len_x, field_len_y)


def fieldStack_from_disp_func(disp_func, npts_x, npts_y, plate_len_x, plate_len_y):
    """
    Make a FielsStack object from a function describing the deflection field.

    The resulting stack contains a single frame.

    Parameters
    ----------
    disp_func : func
        The deflection field as a function on the form deflection = func(x,y)
    npts_x : int
        The number of evaluated points along the x-axis
    npts_y : int
        The number of evaluated points along the y-axis
    plate_len_x : float
        The plate length along the x-axis
    plate_len_y : float
        The plate length along the y-axis

    Returns
    -------
    field_stack : FieldStack
        The field stack
    """
    # calculate out-of-plane displacements
    xs, ys = np.meshgrid(np.linspace(0., 1., npts_x), np.linspace(0., 1., npts_y))
    deflection = disp_func(xs, ys)

    # Calculate slopes
    slope_x = -dF_complex_y(disp_func, xs, ys) / plate_len_x
    slope_y = -dF_complex_x(disp_func, xs, ys) / plate_len_y

    # calculate curvatures
    curv_yy = (-ddF_complex_x(disp_func, xs, ys) / (plate_len_x ** 2.))
    curv_xx = (-ddF_complex_y(disp_func, xs, ys) / (plate_len_x ** 2.))
    curv_xy = (-ddF_complex_xy(disp_func, xs, ys) / (plate_len_x ** 2.))

    # Add an initial axis to make this the first frame in the stack
    deflection = deflection[np.newaxis,:,:]
    slope_x = slope_x[np.newaxis,:,:]
    slope_y = slope_y[np.newaxis,:,:]
    curv_yy = curv_yy[np.newaxis,:,:]
    curv_xx = curv_xx[np.newaxis,:,:]
    curv_xy = curv_xy[np.newaxis,:,:]

    return FieldStack(deflection, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), np.zeros_like(deflection),[0])


def fieldStack_from_disp_fields(disp_fields, acceleration_fields, times, plate_len_x, plate_len_y):
    """
    Make a FielsStack object from deflection fields.

    Parameters
    ----------
    disp_fields : ndarray
        The deflection fields with shape (n_frames,x,y)
    acceleration_fields : ndarray, None
        The acceleration fields with shape (n_frames,x,y).
        If "None", the accelerations are calculated from the displacements.
    times : ndarray
        The times at which the frames are sampled. Has the shape (n_frames)
    plate_len_x : float
        The plate length along the x-axis
    plate_len_y : float
        The plate length along the y-axis

    Returns
    -------
    field_stack : FieldStack
        The field stack
    """


    n_frames = disp_fields.shape[0]
    deflection = []

    slopes_x = []
    slopes_y = []
    curv_xx = []
    curv_yy = []
    curv_xy = []

    for i in range(n_frames):
        disp_field = disp_fields[i]
        npts_x, npts_y = disp_field.shape
        pixel_size_x = plate_len_x / npts_x
        pixels_size_y = plate_len_y / npts_y

        # calculate slopes
        slope_x, slope_y = np.gradient(-disp_field, pixel_size_x, pixels_size_y)

        # calculate curvatures
        aux_k_xx, aux_k_s12 = np.gradient(slope_x, pixel_size_x, pixels_size_y)
        aux_k_s21, aux_k_yy = np.gradient(slope_y, pixel_size_x, pixels_size_y)
        aux_k_xy = .5 * (aux_k_s12 + aux_k_s21)

        slopes_x.append(slope_x)
        slopes_y.append(slope_y)
        curv_xx.append(aux_k_xx)
        curv_yy.append(aux_k_yy)
        curv_xy.append(aux_k_xy)

        deflection.append(disp_field)

    if acceleration_fields is None:
        # Assuming constant time-step size
        time_step_size = float(times[1])-float(times[0])

        vel_fields = np.gradient(deflection, axis=0) / time_step_size
        accel_field = np.gradient(vel_fields, axis=0) / time_step_size

    else:
        accel_field = np.array(acceleration_fields)

    deflection = np.array(deflection)
    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)
    curv_xx = np.array(curv_xx)
    curv_yy = np.array(curv_yy)
    curv_xy = np.array(curv_xy)
    accel_field = np.array(accel_field)
    times = np.array(times)

    return FieldStack(deflection, (slopes_x, slopes_y), (curv_xx, curv_yy, curv_xy), accel_field, times)
