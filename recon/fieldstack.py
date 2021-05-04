from recon.analydisp.diff_tools import dF_complex_x, dF_complex_y, ddF_complex_x, ddF_complex_xy, ddF_complex_y
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from copy import copy
from collections import namedtuple


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

    def __filter_time__(self, field, sigma):
        print("Filtering in time with sigma=%f" % float(sigma))
        return gaussian_filter1d(field, sigma=sigma, axis=0)

    def __filter_space__(self, field, sigma):
        first_axis = gaussian_filter1d(field, sigma=sigma, axis=1)
        return gaussian_filter1d(first_axis, sigma=sigma, axis=2)

    def shape(self):
        return np.shape(self._deflection_)

    def filtered_time(self, sigma):
        deflection = self.__filter_time__(self._deflection_, sigma)

        slope_x = self.__filter_time__(self._slope_x_, sigma)
        slope_y = self.__filter_time__(self._slope_y_, sigma)

        curv_xx = self.__filter_time__(self._curv_xx_, sigma)
        curv_yy = self.__filter_time__(self._curv_yy_, sigma)
        curv_xy = self.__filter_time__(self._curv_xy_, sigma)

        acceleration = self.__filter_time__(self._acceleration_, sigma)
        times = self._times_
        return FieldStack(deflection, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), acceleration, times)

    def filtered_space(self, sigma):
        deflection = self.__filter_space__(self._deflection_, sigma)

        slope_x = self.__filter_space__(self._slope_x_, sigma)
        slope_y = self.__filter_space__(self._slope_y_, sigma)

        curv_xx = self.__filter_space__(self._curv_xx_, sigma)
        curv_yy = self.__filter_space__(self._curv_yy_, sigma)
        curv_xy = self.__filter_space__(self._curv_xy_, sigma)

        acceleration = self.__filter_space__(self._acceleration_, sigma)
        times = self._times_
        return FieldStack(deflection, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), acceleration, times)

    def down_sampled(self, every_n_frame):
        return FieldStack(self._deflection_[::every_n_frame],
                          (self._slope_x_[::every_n_frame], self._slope_y_[::every_n_frame]), (
                              self._curv_xx_[::every_n_frame], self._curv_yy_[::every_n_frame],
                              self._curv_xy_[::every_n_frame]), self._acceleration_[::every_n_frame],
                          self._times_[::every_n_frame])


Fields = namedtuple("Fields",
                    ["deflection", "slope_x", "slope_y", "curv_xx", "curv_yy", "curv_xy", "acceleration", "time"])


def fields_from_abaqus_rpts(abaqus_data, downsample=False,downsample_space=None, bin_downsamples=False, accel_from_disp=True,
                            filter_space_sigma=None, filter_time_sigma=None, noise_amp_sigma=None):
    disp_fields = abaqus_data.disp_fields
    accel_field = abaqus_data.accel_fields
    times = abaqus_data.times
    plate_len_x = abaqus_data.plate_len_x
    plate_len_y = abaqus_data.plate_len_y

    if downsample and not bin_downsamples:
        disp_fields = disp_fields[::downsample, :, :]
        accel_field = accel_field[::downsample, :, :]
        times = times[::downsample]
    elif downsample and bin_downsamples:
        n_frames, n_x, n_y = disp_fields.shape
        n_bins = np.floor(n_frames / downsample)
        n_data_pts = int(n_bins * downsample)
        print("Binning data, losing the %i last data points" % (n_frames - n_data_pts))

        disp_fields = np.reshape(disp_fields[:n_data_pts, :, :], (-1, downsample, n_x, n_y)).mean(axis=1)
        accel_field = np.reshape(accel_field[:n_data_pts, :, :], (-1, downsample, n_x, n_y)).mean(axis=1)
        times = times[:n_data_pts:downsample]

    if downsample_space:
        disp_fields = disp_fields[:,::downsample_space,::downsample_space]
        accel_field = accel_field[:,::downsample_space,::downsample_space]

    if noise_amp_sigma:
        disp_fields = disp_fields + np.random.random(disp_fields.shape) * noise_amp_sigma

    if filter_time_sigma:
        print("Filtering in time with sigma=%f" % float(filter_time_sigma))
        disp_fields = gaussian_filter1d(disp_fields, sigma=filter_time_sigma, axis=0)

    if filter_space_sigma:
        for i in range(len(disp_fields)):
            print("Filtering frame %i" % i)
            disp_fields[i, :, :] = gaussian_filter(disp_fields[i, :, :], sigma=filter_space_sigma, mode="nearest")

    if accel_from_disp:
        return fieldStack_from_disp_fields(disp_fields, None, times, plate_len_x, plate_len_y)
    else:
        return fieldStack_from_disp_fields(disp_fields, accel_field, times, plate_len_x, plate_len_y)


def kinematic_fields_from_experiments(exp_disp_field, pixel_size, sampling_rate, filter_space_sigma=None,
                                      filter_time_sigma=None):

    # Copy to make in-place operations safe
    disp_fields = copy(exp_disp_field)

    n_times, n_pts_x, n_pts_y = disp_fields.shape

    times = np.arange(n_times) * 1. / sampling_rate
    field_len_x = n_pts_x * pixel_size
    field_len_y = n_pts_y * pixel_size

    if filter_time_sigma:
        print("Filtering in time with sigma=%f" % float(filter_time_sigma))
        disp_fields = gaussian_filter1d(disp_fields, sigma=filter_time_sigma, axis=0, mode="nearest")

    if filter_space_sigma:
        for i in range(len(disp_fields)):
            print("Filtering frame %i" % i)
            disp_fields[i, :, :] = gaussian_filter(disp_fields[i, :, :], sigma=filter_space_sigma)
    return fieldStack_from_disp_fields(disp_fields, None, times, field_len_x, field_len_y)


def fieldStack_from_disp_func(disp_func, npts_x, npts_y, plate_x, plate_y):
    # calculate out-of-plane displacements
    xs, ys = np.meshgrid(np.linspace(0., 1., npts_x), np.linspace(0., 1., npts_y))
    deflection = disp_func(xs, ys)

    # Calculate slopes
    slope_x = -dF_complex_y(disp_func, xs, ys) / plate_x
    slope_y = -dF_complex_x(disp_func, xs, ys) / plate_y

    # calculate curvatures
    curv_yy = (-ddF_complex_x(disp_func, xs, ys) / (plate_x ** 2.))
    curv_xx = (-ddF_complex_y(disp_func, xs, ys) / (plate_x ** 2.))
    curv_xy = (-ddF_complex_xy(disp_func, xs, ys) / (plate_x ** 2.))

    # Add an initial axis to make this the first frame in the stack
    deflection = deflection[np.newaxis,:,:]
    slope_x = slope_x[np.newaxis,:,:]
    slope_y = slope_y[np.newaxis,:,:]
    curv_yy = curv_yy[np.newaxis,:,:]
    curv_xx = curv_xx[np.newaxis,:,:]
    curv_xy = curv_xy[np.newaxis,:,:]

    return FieldStack(deflection, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), np.zeros_like(deflection),[0])


def fieldStack_from_disp_fields(disp_fields, acceleration_fields, times, plate_x, plate_y):
    # This function removes the outer most pixels around the whole field
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
        pixel_size_x = plate_x / npts_x
        pixels_size_y = plate_y / npts_y

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
        accel_field = np.array(acceleration_fields)[:, 1:-1, 1:-1]

    deflection = np.array(deflection)
    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)
    curv_xx = np.array(curv_xx)
    curv_yy = np.array(curv_yy)
    curv_xy = np.array(curv_xy)
    accel_field = np.array(accel_field)
    times = np.array(times)

    return FieldStack(deflection, (slopes_x, slopes_y), (curv_xx, curv_yy, curv_xy), accel_field, times)
