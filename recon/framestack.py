from recon.diff_tools import dF_complex_x, dF_complex_y, ddF_complex_x, ddF_complex_xy, ddF_complex_y
import numpy as np
from .analydisp import pressure_sinusoidal
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from copy import copy


class FrameStack(object):
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
        return Frame(self._deflection_[frame_id], (self._slope_x_[frame_id], self._slope_y_[frame_id]),
                     (self._curv_xx_[frame_id], self._curv_yy_[frame_id], self._curv_xy_[frame_id]),
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
        return FrameStack(deflection, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), acceleration, times)

    def filtered_space(self, sigma):
        deflection = self.__filter_space__(self._deflection_, sigma)

        slope_x = self.__filter_space__(self._slope_x_, sigma)
        slope_y = self.__filter_space__(self._slope_y_, sigma)

        curv_xx = self.__filter_space__(self._curv_xx_, sigma)
        curv_yy = self.__filter_space__(self._curv_yy_, sigma)
        curv_xy = self.__filter_space__(self._curv_xy_, sigma)

        acceleration = self.__filter_space__(self._acceleration_, sigma)
        times = self._times_
        return FrameStack(deflection, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), acceleration, times)

    def down_sampled(self, every_n_frame):
        return FrameStack(self._deflection_[::every_n_frame],
                          (self._slope_x_[::every_n_frame], self._slope_y_[::every_n_frame]), (
                           self._curv_xx_[::every_n_frame], self._curv_yy_[::every_n_frame],
                           self._curv_xy_[::every_n_frame]), self._acceleration_[::every_n_frame],
                           self._times_[::every_n_frame])


class Frame(object):
    def __init__(self, deflection, slopes, curvatures, acceleration, time):
        self.deflection = deflection
        self.slope_x, self.slope_y = slopes
        self.curv_xx, self.curv_yy, self.curv_xy = curvatures
        self.acceleration = acceleration
        self.time = time

    def shape(self):
        return np.shape(self.deflection)




def fields_from_abaqus_rpts(abaqus_data, downsample=False,bin_downsamples=False, accel_from_disp=True, filter_space_sigma=None, filter_time_sigma=None):

    #crop=30
    disp_fields = abaqus_data.disp_fields#[:,crop:-crop,crop:-crop]
    #disp_fields = disp_fields - disp_fields[:,0,0][:,np.newaxis,np.newaxis]
    accel_field = abaqus_data.accel_fields#[:,crop:-crop,crop:-crop]
    times = abaqus_data.times
    plate_len_x = abaqus_data.plate_len_x
    plate_len_y = abaqus_data.plate_len_y

    if downsample and not bin_downsamples:
        disp_fields = disp_fields[::downsample,:,:]
        accel_field = accel_field[::downsample,:,:]
        times = times[::downsample]
    elif downsample and bin_downsamples:
        n_frames,n_x,n_y = disp_fields.shape
        n_bins = np.floor(n_frames/downsample)
        n_data_pts = int(n_bins * downsample)
        print("Binning data, losing the %i last data points"%(n_frames-n_data_pts))

        disp_fields = np.reshape(disp_fields[:n_data_pts,:,:],(-1,downsample,n_x,n_y)).mean(axis=1)
        accel_field = np.reshape(accel_field[:n_data_pts,:,:],(-1,downsample,n_x,n_y)).mean(axis=1)
        times = times[:n_data_pts:downsample]

    if filter_time_sigma:
        print("Filtering in time with sigma=%f" % float(filter_time_sigma))
        disp_fields = gaussian_filter1d(disp_fields, sigma=filter_time_sigma, axis=0)

    if filter_space_sigma:
        for i in range(len(disp_fields)):
            print("Filtering frame %i" % i)
            disp_fields[i, :, :] = gaussian_filter(disp_fields[i, :, :], sigma=filter_space_sigma,mode="nearest")

    if accel_from_disp:
        return field_from_displacement(disp_fields, None, times, plate_len_x, plate_len_y)
    else:
        return field_from_displacement(disp_fields, accel_field, times, plate_len_x, plate_len_y)


def fields_from_experiments(abaqus_data, filter_space_sigma=None, filter_time_sigma=None):

    disp_fields = np.moveaxis(abaqus_data,-1,0)

    n_times = disp_fields.shape[0]
    times = np.arange(n_times) * 1./75000.
    plate_len_x = 0.3
    plate_len_y = 0.3


    if filter_time_sigma:
        print("Filtering in time with sigma=%f" % float(filter_time_sigma))
        disp_fields = gaussian_filter1d(disp_fields, sigma=filter_time_sigma, axis=0)

    if filter_space_sigma:
        for i in range(len(disp_fields)):
            print("Filtering frame %i" % i)
            disp_fields[i, :, :] = gaussian_filter(disp_fields[i, :, :], sigma=filter_space_sigma)
    return field_from_displacement(disp_fields, None, times, plate_len_x, plate_len_y)



def field_from_disp_func(disp_func, npts_x, npts_y, plate_x, plate_y):
    # calculate out-of-plane displacements
    xs, ys = np.meshgrid(np.linspace(0., 1., npts_x), np.linspace(0., 1., npts_y))
    deflection = disp_func(xs, ys)

    press = pressure_sinusoidal(100, xs, ys)
    # press = np.zeros_like(deflection)

    # Calculate slopes
    slope_x = -dF_complex_y(disp_func, xs, ys) / plate_x
    slope_y = -dF_complex_x(disp_func, xs, ys) / plate_y

    # calculate curvatures
    curv_yy = (-ddF_complex_x(disp_func, xs, ys) / (plate_x ** 2.))
    curv_xx = (-ddF_complex_y(disp_func, xs, ys) / (plate_x ** 2.))
    curv_xy = (-ddF_complex_xy(disp_func, xs, ys) / (plate_x ** 2.))
    return FrameStack(deflection, press, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), np.zeros_like(deflection))


def field_from_displacement_old(disp_field, acceleration_field, times, plate_x, plate_y):
    npts_x, npts_y = disp_field.shape
    odx = plate_x / npts_x
    ody = plate_y / npts_y

    # calculate out-of-plane displacements
    xs, ys = np.meshgrid(np.linspace(0., 1., npts_x), np.linspace(0., 1., npts_y))
    deflection = disp_field

    # press = pressure_sinusoidal(100, xs, ys)
    press = np.zeros_like(deflection)

    # calculate slopes
    slope_x, slope_y = np.gradient(-deflection, odx, ody)

    ###
    # calculate curvatures
    aux_k_xx, aux_k_s12 = np.gradient(slope_x, odx, ody)
    aux_k_s21, aux_k_yy = np.gradient(slope_y, odx, ody)
    aux_k_xy = .5 * (aux_k_s12 + aux_k_s21)

    slope_x = slope_x[1:-1, 1:-1]
    slope_y = slope_y[1:-1, 1:-1]

    curv_xx = aux_k_xx[1:-1, 1:-1]
    curv_yy = aux_k_yy[1:-1, 1:-1]
    curv_xy = aux_k_xy[1:-1, 1:-1]

    deflection = disp_field[1:-1, 1:-1]
    accel_field = acceleration_field[1:-1, 1:-1]

    return FrameStack(deflection, press, (slope_x, slope_y), (curv_xx, curv_yy, curv_xy), accel_field, times)


def field_from_displacement(disp_fields, acceleration_fields, times, plate_x, plate_y):
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
        odx = plate_x / npts_x
        ody = plate_y / npts_y

        # calculate out-of-plane displacements
        xs, ys = np.meshgrid(np.linspace(0., 1., npts_x), np.linspace(0., 1., npts_y))

        # calculate slopes
        slope_x, slope_y = np.gradient(-disp_field, odx, ody)

        ###
        # calculate curvatures
        aux_k_xx, aux_k_s12 = np.gradient(slope_x, odx, ody)
        aux_k_s21, aux_k_yy = np.gradient(slope_y, odx, ody)
        aux_k_xy = .5 * (aux_k_s12 + aux_k_s21)

        slopes_x.append(slope_x[1:-1, 1:-1])
        slopes_y.append(slope_y[1:-1, 1:-1])

        curv_xx.append(aux_k_xx[1:-1, 1:-1])
        curv_yy.append(aux_k_yy[1:-1, 1:-1])
        curv_xy.append(aux_k_xy[1:-1, 1:-1])

        deflection.append(disp_field[1:-1, 1:-1])

    if acceleration_fields is None:
        analysis_time = times[-1]
        n_time_frames = len(times)
        time_step_size = float(analysis_time) / float(n_time_frames)

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


    return FrameStack(deflection, (slopes_x, slopes_y), (curv_xx, curv_yy, curv_xy), accel_field, times)
