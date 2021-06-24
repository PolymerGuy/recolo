import numpy as np
from scipy import signal
from recon.utils import list_files_in_folder
from matplotlib.pyplot import imread
from scipy.ndimage import map_coordinates
from scipy.signal.windows import triang
from skimage.restoration import unwrap_phase


def gaussian_window(win_size):
    t_noy = np.ceil(4 * win_size)
    xs, ys = np.meshgrid(np.arange(-t_noy, t_noy + 1), np.arange(-t_noy, t_noy + 1))
    conv_matrix = np.exp(-(xs ** 2. + ys ** 2.) / (2. * (win_size ** 2.)))
    return conv_matrix / conv_matrix.sum()


def triangular_window(win_size):
    t_noy = np.floor(win_size)
    g1 = triang(2 * t_noy - 1)
    conv_matrix = np.outer(g1, g1)
    raise NotImplementedError("The triangular window does not yield valid results")
    return conv_matrix / conv_matrix.sum()


def detect_phase(img, grid_pitch, window="gaussian", boundary="symm"):
    s_x, s_y = np.shape(img)
    fc = 2. * np.pi / float(grid_pitch)

    if window == "triangular":
        conv_matrix = triangular_window(grid_pitch)
    else:
        conv_matrix = gaussian_window(grid_pitch)

    xs, ys = np.meshgrid(np.arange(s_y), np.arange(s_x))

    # x-direction
    img_complex_x = img * np.exp(-1j * fc * xs)
    phase_x = signal.convolve2d(img_complex_x, conv_matrix, boundary=boundary, mode='valid') / float(grid_pitch)

    # y-direction
    img_complex_y = img * np.exp(-1j * fc * ys)
    phase_y = signal.convolve2d(img_complex_y, conv_matrix, boundary=boundary, mode='valid') / float(grid_pitch)

    return phase_x, phase_y


def disp_from_phases(phase, phase0, grid_pitch, unwrap=True):
    if unwrap:
        return -grid_pitch * unwrap_phase(2 * np.angle(phase / phase0), wrap_around=True, seed=0) / 2. / 2. / np.pi
    else:
        return -grid_pitch * np.angle(phase / phase0) / 2. / np.pi


def disp_from_phase(phase_x, phase_x_0, phase_y, phase_y_0, grid_pitch, small_disp=True, maxit=10, tol=1e-5,
                    unwrap=True):
    if small_disp:
        u_x = disp_from_phases(phase_x, phase_x_0, grid_pitch, unwrap)
        u_y = disp_from_phases(phase_y, phase_y_0, grid_pitch, unwrap)
        return u_x, u_y

    if not small_disp:
        n_x, n_y = phase_x.shape
        xs, ys = np.meshgrid(np.arange(n_y), np.arange(n_x))
        u_x = disp_from_phases(phase_x, phase_x_0, grid_pitch, unwrap)
        u_y = disp_from_phases(phase_y, phase_y_0, grid_pitch, unwrap)

        u_x_first = u_x
        u_y_first = u_y
        for i in range(maxit):
            phase_x_n = map_coordinates(phase_x, [ys + u_y, xs + u_x], order=4, mode="mirror")
            phase_y_n = map_coordinates(phase_y, [ys + u_y, xs + u_x], order=4, mode="mirror")

            u_x_last = u_x.copy()
            u_y_last = u_y.copy()
            u_x = disp_from_phases(phase_x_n, phase_x_0, grid_pitch, unwrap)
            u_y = disp_from_phases(phase_y_n, phase_y_0, grid_pitch, unwrap)
            if np.max(np.abs(u_x - u_x_last)) < tol and np.max(np.abs(u_y - u_y_last)) < tol:
                print("Phase correction converged in %i iterations correcting %.6f and %.6f pixels" % (
                    i, np.max(np.abs(u_x_first - u_x)), np.max(np.abs(u_y_first - u_y))))
                return u_x, u_y

        print("Large displacement correction diverged, returning uncorrected frame")
        return u_x_first, u_x_first


def angle_from_disp(disp, mirror_grid_dist):
    return np.arctan(disp / mirror_grid_dist) / 2.


def angle_from_disp_large_angles(disp, mirror_grid_dist, coords):
    return (np.arctan((disp + coords) / mirror_grid_dist) - np.arctan((coords) / mirror_grid_dist)) / 2.


def slopes_from_grid_imgs(path_to_grid_imgs, grid_pitch, pixel_size_on_grid_plane, mirror_grid_distance,
                          ref_img_ids=None, only_img_ids=None, crop=None):
    img_paths = list_files_in_folder(path_to_grid_imgs, file_type=".tif", abs_path=True)

    if not ref_img_ids:
        ref_img_ids = [0]
    grid_undeformed = np.mean([imread(img_paths[i]) for i in ref_img_ids], axis=0)

    if crop:
        grid_undeformed = grid_undeformed[crop[0]:crop[1], crop[2]:crop[3]]

    phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

    slopes_x = []
    slopes_y = []

    if only_img_ids:
        img_paths = [img_paths[i] for i in only_img_ids]

    for i, img_path in enumerate(img_paths):
        print("Running deflectometry on frame %s" % img_path)
        grid_displaced_eulr = imread(img_path)

        if crop:
            grid_displaced_eulr = grid_displaced_eulr[crop[0]:crop[1], crop[2]:crop[3]]

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)

        disp_x_from_phase, disp_y_from_phase = disp_from_phase(phase_x, phase_x0, phase_y, phase_y0, grid_pitch,
                                                               small_disp=False, unwrap=True)
        disp_x_from_phase = disp_x_from_phase * pixel_size_on_grid_plane
        disp_y_from_phase = disp_y_from_phase * pixel_size_on_grid_plane

        slopes_y.append(angle_from_disp(disp_x_from_phase, mirror_grid_distance))
        slopes_x.append(angle_from_disp(disp_y_from_phase, mirror_grid_distance))

    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)

    slopes_x = np.moveaxis(slopes_x, 0, -1)
    slopes_y = np.moveaxis(slopes_y, 0, -1)

    return slopes_x, slopes_y
