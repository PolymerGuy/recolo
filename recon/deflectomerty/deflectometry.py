import numpy as np
from scipy import signal
from recon.utils import list_files_in_folder
from matplotlib.pyplot import imread


def gaussian_window(win_size):
    t_noy = np.ceil(4 * win_size)
    xs, ys = np.meshgrid(np.arange(-t_noy, t_noy + 1), np.arange(-t_noy, t_noy + 1))
    conv_matrix = np.exp(-(xs ** 2 + ys ** 2) / (2 * (win_size ** 2)))
    return conv_matrix / conv_matrix.sum()


def detect_phase(img, grid_pitch):
    s_x, s_y = np.shape(img)
    fc = 2. * np.pi / float(grid_pitch)

    # Gaussian window
    conv_matrix = gaussian_window(grid_pitch)

    xs, ys = np.meshgrid(np.arange(s_x), np.arange(s_y))

    # x-direction
    img_complex_x = img * np.exp(-1j * fc * xs)
    phase_x = signal.convolve2d(img_complex_x, conv_matrix, boundary='symm', mode='valid') / grid_pitch

    # y-direction
    img_complex_y = img * np.exp(-1j * fc * ys)
    phase_y = signal.convolve2d(img_complex_y, conv_matrix, boundary='symm', mode='valid') / grid_pitch

    return phase_x, phase_y


def disp_from_phase(phase, phase_0, grid_pitch):
    return grid_pitch * -np.angle(phase / phase_0) / 2. / np.pi


def angle_from_disp(disp, mirror_grid_dist):
    return np.arctan(disp / mirror_grid_dist) / 2.


def angle_from_disp_large_angles(disp, mirror_grid_dist, coords):
    return (np.arctan((disp + coords) / mirror_grid_dist) - np.arctan((coords) / mirror_grid_dist)) / 2.


def slopes_from_grid_imgs(path_to_grid_imgs, grid_pitch, pixel_size_on_grid_plane, mirror_grid_distance,
                          ref_img_ids=None, only_img_ids=None):
    img_paths = list_files_in_folder(path_to_grid_imgs, file_type=".tif", abs_path=True)

    if not ref_img_ids:
        ref_img_ids = [0]
    grid_undeformed = np.mean([imread(img_paths[i]) for i in ref_img_ids], axis=0)

    phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)

    slopes_x = []
    slopes_y = []

    if only_img_ids:
        img_paths = [img_paths[i] for i in only_img_ids]

    for i, img_path in enumerate(img_paths):
        print("Running deflectometry on frame %s" % img_path)
        grid_displaced_eulr = imread(img_path)

        phase_x, phase_y = detect_phase(grid_displaced_eulr, grid_pitch)

        disp_x_from_phase = pixel_size_on_grid_plane * disp_from_phase(phase_x, phase_x0, grid_pitch)
        disp_y_from_phase = pixel_size_on_grid_plane * disp_from_phase(phase_y, phase_y0, grid_pitch)

        # slopes_y.append(angle_from_disp(disp_x_from_phase, mirror_grid_distance))
        # slopes_x.append(angle_from_disp(disp_y_from_phase, mirror_grid_distance))

        n_pix_x, n_pix_y = disp_x_from_phase.shape
        coords_y,coords_x = np.meshgrid(np.arange(-n_pix_x/2,n_pix_x/2),np.arange(-n_pix_y/2,n_pix_y/2))
        coords_x = coords_x * pixel_size_on_grid_plane
        coords_y = coords_y * pixel_size_on_grid_plane

        slopes_y.append(angle_from_disp_large_angles(disp_x_from_phase, mirror_grid_distance,coords_x))
        slopes_x.append(angle_from_disp_large_angles(disp_y_from_phase, mirror_grid_distance,coords_y))

    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)

    slopes_x = np.moveaxis(slopes_x, 0, -1)
    slopes_y = np.moveaxis(slopes_y, 0, -1)

    return slopes_x, slopes_y
