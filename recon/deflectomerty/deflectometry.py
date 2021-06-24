import numpy as np
from scipy import signal
from recon.utils import list_files_in_folder
from matplotlib.pyplot import imread
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import muDIC as dic
from skimage.restoration import unwrap_phase
from scipy.signal.windows import triang


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


def disp_from_phase(phase, phase_0, grid_pitch, small_disp=True, maxit=10, axis=None, tol=1e-5):
    if small_disp:
        return -grid_pitch * unwrap_phase(2 * np.angle(phase / phase_0), wrap_around=True) / 2. / 2. / np.pi

    if not small_disp:
        n_x, n_y = phase.shape
        ys, xs = np.meshgrid(np.arange(n_y), np.arange(n_x))
        u = -grid_pitch * unwrap_phase(2 * np.angle(phase / phase_0), wrap_around=True) / 2. / 2. / np.pi

        u_first = u
        for i in range(maxit):
            if axis == 1:
                phase_n = map_coordinates(phase, [xs + u, ys], order=4, mode="mirror")
            elif axis == 0:
                phase_n = map_coordinates(phase, [xs, ys + u], order=4, mode="mirror")
            else:
                raise ValueError("No valid axis was given")

            u_last = u.copy()
            u = -grid_pitch * unwrap_phase(2 * np.angle(phase_n / phase_0), wrap_around=True) / 2. / 2. / np.pi
            if np.max(np.abs(u - u_last)) < tol:
                print("Large displacement correction converged in %i iterations" % i)
                print("Largest displacement correction was %f" % np.max(np.abs(u_first - u)))
                return u

        print("Large displacement correction diverged, returning uncorrected frame")
        return u_first


def disp_from_phase_both(phase_x, phase_x_0, phase_y, phase_y_0, grid_pitch, small_disp=True, maxit=10, axis=None,
                         tol=1e-5, unwrap=True):
    if unwrap is False:
        def unwrap_phase(x, *args, **kwargs):
            return x

    if small_disp:
        u_x = -grid_pitch * unwrap_phase(2 * np.angle(phase_x / phase_x_0), wrap_around=True, seed=0) / 2. / 2. / np.pi
        u_y = -grid_pitch * unwrap_phase(2 * np.angle(phase_y / phase_y_0), wrap_around=True, seed=0) / 2. / 2. / np.pi
        return u_x, u_y

    if not small_disp:
        n_x, n_y = phase_x.shape
        xs, ys = np.meshgrid(np.arange(n_y), np.arange(n_x))
        u_x = -grid_pitch * unwrap_phase(2 * np.angle(phase_x / phase_x_0), wrap_around=True, seed=0) / 2. / 2. / np.pi
        u_y = -grid_pitch * unwrap_phase(2 * np.angle(phase_y / phase_y_0), wrap_around=True, seed=0) / 2. / 2. / np.pi

        u_x_first = u_x
        u_y_first = u_y
        for i in range(maxit):
            phase_x_n = map_coordinates(phase_x, [ys + u_y, xs + u_x], order=4, mode="mirror")
            phase_y_n = map_coordinates(phase_y, [ys + u_y, xs + u_x], order=4, mode="mirror")

            u_x_last = u_x.copy()
            u_y_last = u_y.copy()
            u_x = -grid_pitch * unwrap_phase(2 * np.angle(phase_x_n / phase_x_0), wrap_around=True,
                                             seed=0) / 2. / 2. / np.pi
            u_y = -grid_pitch * unwrap_phase(2 * np.angle(phase_y_n / phase_y_0), wrap_around=True,
                                             seed=0) / 2. / 2. / np.pi
            if np.max(np.abs(u_x - u_x_last)) < tol and np.max(np.abs(u_y - u_y_last)) < tol:
                print("Phase correction converged in %i iterations with %.6f and %.6f pixels" % (
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

        # disp_x_from_phase = pixel_size_on_grid_plane * disp_from_phase(phase_x, phase_x0, grid_pitch, small_disp=False,
        #                                                                axis=0)
        # disp_y_from_phase = pixel_size_on_grid_plane * disp_from_phase(phase_y, phase_y0, grid_pitch, small_disp=False,
        #                                                                axis=1)

        disp_x_from_phase, disp_y_from_phase = disp_from_phase_both(phase_x, phase_x0, phase_y, phase_y0, grid_pitch,
                                                                    small_disp=False, unwrap=True)
        disp_x_from_phase = disp_x_from_phase * pixel_size_on_grid_plane
        disp_y_from_phase = disp_y_from_phase * pixel_size_on_grid_plane

        print("Largest displacement is %f pixels" % disp_from_phase(phase_y, phase_y0, grid_pitch).max())

        # slopes_y.append(angle_from_disp(disp_x_from_phase, mirror_grid_distance))
        # slopes_x.append(angle_from_disp(disp_y_from_phase, mirror_grid_distance))

        # n_pix_x, n_pix_y = disp_x_from_phase.shape
        # coords_y,coords_x = np.meshgrid(np.arange(-n_pix_x/2,n_pix_x/2),np.arange(-n_pix_y/2,n_pix_y/2))
        # coords_x = coords_x * pixel_size_on_grid_plane
        # coords_y = coords_y * pixel_size_on_grid_plane

        slopes_y.append(angle_from_disp(disp_x_from_phase, mirror_grid_distance))
        slopes_x.append(angle_from_disp(disp_y_from_phase, mirror_grid_distance))

    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)

    slopes_x = np.moveaxis(slopes_x, 0, -1)
    slopes_y = np.moveaxis(slopes_y, 0, -1)

    return slopes_x, slopes_y


def slopes_from_grid_imgs_dic(path_to_grid_imgs, grid_pitch, pixel_size_on_grid_plane, mirror_grid_distance,
                              ref_img_ids=None, only_img_ids=None, crop=None, elm_size=20):
    img_paths = list_files_in_folder(path_to_grid_imgs, file_type=".tif", abs_path=True)

    if not ref_img_ids:
        ref_img_ids = [0]
    grid_undeformed = np.mean([imread(img_paths[i]) for i in ref_img_ids], axis=0)

    if crop:
        grid_undeformed = grid_undeformed[crop[0]:crop[1], crop[2]:crop[3]]

    if only_img_ids:
        img_paths = [img_paths[i] for i in only_img_ids]

    print(imread(img_paths[0]).shape)

    imgs = dic.image_stack_from_list([imread(image_path) for image_path in img_paths])
    mesher = dic.Mesher()
    mesh = mesher.mesh(imgs, Xc1=10, Xc2=512 - 10, Yc1=10, Yc2=512 - 10, n_elx=40, n_ely=40, GUI=False)
    print(mesh.Xc1, mesh.Xc2, mesh.Xc1, mesh.Yc2)
    input = dic.DICInput(mesh, imgs)
    input.pad = 5
    anal = dic.DICAnalysis(input)
    res = anal.run()
    fields = dic.Fields(res)

    print("sdfsdfsdfsdf", fields.disp().shape)
    # plt.imshow(fields.disp()[0, 1, :, :, -1])
    # plt.show()

    slopes_y = angle_from_disp(pixel_size_on_grid_plane * fields.disp()[0, 0, :, :, :], mirror_grid_distance)
    slopes_x = angle_from_disp(pixel_size_on_grid_plane * fields.disp()[0, 1, :, :, :], mirror_grid_distance)

    return slopes_x, slopes_y
