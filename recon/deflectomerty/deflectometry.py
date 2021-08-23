import numpy as np
from recon.data_import import list_files_in_folder
from matplotlib.pyplot import imread
import logging
from recon.deflectomerty.grid_method import detect_phase, disp_fields_from_phases, angle_from_disp


def slopes_from_images(path_to_grid_imgs, grid_pitch, pixel_size_on_grid_plane, mirror_grid_distance,
                       ref_img_ids=None, only_img_ids=None, crop=None, correct_phase=True):


    img_paths = list_files_in_folder(path_to_grid_imgs, file_type=".tif", abs_path=True)

    logger = logging.getLogger(__name__)
    if not ref_img_ids:
        ref_img_ids = [0]
    grid_undeformed = np.mean([imread(img_paths[i]) for i in ref_img_ids], axis=0)

    if crop:
        grid_undeformed = grid_undeformed[crop[0]:crop[1], crop[2]:crop[3]]

    slopes_x = []
    slopes_y = []

    if only_img_ids:
        img_paths = [img_paths[i] for i in only_img_ids]

    for i, img_path in enumerate(img_paths):
        logger.info("Running deflectometry on frame %s" % img_path)
        grid_displaced_eulr = imread(img_path)

        if crop:
            grid_displaced_eulr = grid_displaced_eulr[crop[0]:crop[1], crop[2]:crop[3]]

        slope_x, slope_y = slopes_from_grids(grid_undeformed, grid_displaced_eulr, mirror_grid_distance, grid_pitch,
                                             correct_phase=correct_phase)

        slopes_y.append(slope_x)
        slopes_x.append(slope_y)

    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)

    return slopes_x, slopes_y


def slopes_from_grids(grid_undeformed, grid_deformed, mirror_grid_dist, grid_pitch, correct_phase=True):
    phase_x0, phase_y0 = detect_phase(grid_undeformed, grid_pitch)
    phase_x, phase_y = detect_phase(grid_deformed, grid_pitch)

    disp_x_from_phase, disp_y_from_phase = disp_fields_from_phases(phase_x, phase_x0, phase_y, phase_y0,
                                                                   grid_pitch, correct_phase=correct_phase)

    slopes_x = angle_from_disp(disp_x_from_phase, mirror_grid_dist)
    slopes_y = angle_from_disp(disp_y_from_phase, mirror_grid_dist)

    return slopes_x, slopes_y
