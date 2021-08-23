import numpy as np
from recon.data_import import list_files_in_folder
from matplotlib.pyplot import imread
import logging
from recon.deflectomerty.grid_method import angle_from_disp, disp_from_grids


def slopes_from_images(path_to_img_folder, grid_pitch, mirror_grid_distance, ref_img_ids=None, only_img_ids=None,
                       crop=None, correct_phase=True):
    """ Perform deflectometry to determine the slope fields associated with a series of images.
     the slope fields of a specimen based on the dof every pixel based on the phase modulation in two configurations, see [1] for
    more details.

    Parameters
    ----------
    path_to_img_folder : str
        The path to the folder which contains the grid images.
    grid_pitch : float
        The grid pitch in pixels
    mirror_grid_distance : float
        The distance from the mirrored plate to the grid.
    ref_img_ids : list
        The list of image ids which are used as reference.
        The mean of all images are used as a low noise reference image.
    only_img_ids : list
        The list of image ids which are used.
    crop : tuple
        The coordinates used to crop the images.
        The images a cropped as: image[crop[0]:crop[1], crop[2]:crop[3]]
    correct_phase : bool
        Perform phase unwrapping using [2].

    Returns
    -------
    disps_x,disps_y : ndarray
        The displacement fields with shape [n_frames,x,y]

    References
    ----------
    ..  [1] Michel Grediac, Frédéric Sur, Benoît Blaysat. The grid method for in-plane displacement and
    strain measurement: a review and analysis. Strain, Wiley-Blackwell, 2016, 52 (3), pp.205-243.
    ff10.1111/str.12182ff. ffhal-01317145f
    """

    img_paths = list_files_in_folder(path_to_img_folder, file_type=".tif", abs_path=True)

    logger = logging.getLogger(__name__)

    if only_img_ids:
        img_paths = [img_paths[i] for i in only_img_ids]

    if not ref_img_ids:
        logger.info("No reference images were specified, using first image as reference.")
        ref_img_ids = [0]
    grid_undeformed = np.mean([imread(img_paths[i]) for i in ref_img_ids], axis=0)

    if crop:
        grid_undeformed = grid_undeformed[crop[0]:crop[1], crop[2]:crop[3]]

    slopes_x = []
    slopes_y = []

    for i, img_path in enumerate(img_paths):
        logger.info("Running deflectometry on frame %s" % img_path)
        grid_displaced_eulr = imread(img_path)

        if crop:
            grid_displaced_eulr = grid_displaced_eulr[crop[0]:crop[1], crop[2]:crop[3]]

        disp_x_from_phase, disp_y_from_phase = disp_from_grids(grid_undeformed, grid_displaced_eulr, grid_pitch,
                                                               correct_phase=correct_phase)

        slope_x = angle_from_disp(disp_x_from_phase, mirror_grid_distance)
        slope_y = angle_from_disp(disp_y_from_phase, mirror_grid_distance)

        slopes_y.append(slope_x)
        slopes_x.append(slope_y)

    slopes_x = np.array(slopes_x)
    slopes_y = np.array(slopes_y)

    return slopes_x, slopes_y


