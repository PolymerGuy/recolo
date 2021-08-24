import numpy as np
from scipy.ndimage import zoom
from . import interpolated_disp_field, find_coords_in_undef_conf, dotted_grid


def deform_grid_from_deflection(deflection_field, pixel_size, mirror_grid_dist, grid_pitch, upscale=4, oversampling=5):
    """
    Generate a deformed grid image according to the deflection of the mirrored plate.


    Parameters
    ----------
    deflection_field : ndarray
        The out of plane deflection of the mirrored plate
    pixel_size : float
        The pixel size on the mirrored grid
    mirror_grid_dist : float
        The distance from the grid to the mirror. This is assumed to be the same as the mirror to sensor distance.
    grid_pitch : float
        The grid pitch in pixels
    upscale : int
        Upscaling of the deflection field to produce a grid at higher resolution.
    oversampling : int
        Additional oversampling to reduce sampling artefacts of the grid.
    Returns
    -------
    grid: ndarray
        The grid image deformed according the the deflection of the mirrored plate.

    """

    if upscale < 1:
        raise ValueError("The upscaling has to be larger or equal to one.")

    # Upscale the deflection field to produce a higher resolution grid image
    if upscale > 1:
        disp_fields = zoom(deflection_field, upscale, prefilter=True, order=3)
    else:
        disp_fields = deflection_field

    # Calculate the slope of the mirrored plate
    slopes_x, slopes_y = np.gradient(disp_fields, pixel_size / float(upscale))
    # Calculate the apparent displacement from the slopes
    u_x = slopes_x * mirror_grid_dist * 2.
    u_y = slopes_y * mirror_grid_dist * 2.
    interp_u = interpolated_disp_field(u_x, u_y, dx=1, dy=1, order=3, mode="nearest")

    # Generate the coordinates corresponding to the pixels on the sensor
    n_pix_x, n_pix_y = disp_fields.shape
    xs, ys = np.meshgrid(np.arange(n_pix_x), np.arange(n_pix_y))
    Xs, Ys = find_coords_in_undef_conf(xs, ys, interp_u, tol=1e-9)

    return dotted_grid(Xs, Ys, grid_pitch, oversampling=oversampling)
