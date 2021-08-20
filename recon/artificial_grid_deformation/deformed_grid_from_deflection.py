import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from . import interpolated_disp_field, find_coords_in_undef_conf,make_dotted_grid


def deform_grid_from_deflection(deflection_field, pixel_size, mirror_grid_dist, grid_pitch, upscale=4, oversampling=5):
    if upscale > 1:
        disp_fields = zoom(deflection_field, upscale, prefilter=True, order=3)
    else:
        disp_fields = deflection_field

    slopes_x, slopes_y = np.gradient(disp_fields, pixel_size / float(upscale))
    u_x = slopes_x * mirror_grid_dist * 2.
    u_y = slopes_y * mirror_grid_dist * 2.

    n_pix_x, n_pix_y = disp_fields.shape
    xs, ys = np.meshgrid(np.arange(n_pix_x), np.arange(n_pix_y))

    interp_u = interpolated_disp_field(u_x, u_y, dx=1, dy=1, order=3, mode="nearest")

    Xs, Ys = find_coords_in_undef_conf(xs, ys, interp_u, tol=1e-9)

    grid_deformed = make_dotted_grid(Xs, Ys, grid_pitch, oversampling=oversampling)

    return grid_deformed




