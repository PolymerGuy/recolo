import numpy as np
from recon.slope_integration.sparce_integration import int2D
from scipy.ndimage import gaussian_filter


def disp_from_slopes(slopes_x, slopes_y, pixel_size, zero_at="bottom",zero_at_size=1, extrapolate_edge=0, filter_sigma=0, downsample=1):
    """ Calculate displacement fields by integrating slope fields using sparse matrix integration.
        The slopes can be filtered by a gaussian low-pass filter, downsampled and extrapolated before integration.
        Parameters
        ----------
        slopes_x : np.ndarray
            Slope field with shape (n_pix_x,n_pix_y,n_frames)
        slopes_y : np.ndarray
            Slope field with shape (n_pix_x,n_pix_y,n_frames)
        pixel_size : float
            The size of a pixel in slopes_x and slopes_y
        zero_at : string
            The position where a zero displacement boundary condition is enforced.
            Note that the boundary condition is not enforced exactly.
            Keywords:
                "top", "top_corners", "left", "right", "bottom", "bottom_corners"
        zero_at_size : int
            If zero_at is set to a corner, a rectangluar window with side lengths of zero_at_size  are used.
        extrapolate_edge : int
            Extrapolate edge by n-pixels by padding with the boundary values
        filter_sigma : float
            The standard deviation of the gaussian low pass filter
        downsample : int
            Downsample the field before integration by n-pixels.

        Returns
        -------
        ndarray
            The displacement fields with shape (n_pix_x,n_pix_y,n_frames)
        """

    if slopes_x.ndim != 3 or slopes_y.ndim != 3:
        raise ValueError("The slope fields have to have the shape (n_pix_x,n_pix_y,n_frames)")
    n_pix_x, n_pix_y, n_frames = slopes_x.shape

    if type(downsample) != int or downsample < 1:
        raise ValueError("The downsampling factor has to be an integer larger or equal to 1")

    disp_fields = []

    for i in range(n_frames):
        print("Integrating frame %i cropped by %i pixels" % (i, extrapolate_edge))
        slope_y = slopes_y[:, :, i]
        slope_x = slopes_x[:, :, i]

        slope_y = gaussian_filter(slope_y, sigma=filter_sigma)
        slope_x = gaussian_filter(slope_x, sigma=filter_sigma)

        slope_y = slope_y[::downsample, ::downsample]
        slope_x = slope_x[::downsample, ::downsample]

        if extrapolate_edge > 0:
            slope_x = np.pad(slope_x, pad_width=(extrapolate_edge, extrapolate_edge), mode="edge")
            slope_y = np.pad(slope_y, pad_width=(extrapolate_edge, extrapolate_edge), mode="edge")

        disp_field = int2D(slope_x, slope_y, pixel_size * downsample, pixel_size * downsample)

        if zero_at == "top":
            edge_mean = np.mean(disp_field[0, :])
        elif zero_at == "top corners":
            edge_mean = (np.mean(disp_field[:zero_at_size, :zero_at_size]) + np.mean(disp_field[:zero_at_size, -zero_at_size:])) / 2.
        elif zero_at == "left":
            edge_mean = np.mean(disp_field[:, 0])
        elif zero_at == "right":
            edge_mean = np.mean(disp_field[:, -1])
        elif zero_at == "bottom":
            edge_mean = np.mean(disp_field[-1, :])
        elif zero_at == "bottom corners":
            edge_mean = (np.mean(disp_field[-zero_at_size:, :zero_at_size]) + np.mean(disp_field[-zero_at_size:, -zero_at_size:])) / 2.
        else:
            raise ValueError("No valid zero_at received")

        disp_field = disp_field - edge_mean

        disp_fields.append(disp_field)

    return np.array(disp_fields)
