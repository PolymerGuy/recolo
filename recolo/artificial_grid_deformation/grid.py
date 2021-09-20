import numpy as np


def dotted_grid(xs, ys, pitch, pixel_size=1, oversampling=1, noise_std=None):
    """
    Generate a grid image.
    The gray scale values of the grid is given by:

    g(x,y) = 0.5 * (2. + cos(2. * pi * x / pitch) + cos(2. * pi * y / pitch))

    Parameters
    ----------
    xs : ndarray
        The x-coordinates of the pixels in physical units
    ys : ndarray
        The y-coordinates of the pixels in physical units
    pitch : float
        The grid pitch in pixels
    pixel_size : float
        The size of a pixel in physical units
    oversampling : int
        Additional oversampling to reduce sampling artefacts of the grid.
    noise_std : float
        The standard deviation of the additive gaussian noise
    Returns
    -------
    grid: ndarray
        The grid image

    """
    if np.mod(oversampling, 2) == 0:
        raise ValueError("The oversampling has to be an odd number")
    if oversampling == 1:
        coordinate_spread = np.array([0.])
    else:
        coordinate_spread = np.linspace(-pixel_size / 2., pixel_size / 2., oversampling)
    xs_spread = xs[:, :, np.newaxis, np.newaxis] + coordinate_spread[np.newaxis, np.newaxis, np.newaxis, :]
    ys_spread = ys[:, :, np.newaxis, np.newaxis] + coordinate_spread[np.newaxis, np.newaxis, :, np.newaxis]

    gray_scales_oversampled = 0.5 * (
            2. + np.cos(2. * np.pi * xs_spread / float(pitch)) + np.cos(2. * np.pi * ys_spread / float(pitch)))
    gray_scales = np.mean(gray_scales_oversampled, axis=(-1, -2))
    if noise_std:
        gray_scales_range = np.max(gray_scales) - np.min(gray_scales)
        # The noise standard deviation is multiplied by the grey scale amplitude being two.
        noise = np.random.normal(0, noise_std * gray_scales_range, gray_scales.size).reshape(gray_scales.shape)
        return gray_scales + noise
    else:
        return gray_scales
