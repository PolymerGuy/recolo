import numpy as np

def dotted_grid(xs, ys, pitch, pixel_size=1, oversampling=1):
    if np.mod(oversampling, 2) == 0:
        raise ValueError("The oversampling has to be an odd number")
    if oversampling == 1:
        coordinate_spread = np.array([0.])
    else:
        coordinate_spread = np.linspace(-pixel_size / 2., pixel_size / 2., oversampling)
    xs_spread = xs[:, :, np.newaxis, np.newaxis] + coordinate_spread[np.newaxis, np.newaxis, np.newaxis, :]
    ys_spread = ys[:, :, np.newaxis, np.newaxis] + coordinate_spread[np.newaxis, np.newaxis, :, np.newaxis]

    gray_scales = 0.5*(2.+np.cos(2. * np.pi * xs_spread / float(pitch)) + np.cos(2. * np.pi * ys_spread / float(pitch)))
    return np.mean(gray_scales, axis=(-1, -2))

