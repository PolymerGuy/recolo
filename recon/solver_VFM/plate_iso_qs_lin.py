import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d


def pressure_elastic_thin_plate(fields, plate, virtual_field, shift=False):
    A11 = convolve2d(fields.curv_xx, virtual_field.okxxfield, mode="valid") + convolve2d(fields.curv_yy,
                                                                                         virtual_field.okyyfield,
                                                                                         mode="valid") + 2. * convolve2d(
        fields.curv_xy, virtual_field.okxyfield, mode="valid")
    A11 = np.real(A11)

    A12 = convolve2d(fields.curv_xx, virtual_field.okyyfield, mode="valid") + convolve2d(fields.curv_yy,
                                                                                         virtual_field.okxxfield,
                                                                                         mode="valid") - 2. * convolve2d(
        fields.curv_xy, virtual_field.okxyfield, mode="valid")
    A12 = np.real(A12)

    a_u3 = plate.density * plate.thickness * convolve2d(fields.acceleration, virtual_field.owfield, mode="valid")
    a_u3 = np.real(a_u3)

    U3 = np.sum(virtual_field.owfield)

    op = (A11 * plate.bend_stiff_11 + A12 * plate.bend_stiff_12 + a_u3) / U3

    if shift:
        op = ndimage.shift(op, (-0.5, -0.5), order=3)

    return op
