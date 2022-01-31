import numpy as np
from scipy import ndimage
from scipy.signal import convolve2d
import logging
import recolo

def calc_pressure_thin_elastic_plate(fields, plate, virtual_fields, shift=False):
    """
    Calculate pressure field based on kinematic fields. This approach used the virtual fields method and that the
    pressure is acting on a thin plate under elastic deformation.

    Parameters
    ----------
    fields : Fields object
        The kinematic fields
    plate : Plate object
        The plate metrics
    virtual_fields : Virtual fields object
        The virtual fields
    shift : bool
        Correct for 0.5 pixel shift using bicubic spline interpolation

    Returns
    -------
    press : ndarray
        The reconstructed pressure field
    """
    # TODO: Insert the equation which is solved in the docstring.
    logger = logging.getLogger(__name__)
    if not isinstance(fields,recolo.Fields):
        raise IOError("The kinematic fields have to be given as an instance of the Fields class")

    if not isinstance(plate, recolo.data_structures.plate.Plate):
        raise IOError("The plate metrics have to be given as an instance of the Plate class")

    if not isinstance(virtual_fields,recolo.virtual_fields.Hermite16):
        # TODO: Make an abstract base class for the virtual fields
        raise IOError("The virtual fields have to be given as an instance of the Hermite16 class")

    logger.info("Reconstructing pressure")
    A11 = convolve2d(fields.curv_xx, virtual_fields.curv_xx, mode="valid") + convolve2d(fields.curv_yy,
                                                                                        virtual_fields.curv_yy,
                                                                                        mode="valid") + 2. * convolve2d(
        fields.curv_xy, virtual_fields.curv_xy, mode="valid")
    A11 = np.real(A11)

    A12 = convolve2d(fields.curv_xx, virtual_fields.curv_yy, mode="valid") + convolve2d(fields.curv_yy,
                                                                                        virtual_fields.curv_xx,
                                                                                        mode="valid") - 2. * convolve2d(
        fields.curv_xy, virtual_fields.curv_xy, mode="valid")
    A12 = np.real(A12)

    a_u3 = plate.density * plate.thickness * convolve2d(fields.acceleration, virtual_fields.deflection, mode="valid")
    a_u3 = np.real(a_u3)

    U3 = np.sum(virtual_fields.deflection)

    press = (A11 * plate.bend_stiff_11 + A12 * plate.bend_stiff_12 + a_u3) / U3

    if shift:
        press = ndimage.shift(press, (-0.5, -0.5), order=3)

    return press
