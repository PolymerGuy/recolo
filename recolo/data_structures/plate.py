from collections import namedtuple


Plate = namedtuple("Plate",["E_mod", "pois_ratio","density","thickness", "bend_stiff_11","bend_stiff_12"])


def make_plate(modulus, pois_ratio, density, thickness):
    """
    Make a Plate object containing the metrics describing the plate

    Parameters
    ----------
    modulus : float
        The young's modulus of the material
    pois_ratio : float
        The poissons ratio of the material
    density : float
        The density of the material
    thickness : float
        The thickness of the plate

    Returns
    -------
    plate : Plate
        The Plate object containing all metrics of the plate
    """

    bend_stiff_11 = (thickness ** 3.) / 12. * modulus / (1. - pois_ratio ** 2.)
    bend_stiff_12 = (thickness ** 3.) / 12. * modulus * pois_ratio / (1. - pois_ratio ** 2.)
    return Plate(modulus,pois_ratio,density,thickness,bend_stiff_11,bend_stiff_12)

