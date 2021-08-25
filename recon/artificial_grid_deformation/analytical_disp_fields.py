import numpy as np

def harmonic_disp_field(amplitude, periode, n_periodes, formulation="Lagrangian"):
    """
    Harmonic displacement field with associated reference coordinates, current coordinates and displacement fields.

    The following coordinate definition is used:
        x = X + u(X)
    where x is the current coordinates, X is the coordinates in the undeformed configuration and u(X) is the
    displacement field expressed in therm of coordinates in the undeformed configuration.

    Parameters
    ----------
    amplitude : float
        The amplitude of the harmonic displacement field
    periode : int
        The periode of the harmonic displacement field
    n_periodes : int
        The number of displacement periodes
    formulation : str
        The coodinate formulation used for the displacement field.
        "Lagrangian" corresponds to u(X)
        "Eulerian" corresponds to u(x)
    Returns
    -------
    xs, ys, Xs, Ys, displacement_x, displacement_y : ndarray
        xs,ys are the coordinates in the deformed configuration
        Xs,Ys are the coordinates in the undeformed configuration
        displacement_x, displacement_y are the displacement component fields
    """
    x = np.arange(n_periodes * periode, dtype=float)
    y = np.arange(n_periodes * periode, dtype=float)

    xs, ys = np.meshgrid(x, y)
    Xs, Ys = np.meshgrid(x, y)

    displacement_x = amplitude * np.sin(n_periodes * 2. * np.pi * xs / xs.max())
    displacement_y = amplitude * np.sin(n_periodes * 2. * np.pi * ys / ys.max())

    if formulation == "eulerian":
        return xs, ys, (xs - displacement_x), (ys - displacement_y), displacement_x, displacement_y

    elif formulation == "lagrangian":
        tol = 1.e-12
        for i in range(20):
            Xs = Xs - (xs - Xs - amplitude * np.sin(n_periodes * 2. * np.pi * Xs / xs.max())) / (
                    -1 - amplitude * np.cos(
                n_periodes * 2. * np.pi * Xs / xs.max()) / xs.max() / n_periodes)
            Ys = Ys - (ys - Ys - amplitude * np.sin(n_periodes * 2. * np.pi * Ys / ys.max())) / (
                    -1 - amplitude * np.cos(
                n_periodes * 2. * np.pi * Ys / ys.max()) / ys.max() / n_periodes)

            errors_x = np.max(np.abs(Xs + amplitude * np.sin(n_periodes * 2. * np.pi * Xs / xs.max()) - xs))
            errors_y = np.max(np.abs(Ys + amplitude * np.sin(n_periodes * 2. * np.pi * Ys / ys.max()) - ys))

            if errors_x < tol and errors_y < tol:
                print("Coordinate correction converged in %i iterations" % i)
                break
        displacement_x = amplitude * np.sin(n_periodes * 2. * np.pi * Xs / xs.max())
        displacement_y = amplitude * np.sin(n_periodes * 2. * np.pi * Ys / ys.max())

        return xs, ys, Xs, Ys, displacement_x, displacement_y
    else:
        raise ValueError("formulation has to be lagrangian or eulerian")



