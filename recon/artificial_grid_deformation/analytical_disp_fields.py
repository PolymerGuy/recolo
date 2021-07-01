import numpy as np

def harmonic_disp_field(disp_amp, disp_period, disp_n_periodes, formulation="Lagrangian"):
    x = np.arange(disp_n_periodes * disp_period, dtype=float)
    y = np.arange(disp_n_periodes * disp_period, dtype=float)

    xs, ys = np.meshgrid(x, y)
    Xs, Ys = np.meshgrid(x, y)

    displacement_x = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * xs / xs.max())
    displacement_y = disp_amp * np.sin(disp_n_periodes * 2. * np.pi * ys / ys.max())

    if formulation == "eulerian":
        return xs, ys, (xs - displacement_x), (ys - displacement_y), displacement_x, displacement_y

    elif formulation == "lagrangian":
        tol = 1.e-12
        for i in range(20):
            Xs = Xs - (xs - Xs - disp_amp * np.sin(disp_n_periodes * 2. * np.pi * Xs / xs.max())) / (
                    -1 - disp_amp * np.cos(
                disp_n_periodes * 2. * np.pi * Xs / xs.max()) / xs.max() / disp_n_periodes)
            Ys = Ys - (ys - Ys - disp_amp * np.sin(disp_n_periodes * 2. * np.pi * Ys / ys.max())) / (
                    -1 - disp_amp * np.cos(
                disp_n_periodes * 2. * np.pi * Ys / ys.max()) / ys.max() / disp_n_periodes)

            errors_x = np.max(np.abs(Xs + disp_amp * np.sin(disp_n_periodes * 2. * np.pi * Xs / xs.max()) - xs))
            errors_y = np.max(np.abs(Ys + disp_amp * np.sin(disp_n_periodes * 2. * np.pi * Ys / xs.max()) - ys))

            if errors_x < tol and errors_y < tol:
                print("Coordinate correction converged in %i iterations" % i)
                break
        return xs, ys, Xs, Ys, displacement_x, displacement_y
    else:
        raise ValueError("formulation has to be lagrangian or eulerian")



