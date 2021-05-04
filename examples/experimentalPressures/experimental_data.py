import numpy as np
def read_exp_press_data(experiment="open channel"):
    start = 25500
    end = 26200

    if experiment == "open channel":
        data = np.genfromtxt("./experimentalPressures/trans_open_1.txt", skip_header=20)
    else:
        data = np.genfromtxt("./experimentalPressures/trans_half_1.txt", skip_header=20)
    time = data[start:end, 0] * 1.e-3
    time = time - time[0]
    press = data[start:end, :] / 10.
    return press - press[0, :], time