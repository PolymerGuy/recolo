from recon import sparce_integration
from scipy.io import loadmat
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

data = loadmat("/home/sindreno/Rene/dataset/slopes.txt") #x,y,frame

slopes_x = np.moveaxis(data["slope_x"], -1, 0)
slopes_y = np.moveaxis(data["slope_y"], -1, 0)

n_frames = slopes_x.shape[0]

#frame_id = 110

disps = []

for i in np.arange(90,130):
    print("Integrating frame %i"%i)
    slope_x = slopes_x[i,:90,:90]
    slope_y = slopes_y[i,:90,:90]

    slope_x = gaussian_filter(slope_x,sigma=0)
    slope_y = gaussian_filter(slope_y,sigma=0)

    disp = sparce_integration.int2D(slope_x, slope_y, 0., 1., 1.)
    disps.append(disp)