from recon.slope_integration import sparce_integration
from scipy.io import loadmat
import numpy as np
import os

# Integrate slopes to obtain displacement fields

data = loadmat("/home/sindreno/Rene/dataset/slopes.txt")  # x,y,frame

slopes_x = data["slope_x"]
slopes_y = data["slope_y"]

n_frames = slopes_x.shape[-1]

pixel_size = 2.94 / 1000.

disp_fields = []

crop_pts = -1

for i in np.arange(90,130):
    print("Integrating frame %i" % i)
    slope_y = slopes_x[:, :90, i]
    slope_x = slopes_y[:, :90, i]

    slope_x = np.pad(slope_x,pad_width=6,mode="edge")
    slope_y = np.pad(slope_y,pad_width=6,mode="edge")

    if crop_pts>0:
        slope_y = slope_y[crop_pts:-crop_pts,crop_pts:-crop_pts]
        slope_x = slope_x[crop_pts:-crop_pts,crop_pts:-crop_pts]

    disp_field = sparce_integration.int2D(slope_x, slope_y, pixel_size, pixel_size)
    disp_fields.append(disp_field)

np.save(os.getcwd() + "/disp_fields", np.array(disp_fields))
