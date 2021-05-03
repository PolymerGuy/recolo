from recon.slope_integration import sparce_integration
import numpy as np
import matplotlib.pyplot as plt

# Field data
n_pts_x, n_pts_y = 61, 61
plate_len_x, plate_len_y, = 1.5, 1.5
pressure_peak_amp = 1.

dx = plate_len_x / n_pts_x
dy = plate_len_y / n_pts_y

# Reconstruction parameters
int_const = 0.

# Data cropping is set as a percentage
crop_factor = 0.88
crop = int((1-crop_factor)*n_pts_x/2)
print("Cropping %i pixels on each side"%crop)

# Generate displacement field and gradients
xs, ys = np.meshgrid(np.linspace(0., 1., n_pts_x), np.linspace(0., 1., n_pts_y))
disp_field = pressure_peak_amp * np.sin(np.pi * xs) * np.sin(np.pi * ys)

gradient_x, gradient_y = np.gradient(disp_field, dx, dy)
gradient_x_cropped = gradient_x[crop:-crop,crop:-crop]
gradient_y_cropped = gradient_y[crop:-crop,crop:-crop]

# Reconstruct
disp_field_from_slopes = sparce_integration.int2D(gradient_x_cropped, gradient_y_cropped, dx, dy)

# Crop for comparison
cropped_disp = disp_field[crop:-crop,crop:-crop]


plt.imshow(cropped_disp)
plt.title("Cropped correct answer")
plt.figure()
plt.imshow(disp_field_from_slopes)
plt.title("Integrated from slopes")
plt.show()
