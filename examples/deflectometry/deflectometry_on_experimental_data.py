# This allows for running the example when the repo has been cloned
import sys
import os

sys.path.extend([os.path.abspath(".")])

import matplotlib.pyplot as plt
import recolo

# Minimal example where the slope field is determined from grid images.
path_to_example_imgs = os.path.join(recolo.__path__[0], "tests/ExampleGridImages")
img_paths = recolo.list_files_in_folder(path_to_example_imgs, file_type=".tif", abs_path=True)

# Grid image description
grid_pitch = 5  # pixels
mirror_grid_distance = 1.37  # m

slope_x, slope_y = recolo.deflectomerty.slopes_from_images(path_to_example_imgs, grid_pitch, mirror_grid_distance)

# Plot results
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(slope_x[-1, :, :])
plt.title("Slopes around the the X-axis")

plt.subplot(1, 2, 2)
plt.imshow(slope_y[-1, :, :])
plt.title("Slopes around the Y-axis")

plt.tight_layout()
plt.show()
