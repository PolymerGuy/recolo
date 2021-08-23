import matplotlib.pyplot as plt

from recon.deflectomerty import slopes_from_images
from recon.utils import list_files_in_folder

path = "/home/sindreno/gridmethod_Rene/images_full_2"
img_paths = list_files_in_folder(path, file_type=".tif", abs_path=True)

grid_pitch = 5  # pixels
grid_pitch_len = 5.88 / 1000.  # m

mirror_grid_distance = 1.37  # m

pixel_size_on_grid_plane = grid_pitch_len / grid_pitch

ref_img_ids = range(50,80)
use_imgs = range(110, 120)

slope_x, slope_y = slopes_from_images(path, grid_pitch, mirror_grid_distance, ref_img_ids=ref_img_ids,
                                      only_img_ids=use_imgs, crop=(10, -10, 0, -1))

fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(slope_x[:, :, -1])
plt.title("Slopes around the the X-axis")

plt.subplot(1, 2, 2)
plt.imshow(slope_y[:, :, -1])
plt.title("Slopes around the Y-axis")

plt.tight_layout()
plt.show()
