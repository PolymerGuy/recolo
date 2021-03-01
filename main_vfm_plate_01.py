import matplotlib.pyplot as plt
from recon import plate_iso_qs_lin,field_from_disp_func,analydisp, Hermite16
import numpy as np
from scipy.ndimage import shift

def rms_diff(array1, array2):
    return np.sqrt(np.nanmean((array1-array2))**2.)

def mea_diff(array1, array2):
    return np.nanmean((array1-array2))


###
###
# plate and model parameters
mat_E = 70.e9         # Young's modulus [Pa]
mat_nu = 0.23        # Poisson's ratio []
n_pts_x = 151
n_pts_y = 151
plate_len_x = 0.2
plate_len_y = 0.2
plate_thick = 1e-3
press = 100.
dx = plate_len_x / float(n_pts_x)
dy = plate_len_y / float(n_pts_y)
###
mat_D = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu))   # flexural rigidity [N m]
mat_D11 = (plate_thick ** 3.) / 12. * mat_E / (1. - mat_nu ** 2.)
mat_D12 = (plate_thick ** 3.) / 12. * mat_E * mat_nu / (1. - mat_nu ** 2.)
# pressure reconstruction parameters


for n_pts_x in np.arange(141,151,10):
    error = []
    win_sizes = []
    n_pts_y=n_pts_x
    dx = plate_len_x / float(n_pts_x)
    dy = plate_len_y / float(n_pts_y)
    #for win_size in np.arange(4,50,2):
    for win_size in [24]:
        #for win_size in np.arange(4,40,4):
        #win_size = 24
        bend_stiff = mat_E * (plate_thick ** 3.) / (12. * (1. - mat_nu ** 2.))  # flexural rigidity [N m]

        deflection = analydisp.sinusoidal_load(press, plate_len_x, plate_len_y, bend_stiff)

        fields = field_from_disp_func(deflection, n_pts_x, n_pts_y, plate_len_x, plate_len_y)

        # define piecewise virtual fields
        virtual_fields = Hermite16(win_size, float(dx*win_size))

        recon_press,internal_energy = plate_iso_qs_lin(win_size, fields, mat_D11, mat_D12, virtual_fields)
        recon_press = shift(recon_press,shift=0.5,order=3)

        external_work = np.sum(recon_press*fields.deflection*dx*dy)
        print("Internal energy %f"%internal_energy)
        print("External energy %f"%external_work)

        rms_error = mea_diff(recon_press[win_size:-win_size,win_size:-win_size],fields.press[win_size:-win_size,win_size:-win_size])
        error.append(internal_energy)
        win_sizes.append(win_size)

    plt.plot(win_sizes,error,label="Image size: %i"%n_pts_x)
plt.ylabel("RMS-error [Pa]")
plt.xlabel("Window size [pixels]")
plt.legend(frameon=False)
plt.show()

print(recon_press.shape)
print(fields.press.shape)
#plt.figure()
plt.imshow(np.abs(recon_press-fields.press)[win_size:-win_size,win_size:-win_size])
plt.title("Recon - Truth")
plt.colorbar()

plt.figure()
plt.imshow(fields.press)
plt.colorbar()
plt.show()

plt.plot(recon_press[75,:],label="window size %i"%win_size)
plt.plot(fields.press[75,:],label="Ground truth")

plt.legend(frameon=False)
plt.show()
