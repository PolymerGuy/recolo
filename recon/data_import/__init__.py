import os
import numpy as np
from natsort import natsorted
from collections import namedtuple


def list_files_in_folder(path, file_type=".rpt"):
    """ List all files with a given extension for a given path. The output is sorted
        Parameters
        ----------
        path : str
            Path to the folder containing the files
        file_type : str
            The file extension ex. ".rpt"
        Returns
        -------
        list
            A list of sorted file names
        """
    return natsorted([file for file in os.listdir(path) if file.endswith(file_type)])



AbaqusData = namedtuple("AbaqusSimulation",["disp_fields", "accel_fields", "times", "plate_len_x", "plate_len_y","npts_x","npts_y"])

def load_abaqus_rpts(path_to_rpts,use_only_img_ids=None):

    rpt_file_paths = list_files_in_folder(path_to_rpts,file_type=".rpt")
    print("Reading %i Abaqus .rpt files"%len(rpt_file_paths))

    disp_fields = []
    accel_fields = []
    times = []

    if use_only_img_ids is not None:
        rpt_file_paths = [path for i,path in enumerate(rpt_file_paths) if i in use_only_img_ids]

    for file_name in rpt_file_paths:
        print("Reading: %s "%file_name)
        path_to_rpt = os.path.join(path_to_rpts,file_name)
        field_data = np.genfromtxt(path_to_rpt, dtype=float,
                                   skip_header=19)

        time = np.genfromtxt(path_to_rpt,dtype=str,skip_header=8,max_rows=1)[-1]

        node_label = field_data[:, 0]
        node_coord_x = field_data[:, 1]
        node_coord_y = field_data[:, 2]
        node_disp_z = field_data[:, 3]
        node_acceleration_z = field_data[:, 4]

        # All data is assumed to be sampled on a square grid
        seed = int(node_disp_z.size ** 0.5)

        plate_len_x = (node_coord_x.max() - node_coord_x.min()) * 1e-3
        plate_len_y = (node_coord_y.max() - node_coord_y.min()) * 1e-3

        disp_field = -node_disp_z.reshape((seed, seed)) * 1e-3
        accel_field = -node_acceleration_z.reshape((seed, seed)) * 1e-3

        disp_fields.append(disp_field)
        accel_fields.append(accel_field)
        times.append(float(time))
    npts_x = np.shape(disp_fields)[1]
    npts_y = np.shape(disp_fields)[2]

    #return np.array(disp_fields), np.array(accel_fields), times, plate_len_x, plate_len_y
    return AbaqusData(np.array(disp_fields), np.array(accel_fields), times, plate_len_x, plate_len_y,npts_x,npts_y)
