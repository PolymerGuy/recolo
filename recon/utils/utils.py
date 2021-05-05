import os
from natsort import natsorted

def list_files_in_folder(path, file_type=".rpt", abs_path=True):
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
    file_names = natsorted([file for file in os.listdir(path) if file.endswith(file_type)])
    if abs_path:
        return [os.path.join(path,file_name) for file_name in file_names]
    else:
        return file_names