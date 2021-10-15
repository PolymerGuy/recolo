import numpy as np
import wget
import logging
from recolo import list_files_in_folder
import os
import pathlib

cwd = pathlib.Path(__file__).parent.resolve()

"""" This module provides data from an impact hammer experiment where the hammer was knocked onto a steel plate while 
the deformation of the plate was measured using using a deflectometry setup. 

The data consists of force readings from the impact hammer as well as the images of the mirror plate during deformation.

The dataset is hosted at https://dataverse.no/ and is automatically downloaded when ImpactHammerExperiment is 
instantiated the first time. """

link_to_data = "https://dataverse.no/api/access/datafile/108510"
hammer_force_filename = "hammer_force.txt"
hammer_time_filename = "hammer_time.txt"
data_folder = os.path.join(cwd, "impact_hammer_data")


class ImpactHammerExperiment(object):
    #def __init__(self, path_to_images="/home/sindreno/PressureRecon/hammer_publish"):
    def __init__(self, path_to_images=None):
        """
        Data from an experiment with an impact hammer and a deflectometry setup.

        Parameters
        ----------
        path_to_images : str
            A path to the preferred location where the data is stored.
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        if path_to_images:
            self.path_to_img_folder = path_to_images
        else:
            self.path_to_img_folder = data_folder
            self.logger.info("No image path was specified.\nImage folder is: %s" % self.path_to_img_folder)

        if self.__is_downloaded__():
            self.logger.info("Experimental data was found locally")
        else:
            try:
                self.logger.info("Downloading experimental data")
                self.__download_dataset__()
            except Exception as e:
                self.logger.exception(e)
                raise

        self.path_to_imgs = list_files_in_folder(self.path_to_img_folder, file_type=".tif", abs_path=True)
        self.n_images = len(self.path_to_imgs)

    def __is_downloaded__(self):
        hammer_data_path = os.path.join(self.path_to_img_folder, hammer_force_filename)
        return os.path.exists(hammer_data_path)

    def __download_dataset__(self):
        if not os.path.exists(self.path_to_img_folder):
            self.logger.info("Creating folder %s"%self.path_to_img_folder)
            os.makedirs(self.path_to_img_folder)
        wget.download(link_to_data, out=self.path_to_img_folder, bar=True)

    def hammer_data(self):
        """
        Impact hammer data

        Returns
        -------
        force, disp : ndarray, ndarray
        """
        path_to_impact_hammer_force = os.path.join(self.path_to_img_folder, hammer_force_filename)
        path_to_impact_hammer_time = os.path.join(self.path_to_img_folder, hammer_time_filename)
        force = np.genfromtxt(path_to_impact_hammer_force)
        time = np.genfromtxt(path_to_impact_hammer_time)
        return force, time
