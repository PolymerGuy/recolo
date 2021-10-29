import numpy as np
import wget
import logging
import requests
from bs4 import BeautifulSoup
from urllib import parse
from recolo import list_files_in_folder

import os
import pathlib

cwd = pathlib.Path(__file__).parent.resolve()

"""" This module provides data from an impact hammer experiment where the hammer was knocked onto a steel plate while 
the deformation of the plate was measured using using a deflectometry setup. 

The data consists of force readings from the impact hammer as well as the images of the mirror plate during deformation.

The dataset is hosted at https://dataverse.no/ and is automatically downloaded when ImpactHammerExperiment is 
instantiated the first time. 
"""


def find_dataverse_files(repo_url, params={}):
    dataverse_base = '{uri.scheme}://{uri.netloc}/'.format(uri=parse.urlparse(repo_url))
    response = requests.get(repo_url, params=params)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()
    soup = BeautifulSoup(response_text, 'html.parser')
    file_urls = [parse.urljoin(dataverse_base, node.get('href')) for node in soup.find_all('a')]
    return file_urls


class DataverseDataSet(object):

    def __init__(self, DOI, path_to_data=None, new_data_folder_name="data"):
        """
        Download data from an Dataverse repository.

        Parameters
        ----------
        links_to_data : list
            Paths to all files in the repo
        path_to_data : str
            A path to the preferred location where the data is stored.
        new_data_folder_name : str
            The name of the new data folder
        """
        self._logger_ = logging.getLogger(self.__class__.__name__)
        self._readme_file_name_ = "00_ReadMe.txt"
        self._dataverse_file_repo ='https://dataverse.no/api/datasets/:persistentId/dirindex?persistentId=doi:'
        self.DOI = DOI
        self.repo_url = self._dataverse_file_repo + DOI


        self._file_urls_ = find_dataverse_files(self.repo_url)
        self._logger_.info("Found %i files in the repo" % len(self._file_urls_))
        if path_to_data:
            self.path_to_data_folder = path_to_data
        else:
            self.path_to_data_folder = os.path.join(cwd, new_data_folder_name)
            self._logger_.info("No image path was specified.\nImage folder is: %s" % self.path_to_data_folder)

        if self.__data_is_downloaded__():
            self._logger_.info("Experimental data was found locally")
        else:
            try:
                self._logger_.info("Downloading experimental data")
                self.__download_dataset__()
            except Exception as e:
                self._logger_.exception(e)
                raise
        self.file_paths = list_files_in_folder(self.path_to_data_folder, file_type='', abs_path=True)

    def __data_is_downloaded__(self):
        readme_file_path = os.path.join(self.path_to_data_folder, self._readme_file_name_)
        return os.path.exists(readme_file_path)

    def __download_dataset__(self):
        if not os.path.exists(self.path_to_data_folder):
            self._logger_.info("Creating folder %s" % self.path_to_data_folder)
            os.makedirs(self.path_to_data_folder)
        for link_to_data in self._file_urls_:
            try:
                wget.download(link_to_data, out=self.path_to_data_folder)
            except Exception as e:
                raise


class ImpactHammerExperiment(DataverseDataSet):
    def __init__(self, path_to_data=None):
        """
        Data from an experiment with an impact hammer and a deflectometry setup.

        Parameters
        ----------
        path_to_data : str
            A path to the preferred location where the data is stored.
        """
        self._hammer_force_filename_ = "hammer_force.txt"
        self._hammer_time_filename_ = "hammer_time.txt"
        DOI = '10.18710/WA5YCF'

        super().__init__(DOI, path_to_data=path_to_data, new_data_folder_name="impact_hammer_data")

        self.img_paths = list_files_in_folder(self.path_to_data_folder, file_type=".tif", abs_path=True)
        self.n_images = len(self.img_paths)

    def hammer_data(self):
        """
        Impact hammer data

        Returns
        -------
        force, disp : ndarray, ndarray
        """
        path_to_impact_hammer_force = os.path.join(self.path_to_data_folder, self._hammer_force_filename_)
        path_to_impact_hammer_time = os.path.join(self.path_to_data_folder, self._hammer_time_filename_)
        force = np.genfromtxt(path_to_impact_hammer_force)
        time = np.genfromtxt(path_to_impact_hammer_time)
        return force, time
