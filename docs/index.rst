.. Recolo documentation master file, created by
   sphinx-quickstart on Mon May  3 13:35:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Recolo's documentation!
==================================

This python package provides tools for reconstuction of pressure loads using the Virtual Fields Method (VFM).




.. toctree::
   :maxdepth: 2
   :caption: Getting started:

   overview
   install
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Virtual experiments:

   deflectometryAbaqus

.. toctree::
   :maxdepth: 2
   :caption: Real experiments:

   impactHammer

.. toctree::
   :maxdepth: 2
   :caption: Theory:

   VFM

.. autosummary::
   :toctree: _autosummary
   :caption: API Documentation:
   :template: custom-module-template.rst
   :recursive:

   recolo
   

Citing us:
----------
If you use this toolkit as part of your research, please cite the following:


S. N. Olufsen, R. Kaufmann, E. Fagerholt, V. Aune (2022). RECOLO: A Python package for the reconstruction of surface pressure loads from kinematic fields using the virtual fields method. Journal of Open Source Software, 7(71), 3980, https://doi.org/10.21105/joss.03980
