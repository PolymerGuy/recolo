---
title: 'RECOLO: A Python package for reconstruction of pressure fields from kinematic fields using the virtual fields method'
tags:
  - Python
  - Virtual fields method
  - Load reconstruction
  - Parameter identification
authors:
  - name: Sindre Nordmark Olufsen
    affiliation: 1
  - name: Rene Kaufmann
    affiliation: 1
  - name: Vegard Aune
    affiliation: 1
affiliations:
 - name: Structural Impact Laboratory (SIMLab), Department of Structural Engineering, NTNU, Norwegian University of Science and Technology, NO-7491 Trondheim, Norway
   index: 1
date: 24 March 2021
bibliography: paper.bib
---

# Summary
The ability to determine the load applied to a structure without interfering with the experiment is crucial in experimental mechanics.
Fluid-structure-interaction effects caused by interaction between the deformation of the structure and the applied pressure
are known to cause non-trivial loading scenarios which are difficult to quantify. This project aims at reconstructing the
pressure load acting on a deforming structure by means of the virtual fields method [Kaufmann2019,Kaufmann2020]. If the properties of the structure, here being a plate, is known,
the pressure loading can be reconstructed both temporally and spatially. In order to understand the capabilities and error sources
associated with the technique, the package provides tools for performing virtual experiments based on analytical data or data from finite element simulations. Tools for performing deflectometry using the grid method are also provided.

``Recolo`` is a Python package that allows for the reconstruction of pressure loads acting on plated structures by using the virtual fields method [].
Other VFM toolkits such as PeriPyVFM are readily available but are focused on different applications.

``Recolo`` contains a collection of high-level functions which allows the user to perform virtual experiment on synthetically generated data as well
 as performing pressure reconstruction on experimental datasets. The pressure reconstruction algorithm is based on the work by Kaufmann et al. [@Kaufmann2019,Kaufmann2020].
The implementation is highly on numerical operations provided by NumPy [@Numpy] and SciPy [@SciPy] as well as visualization by Matplotlib [@Matplotlib].

``Recolo`` was implemented to determine the blast pressure load acting on plated structures a shock-tube apparatus.
This project is a part of the ongoing research within the SFI CASA research group at NTNU and is a key component in the pursuit of better understanding of fluid-structure-interaction effects in blast load scenarios.

# Acknowledgements
The author gratefully appreciates the financial support from the Research Council of Norway through the Centre for Advanced Structural Analysis, Project No. 237885 (SFI-CASA).

# References