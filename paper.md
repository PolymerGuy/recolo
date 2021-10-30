---
title: 'RECOLO: A Python package for the reconstruction of surface pressure loads from kinematic fields using the virtual fields method'
tags:
  - Python
  - Virtual fields method
  - Load reconstruction
  - Parameter identification
authors:
  - name: Sindre Nordmark Olufsen
    affiliation: "1, 2"
  - name: Rene Kaufmann
    affiliation: 1
  - name: Egil Fagerholt
    affiliation: 1
  - name: Vegard Aune
    affiliation: "1, 2"
affiliations:
 - name: Structural Impact Laboratory (SIMLab), Department of Structural Engineering, NTNU - Norwegian University of Science and Technology, Trondheim, Norway
   index: 1
 - name: Centre for Advanced Structural Analysis (CASA), Department of Structural Engineering, NTNU - Norwegian University of Science and Technology, Trondheim, Norway
   index: 2
date: 30 October 2021
bibliography: paper.bib
---

# Summary
In experimental mechanics, it is well known that it is very challenging to conduct non-intrusive measurements of surface pressures on blast-loaded structures, even in controlled, laboratory environments (see e.g., [@Pannell2021]). Still, it is of utmost importance to provide structural engineers with a detailed knowledge of loads, understand the underlying physics and predict how structures respond during extreme loading events. When pressure loads are imposed on a deformable structure, fluid-structure interaction (FSI) effects are known to cause non-trivial loading scenarios which are difficult to quantify (see e.g., [@Aune2021]).
This project aims at reconstructing the full-field surface pressure loads acting on a deforming structure employing the virtual fields method (VFM) on full-field kinematic measurements [@Kaufmann2019].
Even though the current framework is limited to reconstructions of full-field pressure information from deformation data of thin plates in pure bending, it also allows for future extensions to other loading and deformation scenarios.
Provided that the properties of the structure are known,
the pressure loading can be reconstructed both temporally and spatially. To understand the capabilities and accuracy
associated with the reconstruction methodology, the package provides the scientific tools for performing virtual experiments based on analytical data or data from finite element simulations. The current implementation is based on the deflectometry technique, using the grid method to obtain the deformation measurements and corresponding kinematics of the structure.

``Recolo`` is a Python package that allows for the reconstruction of surface pressure loads acting on plated structures by using the VFM [@Pierron2012].
Other VFM toolkits such as PeriPyVFM are readily available but are focused on different applications.

``Recolo`` contains a collection of science-driven functions which enables the user to perform virtual experiments on synthetically generated data as well
 as performing pressure reconstruction on experimental datasets. The pressure reconstruction algorithm is based on the work by [@Kaufmann2019].
The implementation is based on numerical operations provided by NumPy [@Numpy] and SciPy [@SciPy] as well as visualization by Matplotlib [@Matplotlib].



# Statement of need
``Recolo`` was established to quantify the blast loading acting on plated structures in a purpose-built shock tube apparatus at SIMLab, NTNU [@Aune2016]. No open-source software known to the authors provided the functionality neccessary to perform pressure load reconstruction based on the kinematics of the specimen subjected to the load, motivating the ``Recolo` project.

The methodology developed in this project is directly applicable to obtain new, unique insight into surface pressure distributions on plated structures subjected to blast loading. This project is part of the ongoing research within the SIMLab research group at NTNU.

# Acknowledgements
The authors gratefully appreciate the financial support from the Research Council of Norway (RCN) through the Centre for Advanced Structural Analysis (SFI-CASA RCN Project No. 237885) and the SLADE KPN project (RCN Project No. 294748). The financial support by the Norwegian Ministry of Justice and Public Security is also greatly appreciated.

# References