---
title: 'RECOLO: A Python package for the reconstruction of surface pressure loads from kinematic fields using the virtual fields method'
tags:
  - Python
  - Virtual fields method
  - Load reconstruction
  - Parameter identification
authors:
  - name: Sindre Nordmark Olufsen
    affiliation: 1,2
  - name: Rene Kaufmann
    affiliation: 1
  - name: Egil Fagerholt
    affiliation: 1
  - name: Vegard Aune
    affiliation: 1,2
affiliations:
 - name: Structural Impact Laboratory (SIMLab), Department of Structural Engineering, NTNU - Norwegian University of Science and Technology, Trondheim, Norway
   index: 1
 - name: Centre for Advanced Structural Analysis (CASA), NTNU, Trondheim, Norway
   index: 2
date: 13 September 2021
bibliography: paper.bib
---

# Summary
In experimental mechanics, it is well known that it is very challenging to measure non-intrusive surface pressures on blast-loaded structures even in controlled, laboratory environments (see e.g., [Pannell2021]). Still, it is of utmost importance to provide structural engineers with a detailed knowledge of loads and underlying physics to understand and predict how structures respond during extreme loading events. When pressure loads are imposed on a deformable structure, fluid-structure interaction (FSI) effects are known to cause non-trivial loading scenarios which are difficult to quantify (see e.g., [Aune2021]).
This project aims at reconstructing the full-field surface pressure loads acting on a deforming structure employing the virtual fields method on full-field kinematic measurements [Kaufmann2019,Kaufmann2020]. Provided that the properties of the structure are known,
the pressure loading can be reconstructed both temporally and spatially. To understand the capabilities and error sources
associated with the reconstruction methodology, the package provides the scientific tools for performing virtual experiments based on analytical data or data from finite element simulations. Tools for performing deflectometry using the grid method are also provided.

``Recolo`` is a Python package that allows for the reconstruction of surface pressure loads acting on plated structures by using the virtual fields method [Pierron2012].
Other VFM toolkits such as PeriPyVFM are readily available but are focused on different applications.

``Recolo`` contains a collection of science-driven functions which allows the user to perform virtual experiment on synthetically generated data as well
 as performing pressure reconstruction on experimental datasets. The pressure reconstruction algorithm is based on the work by Kaufmann et al. [@Kaufmann2019,Kaufmann2020].
The implementation is based on numerical operations provided by NumPy [@Numpy] and SciPy [@SciPy] as well as visualization by Matplotlib [@Matplotlib].

``Recolo`` was implemented to determine the blast loading acting on plated structures in a purpose-built shock tube apparatus at SIMLab, NTNU [Aune2016]. The methodology developed in this project is directly applicable to obtain new, unique insight into surface pressure distributions on plated structures subjected to blast loading. This project is part of the ongoing research within the SIMLab research group at NTNU.

# Acknowledgements
The authors gratefully appreciate the financial support from the Research Council of Norway (RCN) through the Centre for Advanced Structural Analysis (SFI-CASA RCN Project No. 237885) and the SLADE KPN project (RCN Project No. 294748). The financial support by the Norwegian Ministry of Justice and Public Security is also greatly appreciated.

# References