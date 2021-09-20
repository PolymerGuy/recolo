VFM
===
Pressure Reconstruction
-----------------------
The virtual fields method can be applied to solve a broad range of problems in solid mechanics [] and only the highly specialized case of load reconstruction is presented here.
The procedure for pressure reconstruction is based on the work of Pierron et al. [] and :cite:t:`Kaufmann2019,Kaufmann2019`.

The dynamic equilibrium equations for a thin plate can be written written on weak for using the principal of virtual work as [dym1973solid]:

.. math::
   :nowrap:

    \begin{equation}
    \underbrace{\int\limits_{V} \rho ~ \boldsymbol{a} ~ \boldsymbol{u^*} dV}_{W_{inertial}^*} ~=~ \underbrace{ -\int\limits_{V} \boldsymbol{\sigma} : \boldsymbol{\varepsilon ^*} dV}_{W_{int}^*} + \underbrace{ \int\limits_{S} \overline{\boldsymbol{T}} \boldsymbol{u^*} ~dS + \int\limits_{V} \rho ~ \boldsymbol{F_{Vol}} ~ \boldsymbol{u^*} ~dV}_{W_{ext}^*} ~~~,
    \end{equation}

where :math:`W_{inertial}^*`, :math:`W_{int}^*` and :math:`W_{ext}^*` denotes the inertial virtual work, the internal virtual work and the external virtual work respectively.

The test specimen is assumed to behave like a thin plate in pure bending based on the small thickness and the expected loading conditions.
This allows describing the problem with the Kirchhoff-Love plate theory.
Since the plate material can be assumed as isotropic, homogeneous and linear elastic, the principle of virtual work for the present problem is expressed by:


.. math::
   :nowrap:

    \begin{equation}
	    \begin{aligned}
	    \int\limits_{S} p w^{*} dS ~ =
	    ~& D_{xx} \int\limits_{S} \left( \kappa _{xx} \kappa ^* _{xx} +\kappa _{yy} \kappa ^* _{yy} + 2 \kappa _{xy} \kappa ^* _{xy} \right) dS \\
	    +~&D_{xy} \int\limits_{S} \left( \kappa _{xx} \kappa ^* _{yy} +\kappa _{yy} \kappa ^*  _{xx} -2 \kappa _{xy} \kappa ^* _{xy} \right) dS
	    +~\rho ~ t_S \int\limits_{S} a ~w^{*} dS ~~~.
	    \end{aligned}
    \end{equation}

Here, :math:`S` is the surface area, :math:`p` the investigated pressure, :math:`D_{xx}` and :math:`D_{xy}` the plate bending stiffness matrix components, :math:`\kappa` the curvatures, :math:`\rho` the plate material density, :math:`t_S` the plate thickness, :math:`a` the acceleration, :math:`w^{*}` the virtual deflections and :math:`\kappa^{*}` the virtual curvatures.
In the present study :math:`D_{xx}`, :math:`D_{xy}`, :math:`\rho` and :math:`t_S` were known \textit{a prori}.
The surface slope measurements allowed calculating $\kappa$ and $a$.
The virtual fields :math:`w^{*}` and :math:`\kappa^{*}` were chosen according to theoretical and practical requirements of the investigate problem like continuity, boundary conditions and sensitivity to noise.
In order to obtain local information, the investigated surface was divided into subdomains.
Pressure is assumed constant over each subdomain, but by shifting the subdomain iteratively by one data point in each direction until the entire surface is covered, the spatial data can be oversampled and a high data point density can be obtained.
This subdomain is referred to as pressure reconstruction window (PRW) in the following.

By approximating the integrals with discrete sums one obtains:

.. math::
   :nowrap:

    \begin{equation}
    \begin{aligned}
    p ~ = ~
    \Biggl( ~&D_{xx} \sum\limits_{i = 1} ^{N}  \kappa ^{i} _{xx} \kappa ^{*i} _{xx} +\kappa _{yy}^{i} \kappa ^{*i}  _{yy} +2\kappa ^{i}_{xy} \kappa ^{*i} _{xy} \\
    +~&D_{xy} \sum\limits_{i = 1} ^{N} \kappa ^{i}_{xx} \kappa ^{*i} _{yy} +\kappa ^{i}_{yy} \kappa ^{*i} _{xx}
    -2\kappa ^{i}_{xy} \kappa ^{*i} _{xy}
    +~\rho ~ t_S \sum\limits_{i = 1} ^{N} a^{i} ~w^{*i} \Biggr) ~ \left( \sum\limits_{i = 1} ^{N} w^{*i} \right) ^{-1} ~~~,
    \end{aligned}
    \end{equation}

with the total number of discrete surface elements  $N$.

Virtual Fields
--------------
The virtual fields used in this study were defined using 4-node Hermite 16 element shape functions \citealp{zienkiewicz1977}.
They yield the required :math:`C^{1}` continuous virtual deflections.
Furthermore they allow eliminating the unknown contributions of virtual work over the domain boundaries because the obtained virtual displacements and curvatures vanish at the borders.
9 nodes were defined for each Hermite element in the present study.
All degrees of freedom were set to zero except for the virtual deflection of the center node, which was set to 1.
Figure \ref{fig:hermitevirtual} shows the virtual fields used in this study.
Their full formulation and an implementation example is given in \citealp{Pierron2012}.
The chosen PRW size is an important processing parameter in terms of spatial resolution, noise sensitivity and systematic error.
The PRW acts as a low-pass filter, but for large sizes it tends to result in an underestimation of local pressure amplitudes.
This is particularly true if the spatial scale of the investigated pressure event is small in comparison to the PRW.
A methodology to determine the optimal PRW size by simulating experiments using finite element simulations and artificial grid deformation is described in \citealp{Kaufmann2019}.

.. bibliography::
   :style: unsrt

