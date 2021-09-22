VFM
===
Pressure Reconstruction
-----------------------
The Virtual fields method can be applied to solve a broad range of problems in solid mechanics [:cite:t:`Pierron2012`]. This python package is limited to the particular case of load reconstruction during fast transient dynamics.
The procedure for surface pressure reconstruction is based on the work of :cite:t:`Pierron2012` and :cite:t:`Kaufmann2019,Kaufmann2019`.

The dynamic equilibrium equations for a thin plate can be written written on weak form using the principal of virtual work as :cite:`dym1973solid`:

.. math::
   :nowrap:

    \begin{equation}
    \underbrace{\int\limits_{V} \rho ~ \boldsymbol{a} ~ \boldsymbol{u^*} dV}_{W_{inertial}^*} ~=~ \underbrace{ -\int\limits_{V} \boldsymbol{\sigma} : \boldsymbol{\varepsilon ^*} dV}_{W_{int}^*} + \underbrace{ \int\limits_{S} \overline{\boldsymbol{T}} \boldsymbol{u^*} ~dS + \int\limits_{V} \rho ~ \boldsymbol{F_{Vol}} ~ \boldsymbol{u^*} ~dV}_{W_{ext}^*} ~~~,
    \end{equation}

where :math:`W_{inertial}^*`, :math:`W_{int}^*` and :math:`W_{ext}^*` denotes the inertial virtual work, the internal virtual work and the external virtual work, respectively.

For the particular case of a thin plate represented by an isotropic linear elastic material, the principal of virtual work can be written using the Kirchoff-Love theory as:

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

where :math:`S` is the surface of the plate, :math:`p` is the pressure acting on the surface of the plate.
The deformation of the plate is given by the curvatures :math:`\kappa` and the acceleration :math:`a`. The density of the plate material is denoted :math:`\rho`, the plate thickness is denoted :math:`t_S`, and :math:`D_{xx}` and :math:`D_{xy}` are the plate bending stiffness matrix components. Virtual quantities are marked with :math:`^*`.

As local pressure values are of interest, the surface is divided into subdomains. By assuming a constant pressure distribution within each subdomain, the integrals in the above equation is reformulated as discrete sums and the pressure :math:`p` is solved for:

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

where :math:`N` is number of discrete surface elements.

The virtual fields based on 4-node Hermite 16 element shape functions :cite:t:`zienkiewicz1977` are available for pressure reconstruction, see :cite:t:`Pierron2012` for more details.

Bibliography
------------

.. bibliography::
   :style: plain
