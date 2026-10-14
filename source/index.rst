TIRAMISU
========

TIRAMISU models the spectra of exoplanet atmospheres with molecules out of local thermodynamic equilibrium (NLTE). It
computes molecular opacities from ExoMol line lists, solves the statistical equilibrium of each NLTE species' level
populations together with the radiation field (accelerated Λ iteration), and integrates the radiative transfer equation
in plane-parallel or spherical geometry to give emission and transmission spectra.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   overview

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/radiative_transfer

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api

Citing TIRAMISU
---------------

If you use TIRAMISU, please cite `Bowesman, Yurchenko, Al-Refaie & Tennyson (2026), ApJ 997, 203
<https://doi.org/10.3847/1538-4357/ae27a1>`__:

.. code-block:: bibtex

   @article{Bowesman2026TIRAMISU,
     author  = {Bowesman, Charles A. and Yurchenko, Sergei N. and Al-Refaie, Ahmed F. and Tennyson, Jonathan},
     title   = {{TIRAMISU}: Non-{LTE} Radiative Transfer for Molecules in Exoplanet Atmospheres},
     journal = {The Astrophysical Journal},
     volume  = {997},
     number  = {2},
     pages   = {203},
     year    = {2026},
     doi     = {10.3847/1538-4357/ae27a1},
   }

* :ref:`genindex`
* :ref:`search`
