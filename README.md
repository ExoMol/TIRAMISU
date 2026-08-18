# TIRAMISU

Solves the coupled radiative transfer (RT) and statistical equilibrium (SE) equations for an atmosphere containing an 
arbitrary mixture of absorbers, divided into a set of discrete layers. Species can be treated in LTE or non-LTE; all 
non-LTE species are solved simultaneously and are fully coupled to emission and absorption from all other sources 
(whether non-LTE or LTE).

This code can be referenced by citing the article [Bowesman et al. (2026)](https://doi.org/10.3847/1538-4357/ae27a1).
Significant changes have been made since this publication, particularly regarding the handing of an arbitrary number of
coupled non-LTE species as well as significant performance increases resulting in a factor 100 speed-up. An update
detailing this work is forthcoming.

The SE equations are solved with the use of an approximate Lambda operator and are fully preconditioned (accounting for 
overlap between all emission and absorption features) using the formalism of [Rybicki & Hummer 
(1992)](https://ui.adsabs.harvard.edu/abs/1992A%26A...262..209R/abstract). These equations have been modified to include
overlap from continuum features. Here we perform Gauss-Seidel iterations comprising inward and outward propagations of 
radiation and associated quantities through the atmosphere. Short-characteristics solutions are taken to solve the RT
during these passes using Bezier spline interpolants, which are known to be more accurate than the more common parabolic 
interpolants. These Bezier interpolants are also used in the construction of the approximate Lambda operator.

A fixed number of LTE layers can be specified for the inner region of the atmosphere. This saves having to solve the 
SE equations for layers where the species will likely be in LTE (i.e.: high pressure regions > 1 bar).

Stellar radiation incident on the atmosphere can be configured and relies on the 
[Phoenix4All](https://github.com/taurex-space/phoenix4all) python package to fetch synthetic stellar spectra.  

To enable solution to the SE equations, collisional rates for all desired states must be configured in the colchem.py 
module. Currently collisional rate data are only confiured for OH, CO and H2O. The measurement or calculation of such 
rates is extremely time-consuming and DOIs are provided in the module for the relevant sources: please cite these if you
use their data!
