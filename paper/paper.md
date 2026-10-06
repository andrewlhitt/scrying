---
title: 'SCRYiNG: A Python package for Simulating 2D Polycrystal Growth'
tags:
  - Python
  - simulation
  - materials science
authors:
  - name: Andrew L. Hitt
    # orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Ming Tang 
    orcid: 0000-0001-7194-3485
    affiliation: 1
affiliations:
 - name: Rice University, United States 
   index: 1
   ror: 008zs3103 
date: 4 October 2026 
bibliography: paper.bib

---

# Summary

Two-dimensional (2D) materials are a topic of substantial interest within materials science due to their unique physical, chemical, and electronic properties. Manufacturing 2D crystalline materials is often done via chemical vapor deposition (CVD), wherein material precursors are vaporized and allowed to deposit onto a substrate, forming a polycrystalline film. Unfortunately, isolating the impact of experimental changes in CVD growth conditions can be quite difficult, inhibiting efforts to improve the reliability and scalability of this process. Efficient simulation of 2D crystal growth can thus reveal valuable insight into the processing-structure relationship of these materials, enabling further optimization of the CVD process and allowing for the production of higher quality 2D material films. 

`SCRYiNG` (Simulated CRYstal Nucleation and Growth) is a lightweight simulation package designed to rapidly model the evolution of 2D polycrystalline microstructures. Unlike other 2D materials simulation techniques, which use physics-based mechanisms and track the positions of individual atoms, `SCRYiNG` instead uses deterministic growth mechanisms and simple geometric descriptions for crystals to replicate experimentally observed CVD crystal growth behaviors. This reduced-order formulation facilitates the efficient simulation of 2D microstructural evolution over large spatial and temporal scales. 

`SCRYiNG` offers a highly customizable Python interface capable of using both randomly generated or user-specified growth conditions; simulations can also be initialized from experimentally observed crystal configurations, allowing them to run in parallel with actual growth processes. The software is intended to support a variety of research efforts into 2D materials synthesis, including the execution of large-scale parametric studies, the generation of training data for machine learning models, and the prediction of microstructural evolution from experimental data.

# Statement of need

While CVD is widely used for 2D material synthesis, the quality of the films produced is highly sensitive to growth conditions such as reactor geometry, precursor ratios, and substrate temperatures. This multivariate sensitivity makes CVD growth experiments difficult to control, creating a barrier to optimizing the manufacturing process of valuable 2D materials. The performance of these 2D materials is highly dependent on their microstructure, which can be difficult and expensive to characterize ex situ. Computational techniques that can reliably predict that microstructure and reveal how growth conditions influence the resultant film can thus offer a basis for improving the CVD growth process [@Momeni:2018; @Momeni:2022; @Bets:2021]. 

One potential strategy would be to leverage machine learning to build models that can predict the microstructural evolution of crystal configurations extracted directly from the early stages of CVD growth processes. If sufficiently accurate, these predictions could be used to adjust growth conditions during the process, directing the system towards a desired outcome. However, the development of these predictive models would require large quantities of growth data covering a wide range of initial conditions and microstructural configurations. As producing such a dataset through experimentation or simulation would be prohibitively time-consuming and expensive, there is a need for a cheaper, more efficient alternative that still allows for a high degree of control over the growth conditions. 

# State of the field 

There are several well-established methods for simulating the growth of 2D materials, including molecular dynamics (MD) [@Kushima:2015; @Neyts:2013], kinetic Monte Carlo (KMC) [@Bets:2021], density functional theory (DFT) [@Zou:2013], and front tracking [@Lazar:2010]. These methods describe crystal growth as the movement of atoms via physical mechanisms, making them highly accurate but also computationally expensive, especially when scaled up to the mesoscale, when simulating long periods of time, or when running many independent simulations. This computational cost can cause problems when attempting to run the simulation in parallel with an actual CVD growth process as part of a digital twin strategy (several minutes of real time) or when generating training data for use in machine learning models (thousands of simulations). 

`SCRYiNG` addresses this computational challenge by simulating the evolution of 2D microstructures through a simple geometric mechanism instead of an atomistic process. Based on observations that individual crystals retain their shape throughout the CVD growth process [@Wang:2014], `SCRYiNG` depicts crystal growth as fixed shapes that slowly expand in diameter over time. The collision between expanding crystals (“impingement”) is deterministically resolved via simple rules, ensuring a consistent construction of shared grain boundaries. This focus on mesoscale morphology rather than nanoscale detail allows `SCRYiNG` to achieve significant performance gains while still reliably producing microstructures comparable to those created by KMC simulations [@Chen:2019], as shown in \autoref{fig:kmc_comparison}. Furthermore, its scale-independent and deterministic nature enables `SCRYING` to accurately predict the smooth (at mesoscale) boundaries between crystals, which KMC’s stochastic mechanisms will often predict to be jagged or rough.

![Three examples comparing the results of `SCRYiNG` and KMC simulations for given configurations of triangular MoS~2~ crystals, including (i) simulation parameters, (ii) early-stage structures produced with `SCRYiNG`, (iii) late-stage structures produced with `SCRYiNG`, and (iv) the same configuration produced with KMC (adapted with permission from [@Chen:2019]; copyright 2019 American Chemical Society). \label{fig:kmc_comparison}](figure_kmc_comparison.png)

# Software design
Simulations in `SCRYiNG` are structured like conventional lattice KMC simulations, with space discretized into a grid of pixels, and time divided into alternating nucleation and growth steps. As in KMC, the nucleation step is stochastic: the simulation determines how many new crystals will form (via sampling a Poisson distribution) and then places those new crystals randomly at available pixels. 

The growth step differs more substantially. `SCRYiNG` uses a mechanism comparable to front-tracking algorithms, where growth events are constrained to the existing surface of crystals. Each crystal tracks all pixels adjacent to its surface and calculates the minimum size a “phantom” crystal of a given position and orientation would need to be to contain that pixel (via vector projection). After the crystal’s size is updated, it checks each of its tracked pixels and expands into any that now lies within this phantom crystal. Newly adjacent pixels are then evaluated, and this process repeats until no new pixels can be added to the crystal. As multiple crystals might be able to expand into the same pixel during the same growth step, as with the extended-volume construction seen in Johnson-Mehl-Avrami-Kolmogorov models [@Avrami:1939; @Johnson:1939], impingement between crystals is resolved through simple, deterministic rules. This implementation offers substantial advantages in computational efficiency as only a small fraction of the simulation space needs to be checked for state changes (nucleation of a new crystal or expansion of an existing crystal) during each time step.   

`SCRYiNG` is intended to serve as a modular, extensible framework for predicting 2D crystal growth. Although early iterations were designed specifically for the triangular crystals of molybdenum disulfide (MoS~2~), `SCRYiNG` supports crystals of any convex polygonal shape, with innate support for both regular polygons and other user-specified shapes. `SCRYiNG`’s Simulator object can be easily configured for a variety of expected use cases, and its object-oriented design enables further user customization. `SCRYiNG` can also be configured to utilize imported experimental data, allowing it to simulate the microstructural evolution of real crystal growth processes. A graphical user interface with many of the core features is also provided to make the package accessible to users without extensive programming or modeling experience. 

# Research impact statement

`SCRYiNG` was initially developed to support studies of monolayer MoS~2~ films grown via CVD and imaged in situ with optical microscopy [@Zhang:2024]. `SCRYiNG` was incorporated into these efforts in several ways. First, `SCRYiNG` provided large quantities of simulated crystal growth data for the training and evaluation of machine learning models that could predict film quality. Second, `SCRYiNG` was used to predict the microstructural evolution of early-stage configurations of crystals extracted from experimental observations during the growth process [@Arifurrahman:2026]; by initializing a simulation directly from the experimental configuration, `SCRYiNG` is able to predict the future evolution of the microstructure in parallel with the actual experiment. Third, `SCRYiNG` was used to investigate effects that would be difficult or impractical to evaluate experimentally. For example, large-scale case studies were performed to isolate how different crystal shapes, growth anisotropies, or extents of epitaxial alignment influenced the resulting polycrystalline microstructure, while being able to hold all other parameters constant. 

`SCRYiNG` has also enabled applications beyond direct simulation and prediction. Its ability to quickly produce large quantities of microstructural evolutions facilitates the development of methods to infer otherwise inaccessible microstructural features from optical images. For example, the grain boundaries between crystals are generally not visible in optical images, but their locations can be resolved geometrically from the shapes of the impinging crystals. Comparison of the geometrically constructed grain boundaries to the ground truth (the actual grain boundaries generated via `SCRYiNG`) allows reconstruction methods to be systematically evaluated and improved. This strategy offers an analytical way to extract microstructural information from optical images as an alternative to specialized experimental techniques such as second-harmonic generation microscopy, at a fraction of the cost. 

# AI usage disclosure

No generative AI tools were used while creating and developing this software, writing the accompanying documentation, or producing any research results reported in this manuscript. The only use of generative AI was for recommendations about the grammar and clarity of the penultimate draft of the manuscript, some of which were incorporated manually. All text has been written and reviewed by the authors.  

# Acknowledgements

AH was supported by the Air Force Office of Scientific Research through Grant No. FA9550-24-1-0004, via a subcontract with Clarkson Aerospace Corporation. MT acknowledges support from the Air Force Office of Scientific Research through Award No. FA9550-23-1-0444. The authors also thank Dr. Zhili Hu for his early work that inspired the development of this project.

# References