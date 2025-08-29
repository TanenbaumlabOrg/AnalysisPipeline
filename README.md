# Installation (conda environment)
```bash
conda env create --file=/environment/environment.yml
```

# Single cell live analysis
Notebook: `SingleCell/ND2Native_Stitch_Pipeline.ipynb`

A compilation of existing tools and custom functions for detection of fluorescent foci and extraction of fluorescence intensity over time for single cells in timelapse imaging.
This approach assumes large field of view input composed of yet unstitched overlapping tile images, in nd2 (Nikon Instruments Inc.) format.

Potential applications include monitoring of Influenza A virus (IAV) status over time in single cells by vRNP number and HA expression[^1].

Functonality can be extended by [SingleObjectGrid](https://github.com/TanenbaumLab/SingleObjectGrid) to create individual cell track videos.

## Core libraries
- ND2[^4] (image data loading & metadata parsing)
- PyImageJ[^5][^6] (FIJI launch)
    - MIST[^7] (Stitching)
    - TrackMate[^8] (Cell tracking)
- TrackAstra[^15] (Cell tracking)
- CellPose[^9] (Cell segmentation)
- SpotiFlow[^10] (Spot detection)
- Scikit-Image[^11] (Feature extraction)
- Napari[^12] (Data visualisation & exploration)

# Spot tracking
Notebook: `SpotTracking/CircTracker.ipynb`

A compilation of existing tools and custom functions for tracking fluorescent foci in timelapse imaging, including track cleanup and visualisation.

Potential applications include tracking of socRNA spots[^2], and RSV vRNPs[^3].

## Core libraries
- ND2[^4] (image data loading)
- tifffile[^13] (image data loading)
- Scikit-Image[^11] (Feature extraction)
- LapTrack[^14] (Spot tracking)
- Napari[^12] (Data visualisation & exploration)

# References
[^1]: Rabouw, H. H. et al. Mapping the complete influenza a virus infection cycle through single vRNP imaging. Biorxiv 2025.01.20.633851 (2025) doi:10.1101/2025.01.20.633851.

[^2]: Madern, M. F. et al. Long-term imaging of individual ribosomes reveals ribosome cooperativity in mRNA translation. Cell 188, 1896-1911.e24 (2025).

[^3]: Ratnayake, D. et al. Pre-assembly of biomolecular condensate seeds drives RSV replication. Biorxiv 2025.03.26.645422 (2025) doi:10.1101/2025.03.26.645422.

[^4]: https://github.com/tlambert03/nd2/tree/main

[^5]: Rueden, C. T. et al. PyImageJ: A library for integrating ImageJ and Python. Nat Methods (2022) doi:10.1038/s41592-022-01655-4.

[^6]: Schindelin, J. et al. Fiji: an open-source platform for biological-image analysis. Nature Methods 9, 676–682 (2012).

[^7]: Chalfoun, J. et al. MIST: Accurate and Scalable Microscopy Image Stitching Tool with Stage Modeling and Error Minimization. Sci Rep 7, 4988 (2017).

[^8]: Tinevez, J.-Y. et al. TrackMate: An open and extensible platform for single-particle tracking. Methods 115, 80–90 (2017).

[^9]: Stringer, C. & Pachitariu, M. Cellpose3: one-click image restoration for improved cellular segmentation. Nat. Methods 22, 592–599 (2025).

[^10]: Dominguez Mantes, A. et al. Spotiflow: accurate and efficient spot detection for fluorescence microscopy with deep stereographic flow regression. Nature Methods 22, 1495–1504 (2025).

[^11]: van der Walt, S. et al. scikit-image: image processing in Python. PeerJ 2, e453 (2014).

[^12]: Ahlers, J. et al. napari: a multi-dimensional image viewer for Python. Zenodo https://doi.org/10.5281/zenodo.8115575 (2023).

[^13]: https://github.com/cgohlke/tifffile

[^14]: Fukai, Y. T. & Kawaguchi, K. LapTrack: linear assignment particle tracking with tunable metrics. Bioinformatics 39, (2023).

[^15]: Gallusser, B. & Weigert, M. Trackastra: Transformer-based cell tracking for live-cell microscopy.