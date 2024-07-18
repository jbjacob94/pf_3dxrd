# point-fit 3DXRD (pf_3dxrd)


## Description
This repository contains tools to process scanning 3D X-ray diffraction (3DXRD) data on a per-pixel basis, based on the relocation of diffracted X-rays in the sample using Friedel pairs. Pairs of diffracted (h,k,l) (-h,-k,-l) reflections arising from the same point in a crystal are identified in the dataset. The symmetry properties of these pairs are used to correct the diffraction (2-theta) angle and find the coordinates of the diffracting region in the sample reference frame. Friedel pairs are then assigned to a pixel on a 2D map based on their sample coordinates, which allows per-pixel fitting of phase and lattice vectors (local indexing). 

The pf_3dxrd package is built upon ImageD11 and contains the following files:
### modules
- friedel_pairs.py: Identification of Friedel pairs in a dataset and geometry corrections based on Friedel pairs symmetry
- crystal_structure.py: A class to store crystal structure information, loaded from a cif file
- phase_mapping.py: A class to perform phase mapping on a pixel grid
- peak_mapping.py: Peaks-to-pixel and peaks-to-grains mapping. Manage labels to assign diffraction peaks to pixels/grains and grains/pixels to peaks. Also includes functions for refining grain UBIs (lattice vector matrices)
- pixelmap.py: A class to store outputs from local indexing on a pixel grid and make 2D plots.
- utils.py: general functions used in other modules, mainly to work on ImageD11 columnfiles

### scripts
Scripts to be executed in command lines from a terminal. Makes bathc processing of a series of peakfiles more handy. 
- find_friedel_pairs.py : Run Friedel pairs search for all scans in a peakfile/peaks table, using parallelization on multiple cores for faster computation
- local_indexing.py : Run local indexing (find lattice vectors matrix UBI on each pixel) on a peakfile, using parallelization on multiple cores

### NB_tutorials
A detailed tutorial organized in a series of Jupyter notebooks, to obtain 2D orientation and strain /stress maps from a raw peakfile. 

### NB_examples
Shortened versions of the Jupyter notebooks in NB_tutorials, which can be more easily adapted to build your processing workflow

## Installation
So far, pf_3dxrd is not a proper package and only consists of a collection of python modules and scripts.
To use it, just clone the repository to your project folder. 

```git clone git@github.com:jbjacob94/pf_3dxrd.git```

## Dependencies
To be included in setup.py file. 

pf_3dxrd is built upon ImageD11 (https://github.com/FABLE-3DXRD/), which needs to be installed with all its dependencies. 
See https://github.com/FABLE-3DXRD/

crystal_structure.py module relies on orix, Dans_diffraction and diffpy.structure modules, which can all be installed with pip:

```python -m pip install Dans_diffraction diffpy.structure orix```

## Usage
To use the pf_3dxr module, just import it into your project file. 

## Documentation
Details about the processing pipeline are explained in the tutorials. For specific information about each function, read the docstrings.

## License

## Credits
Jean-Baptiste Jacob (jbjacob94)


# Contact
j.b.jacob@mn.uio.no
