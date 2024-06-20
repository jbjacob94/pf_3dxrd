# point-fit 3DXRD (pf_3dxrd)


## Description
This package contains tools to process scanning 3D X-ray diffraction (3DXRD) data on a per-pixel basis, based on relocation of diffracted X rays in the sample using Friedel pairs. Pairs of diffracted (h,k,l) (-h,-k,-l) reflections arising from the same point in a crystal are identified in the dataset. Symmetry properties of these pairs are used to correct the diffraction (2-theta) angle and find the coordinates of the diffracting region in the sample reference frame. Friedel pairs are then assigned to a pixel on a 2D map based on their sample coordinates, which allows per-pixel fit of lattice vectors. 

The pf_3dxrd package is built upon ImageD11 and contains the following modules:
- friedel_pairs.py: Identification of Friedel pairs in a dataset and geometry corrections based on Friedel pairs symmetry
- crystal_structure.py: class to store crystal structure information, loaded from a cif file
- peak_mapping.py: Peaks to 2D pixel map and peaks to grains mapping. Manage labels to assign diffraction peaks to pixels / grains, select peaks belonging to a group of pixels / grain
- pixelmap.py: class to map and plot data on a 2D pixel grid, allowing per-pixel data analysis and processing
- utils.py: general functions to work with ImageD11 peakfiles

### Installation

So far, pf_3dxrd is not a proper package and only consists of a collection of python modules. 

To use it, just download the full folder, paste it in your project directory and add import pf_3dxrd to your code.

### Dependencies
To be included in setup.py file. 

pf_3dxrd is built upon ImageD11 (https://github.com/FABLE-3DXRD/), which needs to be installed with all its dependencies. 
available at https://github.com/FABLE-3DXRD/.

crystal_structure.py module also relies on orix, Dans_diffraction and diffpy.structure modules, which can all be downloaded with pip.


## Usage
pf_3dxrd is basically a collection of function meant to be used with ImageD11. Tutorials are provided as jupyter notebooks in the Tutorial folder


## Documentation


## License


## Credits
Jean-Baptiste Jacob (jbjacob94)


# Contact
j.b.jacob@mn.uio.no
