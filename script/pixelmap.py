import os, sys, copy, h5py, tqdm
import numpy as np, pylab as pl

from matplotlib_scalebar.scalebar import ScaleBar
import scipy.ndimage as ndi

import ImageD11.cImageD11
import ImageD11.columnfile
import ImageD11.finite_strain
import ImageD11.grain
import ImageD11.refinegrains
import ImageD11.stress
import ImageD11.sym_u
import ImageD11.unitcell
import xfab

from orix import data, io, plot as opl, crystal_map as ocm, quaternion as oq, vector as ovec

from pf_3dxrd import utils, crystal_structure, peak_mapping

"""
Pixelmap Class: A class to map and plot data on a 2D pixel grid., allowing per-pixel data analysis and processing. 
"""
    
       
# Pixelmap Class
###########################################################################

class Pixelmap:
    """ A class to store pixel information on a 2d grid """
    
    # Init
    ##########################
    def __init__(self, xbins, ybins, h5name=None):
        # grid + pixel index
        self.grid = self.GRID(xbins, ybins)
        self.xyi = np.asarray([i + 10000*j for j in ybins for i in xbins]).astype(np.int32)
        self.xi = np.array(self.xyi % 10000, dtype=np.int16)
        self.yi = np.array(self.xyi // 10000, dtype=np.int16)
        
        # phase / grain labeling  + crystal structure information
        self.phases = self.PHASES() 
        self.phase_id = np.full(self.xyi.shape, -1, dtype=np.int8)   # map of phase_ids
        self.grain_id = np.full(self.xyi.shape, -1, dtype=np.int16)   # map of grain_ids
        
        # grains
        self.grains = self.GRAINS_DICT()
        
        self.h5name = h5name
    
    def __str__(self):
        return f"Pixelmap:\n size: {self.grid.shape},\n phases: {self.phases.pnames},\n phase_ids: " +\
               f"{self.phases.pids},\n titles: {self.titles()}, \n grains: {len(self.grains.glist)}"
    
    def get(self,attr):
        """ alias for __getattribute__"""
        return self.__getattribute__(attr)
    
    
    # subclasses
    ##########################  
    class GRID:
        """subclass for grid properties"""
        def __init__(self, xbins, ybins):
            self.xbins = xbins
            self.ybins = ybins
            self.shape = (len(xbins),len(ybins))
            self.nx = len(xbins)
            self.ny = len(ybins)
            self.pixel_size = 1
            self.pixel_unit = 'um'
            
            
        def __str__(self):
            return f"grid: size: {self.shape}, pixel size: {str(self.pixel_size)+' '+self.pixel_unit}"
   
        def scalebar(self):
            """ scalebar for plotting maps"""
            scalebar =  ScaleBar(dx = self.pixel_size,
                                     units = self.pixel_unit,
                                     length_fraction=0.2,
                                     location = 'lower left',
                                     box_color = 'w',
                                     box_alpha = 0.5,
                                     color = 'k',
                                     scale_loc='top')
            return scalebar
        
    
    class PHASES:
        """ sub-class to store information on crystal structures."""
        def __init__(self):
            self.notIndexed = crystal_structure.CS(name='notIndexed')
            self.pnames = ['notIndexed']
            self.pids = [-1]
           
        
        def __str__(self):
            return f"phases: {self.pnames}"
        
        
        def get(self,attr):
            return self.__getattribute__(attr)
            
            
        def add_phase(self, pname, cs):
            """ add phase to pixelmap.phases. 
            
            Parameters
            ----------
            pname (str) : phase name
            cs          : crystal_structure.CS object
            """
            # if this phase name already exists, delete it
            if pname in self.pnames:
                print(pname, ': There is already phase with this name in self.phases. Will overwrite it.')
                self.delete_phase(pname)
                
            # write new phase and update pnames and pids lists    
            setattr(self, pname, cs)
            self.pnames.append(pname)
            self.pids.append(cs.phase_id)
            self.sort_phase_lists()
            
            
        def delete_phase(self, pname):
            cs = self.get(pname)
            pid = cs.get('phase_id')
            path = cs.get('cif_path')
            self.pnames = [p for p in self.pnames if p != pname]
            self.pids = [i for i in self.pids if i != pid]
            delattr(self, pname)
            self.sort_phase_lists()
             
                   
        def sort_phase_lists(self):
            """ sort pnames and pids by phase id """
            sorted_pids = [l1 for (l1, l2) in sorted(zip(self.pids, self.pnames), key=lambda x: x[0])]
            sorted_pnames = [l2 for (l1, l2) in sorted(zip(self.pids, self.pnames), key=lambda x: x[0])]
            self.pids = sorted_pids
            self.pnames = sorted_pnames
            
            
            
    class GRAINS_DICT:
        """ sub-class to store grains information. Mainly a wrapper containing a dictionnary of ImageD11.grain.grain objects """
        def __init__(self):
            self.dict = {}
            self.gids = list(self.dict.keys())
            self.glist = list(self.dict.values())
            
            
        def __str__(self):
            return f"nb grains: {len(self.glist)}"
        
        
        def get(self,prop, grain_id):
            """ Return a property for a grain. Shortcut for g.__getattribute__(prop)"""
            g = self.dict[grain_id]
            return g.__getattribute__(prop)
        
            
        def get_all(self, prop, pname=None):
            """ 
            return selected grain property for all grains in grains_dict as an array
            prop (str): property to select in grains , e.g. 'UBI'
            pname : select specific phase. If None, all grains are selected. Default is None 
            """
            if pname is None:
                glist = self.glist
            else:
                glist = self.select_by_phase(pname)
            return np.array( [g.__getattribute__(prop) for g in glist] )
        
        
        def select_by_phase(self, pname):
            """ return all grains corresponding to a given phase. pname : phase name, must be in xmap.phases"""
            gsel = [g for g in self.glist if g.phase == pname]
            return gsel
        
        
        def add_prop(self, prop, grain_id, val):
            """ add new property to a grain.
            prop : name for new property to add
            val  : value of new property"""
            g = self.dict[grain_id]
            setattr(g, prop, val)
        
                     
        def plot_grains_prop(self, prop, s_factor=10, autoscale=True, percentile_cut=[5,95], out=False, **kwargs):
            """ scatter plot of grains colored by property prop, where (x,y) is grain centroid position and s is grainsize. 
            Args:
            ---------
            prop (str): a grain property (e.g. grains size, GOS, etc.). If "strain" or "stress" are entered, it will combine all strain /stress components in a single plot
            s_factor (int): factor to adjust spot size on the scatter plot. size = grainsize / s_factor
            autoscale (bool) : automatically adjust color scale to distribution for each strain / stress component. Default is True
            percentile_cut [low,up]: percentile thresholds to cut distribution and adjust colorbar limits (for stress / strain with autoscale)
            out (bool) : return figure. Default is False"""
            
            try:
                cen = self.get_all('centroid')
                gs = self.get_all('grainsize')
            except:
                print('missing grainSize or centroid position')
                return
            
            if prop not in 'strain,stress'.split(','): 
                assert np.all( [hasattr(g, prop) for g in self.glist] )
                colorsc = self.get_all(prop) # color scale defined by selected property
        
                fig = pl.figure(figsize=(6,6))
                ax = fig.add_subplot(111, aspect='equal')
                ax.set_axis_off()
                sc = ax.scatter(cen[:,0], cen[:,1], s = gs/s_factor, c = colorsc, **kwargs)
                ax.set_title(prop)
                cbar = pl.colorbar(sc, ax=ax, orientation='vertical', pad = 0.05, shrink=0.65)
                cbar.formatter.set_powerlimits((-1, 1)) 
            
            else:
                vals = self.get_all(prop)
                if prop == 'strain':
                    titles = 'e11,e22,e33,e23,e13,e12'.split(',')
                else:
                    titles = 's11,s22,s33,s23,s13,s12'.split(',')
                
                fig, ax = pl.subplots(2,3, figsize=(10,7), sharex=True, sharey=True)
                ax = ax.flatten()
                
                for i, (a,t) in enumerate(zip(ax, titles)):
                    a.set_aspect('equal')
                    a.set_axis_off()
                    x = vals[:,i]
                    low, up = np.percentile(x, (percentile_cut[0],percentile_cut[1]))
    
                    # plots
                    if autoscale:
                        norm=pl.matplotlib.colors.CenteredNorm(vcenter=np.median(x), halfrange=up)
                        sc = a.scatter(cen[:,0], cen[:,1], s = gs/s_factor, c = x, norm=norm, **kwargs)
                    else:
                        sc = a.scatter(cen[:,0], cen[:,1], s = gs/s_factor, c = x, **kwargs)
                    a.set_title(t)
            
                    # colorbar
                    cbar = pl.colorbar(sc, ax=a, orientation='vertical', pad=0.04, shrink=0.65)
                    cbar.formatter.set_powerlimits((-1, 1)) 
            
            # Adjust layout
            fig.tight_layout()
            fig.suptitle('grain scatterplot - '+prop, y=1.0)
                    
            if out:
                return fig
            
            
            
        def hist_grains_prop(self, prop, percentile_cut=[2,98], nbins = 100, out=False, **kwargs):
            """ plot histogram of selected grains property. 
            Args:
            --------
            prop (str): grain property. To plot all strain /stress components at once, type prop='stress' or prop='strain' 
            percentile_cut [low,up]: percentile thresholds to trim distribution and adjust histogram width
            nbins (int) : number of bins
            out (bool): return figure """
            
            if prop not in 'strain,stress'.split(','): 
                assert np.all( [hasattr(g, prop) for g in self.glist] )
                x = self.get_all(prop) 
                low, up = np.percentile(x, (percentile_cut[0],percentile_cut[1]))
                bins = np.linspace(low,up, nbins)
        
                fig = pl.figure(figsize=(6,6))
                ax = fig.add_subplot(111)
                h = ax.hist(x, bins, **kwargs)
                ax.vlines(np.median(x), ymin=0, ymax=h[0].max(), colors='r', label='median')
                ax.set_xlim(low, up)
                ax.set_title(prop)
            
            else:
                vals = self.get_all(prop)
                if prop == 'strain':
                    titles = 'e11,e22,e33,e23,e13,e12'.split(',')
                else:
                    titles = 's11,s22,s33,s23,s13,s12'.split(',')
                
                fig, ax = pl.subplots(2,3, figsize=(10,6))
                ax = ax.flatten()
                for i, (a,t) in enumerate(zip(ax, titles)):
                    x = vals[:,i]
                    low, up = np.percentile(x, (percentile_cut[0],percentile_cut[1]))
                    bins = np.linspace(low,up, nbins)
                    h = a.hist(x, bins, **kwargs)
                    a.vlines(np.median(x), ymin=0, ymax=h[0].max(), colors='r', label='median')
                    a.set_xlim(low,up)
                    a.set_title(t)
                    a.legend(loc='upper left', fontsize=7)
            
            # Adjust layout
            fig.tight_layout()
            fig.suptitle('distribution - '+prop, y=1.0)
            
            if out:
                return fig

                

    # methods
    ######################
    def add_data(self, data, dname):
        """ add a data column to pixelmap.
        preferentially use numpy array or ndarray of shape(nx*ny,n), but lists may work as well"""
        assert len(data) == self.grid.nx * self.grid.ny
        setattr(self, dname, data)
        
        
        
    def rename_data(self, oldname, newname):
        """ rename data column """
        data = self.__getattribute__(oldname)
        setattr(self, newname, data)
        delattr(self, oldname)
        
        
        
    def titles(self):
        return [t for t in self.__dict__.keys() if t not in ['grid', 'phases', 'grains', 'h5name'] ]
        
    
    
    def copy(self):
        """ returns a deep copy of the pixelmap """
        pxmap_new = copy.deepcopy(self)
        return pxmap_new
    
    
    
    def update_pixels(self, xyi_indx, dname, newvals):
        """ update data column dname with new values for a subset of pixel selected by xyi indices, without touching other pixels
        
        Args:
        ---------
        xyi_indx: list of xyi index of pixels to update
        dname: name of the data column to update
        newvals: array of new values. Must have same length as xyi_indx"""
        
        assert dname in self.__dict__.keys()
        
        xyi_indx = np.array(xyi_indx)
        
        # select data column and pixels to update
        dat = self.get(dname)
        dtype = type(dat.flatten()[0])
        pxindx = np.searchsorted(self.xyi, xyi_indx)
        
        # update data
        assert newvals.shape[1:] == dat.shape[1:]  # check array shape compatibility
        if len(dat.shape) == 1:  # dat is simple 1d array
            dat[pxindx] = newvals.astype(dtype)
        else:    # nd array of arbitrary size
            dat[pxindx,:] = newvals.astype(dtype)
            
        setattr(self, dname, dat)
        
        
        
    def update_grains_pxindx(self, mask=None, update_map=False):
        """ update grains pixel masks (pxindx / xyi_indx in grain properties), according to criteriosn defined in mask.
        Allows to remove bad pixels (large misorientation, low npks indexed, high drlv2, etc.) from grain masks. 
        
        Args:
        --------
        mask: bool array of same shape as data columns (grid.nx*grid.ny,) to filter bad pixels
        update_map: if True, grain_id in pixelmap will also be updated. 
        MAKE A COPY OF PIXELMAP FIRST, OR INITIAL GRAIN INDEXING WILL BE LOST Default is False"""
        
        if mask is None:
            mask = np.full(self.xyi.shape, True)
    
        assert mask.shape == self.xyi.shape # make sure mask is the good size
    
        for gi,g in tqdm.tqdm(zip(self.grains.gids, self.grains.glist)):
            gm = np.all([mask, self.grain_id==gi], axis=0)  # select pixels for each grain
            g.pxindx = np.argwhere(gm)[:,0].astype(np.int32)  # reassign pxindx
            g.xyi_indx = self.xyi[g.pxindx].astype(np.int32)    # pixel labeling using XYi indices. needed to select peaks from cf
        # update grain ids
        if update_map:
            self.grain_id[~mask] = -1
        
        
        
    def filter_by_phase(self, pname):
        """ Returns a new pixelmap containing only the selected phase. It will make a deep copy of the pixelmap and reinitialize
        all pixels not corresponding to the selected phase. It will also update h5name in new pixelmap to avoid overwriting the former file
        
        Args    : pname : phase name. must be in self.phases
        Returns : xmap_p: new pixelmap with only the selected phase """
        
        # make a copy of pixelmap
        xmap_p = self.copy()
        xmap_p.h5name = self.h5name.replace('.h5','_'+pname+'.h5')
        # select phase
        phase = xmap_p.phases.get(pname)
        pid = phase.phase_id
        
        # update columns
        for dname in self.__dict__.keys():
            if dname in ['grid', 'xyi', 'xi', 'yi', 'phases', 'h5name', 'grains']:
                continue
            
            msk = self.phase_id == pid
            array = self.get(dname)
            
            if 'strain' in dname or 'stress' in dname:
                new_array = np.full(array.shape, float('inf'))
            elif dname == 'phase_id' or dname == 'grain_id':
                new_array = np.full(array.shape, -1, dtype=int)
            else:
                new_array = np.zeros_like(array)
                
            new_array[msk] = array[msk]
            xmap_p.add_data(new_array, dname)
        
        # update phases
        for p in xmap_p.phases.pnames:
            if p == 'notIndexed' or p == pname:
                continue
            xmap_p.phases.delete_phase(p)
        
        return xmap_p
    
    
    
    def add_grains_from_map(self, pname, overwrite=False):
        """ Add grains to self.grains from the indexed pixelmap. 2D grain masks must be present (grain_id column), and the pixelmap must have been indexed, ie a UBI column is present as well. For each grain, the UBI value is computed as the median of all UBIs over the grain mask

        Args:
        ----------
        pname: name of phase to select. must be in self.phases
        overwrite: re-initialize grains dict. Default is False
        
        
        NB: grains ubis at this stage are 'median' ubis obtained by averaging each component of the UBI matrix individually.
        This is quite dodgy, so you might want to refine ubis after this step. For this, you need to map the peaks from the original peakfile
        used for indexing to the grains in xmap, and then refine ubis using these peaks. These two steps are done using the methods 
        "map_pks_to_grains" and "refine_ubis" """
        
        assert 'UBI' in self.__dict__.keys()
              
        # crystal structure
        cs = self.phases.get(pname)
        pid = cs.phase_id
        sym = cs.orix_phase.point_group.laue
        # phase + UBI masks. pm: also select notindexed pixels, because some may be included to grain masks by smoothing (e.g if grain_id have been computed from mtex and involved smoothing of grain boundaries)
        pm = np.any([self.phase_id == pid, self.phase_id == -1], axis=0)  
        isUBI = np.asarray( [np.trace(ubi) != 0 for ubi in self.UBI] )   # mask: True if UBI at a given pixel position is not null
      
        # list of unique grain ids for the selected phase
        gid_u = np.unique(self.grain_id[pm]).astype(np.int16)  
        
        # if overwrite, re-initialize grains dict. Otherwise, keep existing grains in grains dict and append new ones
        if overwrite:
            self.grains.__init__()
        
        # loop through unique grain ids, select pixels, compute mean ubi and create grain
        ########################################
        for i in tqdm.tqdm(gid_u):
            # skip notindexed domains
            if i == -1:
                continue  
            # compute median grain properties and create grain.
            gm = self.grain_id==i
            try:   
                ubi_g = np.nanmedian(self.UBI[pm*gm*isUBI], axis=0)
                g = ImageD11.grain.grain(ubi_g)  
            except:
                print('could not create grain for id', i)
                continue
        
            # grain to xmap mapping
            g.gid = i
            g.phase = pname
            g.pxindx = np.argwhere(gm*isUBI)[:,0].astype(np.int32)  # pixel indices in grainmap matching with this grain
            g.grainsize = len(g.pxindx)
            g.surf = g.grainsize * self.grid.pixel_size**2  # grain surface in pixel_unit square
            g.xyi_indx = self.xyi[g.pxindx]    # pixel labeling using XYi indices. needed to select peaks from cf
        
            # phase properties + misorientation
            try:
                og = oq.Orientation.from_matrix(g.U, symmetry = sym)
                opx = oq.Orientation.from_matrix(self.U[gm*isUBI], symmetry=sym)
                misOrientation = og.angle_with(opx, degrees=True)
                g.GOS = np.median(misOrientation)  # grain orientation spread defined as median misorientation over the grain
            except:
                f
                
            # grain centroid
            cx = np.average(self.xi[g.pxindx], weights = self.nindx[g.pxindx])
            cy = np.average(self.yi[g.pxindx], weights = self.nindx[g.pxindx])
            g.centroid = np.array([cx,cy])
                
            # add grain to grains dict
            self.grains.glist.append(g)
            self.grains.gids.append(g.gid)
        
        # update grains dict
        self.grains.dict = dict(zip(self.grains.gids, self.grains.glist))
        
        
        
    def map_pks_to_grains(self, pname, cf, overwrite=False):
        """ map peaks from cf to grains in pixelmap for grains in self.grains.glist. 
        updates cf.grain_id column in peakfile and g.pksindx prop for each grain in self.grain.glist        
        
        Args:
        ---------
        pname : phase name to select
        cf    : peakfile which has been used for indexing. 
        overwrite : if True, reset 'grain_id' column in cf. default if False
        """
        glist = self.grains.select_by_phase(pname)        
        print('peaks to grains mapping...')
        peak_mapping.map_grains_to_cf(glist, cf, overwrite=overwrite)
        self.grains.dict = dict(zip(self.grains.gids, self.grains.glist))
    
    
    
    def refine_ubis(self, pname, cf, hkl_tol, nmedian, sym):  
        """  Refine peaks_to_grain assignement and fit ubis for all grains from the  selected phase. 
        returns statistics about fraction of peaks retained for each grain and rotation between old and new orientations
        
        Args:
        ---------
        pname : phase name to select
        cf    : peakfile used for indexing
        hkl_tol : tolerance to pass to score_and_refine
        nmedian : threshold to remove outliers ( abs(median err) > nmedian ). Use np.inf to keep all peaks
        sym : crystal symmetry, used to compute angular shift between new and old orientation
        Output: prop of peaks retained, angle deviation (deg) between old and new grain orientation
        """
        glist = self.grains.select_by_phase(pname)
        print('refining ubis...')
        prop_indx, ang_dev = peak_mapping.refine_grains(glist, cf, hkl_tol = hkl_tol, nmedian = nmedian, sym = sym, return_stats=True)
        self.grains.dict = dict(zip(self.grains.gids, self.grains.glist))
        return prop_indx, ang_dev
    
    
    
    ##WORK IN PROGRESS
    def refine_ubis_px(self, pname, cf, hkl_tol, nmedian, sym):
        """ Refine UBIs for each pixel. Work in progres..."""
        pid = self.phases.get(pname).phase_id
        sel = self.phase_id == pid
        
        # create grains from pixel ubis and add peak mapping to these grains
        glist = [ImageD11.grain.grain(ubi) for ubi in self.UBI[sel]]
        
        indpos = np.searchsorted( cf.xyi, np.append(self.xyi, self.xyi.max()+1) )
        inds = [ np.arange(indpos[i], indpos[i+1]) for i in range(len(self.xyi)) ]
        inds_sel = [ind for i,ind in enumerate(inds) if sel[i]]
        
        for i,g in zip(inds_sel,glist):
            g.pksindx = i
            g.phase = pname
               
        # refine ubis and update xmap.ubi data
        print('refining ubis...')
        prop_indx, ang_dev = peak_mapping.refine_grains(glist, cf, hkl_tol = hkl_tol, nmedian = nmedian, sym = sym, return_stats=True)
        
        # update grain properties (ubi, U, unitcell) in xmap
        self.update_pixels(self.xyi[sel], 'UBI', np.array([g.ubi for g in glist]))
        self.update_pixels(self.xyi[sel], 'U', np.array([g.U for g in glist]))
        self.update_pixels(self.xyi[sel], 'unitcell', np.array([g.unitcell for g in glist]))
        self.update_pixels(self.xyi[sel], 'nindx', np.array([p*n for p,n in zip(prop_indx,self.nindx[sel])]) )
        
        return prop_indx, ang_dev
              
        

    
    
    def map_grain_prop(self, prop, pname, debug=0):
        """ map a grain property (U, UBI, unitcell, grainsize, etc.) taken from grains in grains.dict to the 2D grid.
        For a grain property p, this function creates a new data column 'p_g' in pixelmap to map this property for each grain on
        the 2D grid using grain masks.
        
        To quickly map all six strain / stress components, simply type 'stress" or 'strain' as a prop and the function will
        look for all tensor components and return a single output as a ndarray.
        
        If grain orientation (U matrix) is selected and pixel orientations are available in pixelmap, 
        it will also compute misorientation (angle between grain and pixel orientation in degree)
        
        Args:
        ------
        prop : grain property to map
        pname : phase name to select
        """
        
        # Initialize new array
        #####################################
        array_shape = [ self.grid.nx * self.grid.ny ]  # size of pixelmap
        compute_misO = False  # flag to compute misOrientation
        
        if any([s in prop for s in 'stress,strain,eps,sigma'.split(',')]):   # special case for stress / strain related data
            prop_name = prop+'_g'
            prop_shape = list( self.grains.get_all(prop, pname).shape[1:] )
            array_shape.extend(prop_shape)
            newarray = np.full(array_shape, np.inf)  # default value = inf to avoid confonding zero strain / stress with no data
        else:
            prop_shape = list( self.grains.get_all(prop, pname).shape[1:] )
            prop_name = str(prop)+'_g'  # add g suffix to make it clear it is derived from a grain property
            array_shape.extend(prop_shape)
        
            # special values to initialize grain/phase id: -1. Default: 0
            if any([s in prop for s in 'gid,grain_id,phase_id'.split(',')]):
                init_val = -1
            elif any([s in prop for s in 'I1,J2'.split(',')]):
                init_val = np.inf
            else:
                init_val = 0
            
            # dtype: float (default) or int 
            if isinstance(self.grains.get_all(prop)[0], 'int'):
                dtype = 'int'
            else:
                dtype = 'float'
            newarray = np.full(array_shape, init_val, dtype = dtype)
        
        #if grain orientation U is selected, try to compute misorientation as well
        if prop == 'U':
            try:
                U_px = self.get('U')
                misO = np.full(self.xyi.shape, 180, dtype=float)   # default misorientation to 180°
                # crystal structure
                cs = self.phases.get(pname)
                sym = cs.orix_phase.point_group.laue
                compute_misO = True
            except:
                print('No orientation data in pixelmap, or name not recognized (must be "U"). Will not compute misorientation.')
                
        # update with values from grains in graindict
        #####################################
        gid_map = self.get('grain_id')
        
        for gi,g in tqdm.tqdm(zip(self.grains.gids, self.grains.glist)):
            gm = np.argwhere(gid_map == gi).T[0]  # grain mask
            
            # fill newarray. Different cases depending of prop shape
            if len(prop_shape) == 0:
                newarray[gm] = self.grains.get(prop,gi)
            else:
                newarray[gm,:] = self.grains.get(prop,gi)
                
            # misorientation
            if prop=='U' and compute_misO:
                og = oq.Orientation.from_matrix(newarray[gm], symmetry=sym)
                opx =  oq.Orientation.from_matrix(self.get('U')[gm], symmetry=sym)
                misO[gm] = og.angle_with(opx, degrees=True)
        
        # add newarray to pixelmap
        self.add_data(newarray, prop_name)
        if compute_misO:
            self.add_data(misO, 'misorientation')

                                                 

    
    def plot(self, dname, save=False, hide_cbar=False, autoscale=True, percentile_cut = [2,98],
             smooth=False, mf_size=1, out=False, **kwargs):
        """ Make a pcolormesh plot for a selecte ddata column
        
        Args:
        --------
        dname (str) : name of data array. data in self.dname must be a single array (shape (N,1))
        save (bool) : save plot (default is False)
        hide_cbar (bool) : hide colorbar from plot (delault is False)
        smooth (bool) : apply median filter for smoothing plot
        mf_size (int) : size of median filter kernel for smoothing. default is 1 
        out (bool) : return figure as output (default is False)
        autoscale (bool): automatically adjust color scale to distribution for each strain / steess component (default is True)
        percentile_cut ([low,up]): percentile thresholds to cut distribution (for autoscale)
        kwargs (dict) : keyword arguments passed to matplotlib"""
        
        nx, ny = self.grid.nx, self.grid.ny
        xb, yb = self.grid.xbins, self.grid.ybins
        dat = self.get(dname).reshape(nx, ny)
        
        if smooth:
            dat = ndi.median_filter(dat, size=mf_size)
        
        fig = pl.figure(figsize=(6,6))
        ax = fig.add_subplot(111, aspect ='equal')
        ax.set_axis_off()
        
        if autoscale:
            m = np.all([dat!=0, dat!=-1, dat!=float('inf')], axis=0)
            dat_u = np.unique(dat[m])
            low, up = np.percentile(dat_u, (percentile_cut[0],percentile_cut[1]))
            im = ax.pcolormesh(xb, yb, dat, vmin=low, vmax=up, **kwargs)
        else:
            im = ax.pcolormesh(xb, yb, dat, **kwargs)
        ax.set_title(dname)
        ax.add_artist(self.grid.scalebar())
        
        if not hide_cbar:
            fig.suptitle(self.h5name.split('/')[-1].split('.h')[0], y=.9)
            if 'phase_id' in dname:
                cbar = pl.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7, ticks = self.phases.pids)
                cbar.ax.set_yticklabels(self.phases.pnames)
            else:
                cbar = pl.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7, label=dname)
                cbar.formatter.set_powerlimits((-1, 1)) 
        if hide_cbar:
            fig.suptitle(self.h5name.split('/')[-1].split('.h')[0], y=1.)
        
        if save:
            fname = self.h5name.replace('.h5', '_'+dname+'.png')
            fig.savefig(fname, format='png', dpi = 300) 
        if out:
            return fig
            
            
            
    def plot_voigt_tensor(self, dname, autoscale=True, percentile_cut = [2,98],
                          save=False, hide_cbar=False, smooth=False, mf_size=1, out=False, **kwargs):
        """ plot all components of strain / stress tensor (voigt notation) in a single figure
        
        Args: 
        ---------
        dname (str) : name of data array. data in self.dname must be a Nx6 array with strain / stress components in the following order:
        e11,e22,e33,e23,e13,e12
        autoscale (bool): automatically adjust color scale to distribution for each strain / steess component (default is True)
        percentile_cut ([low,up]): percentile thresholds to cut distribution (for autoscale)
        save (bool) : save plot (default is False)
        hide_cbar (bool) : hide colorbar from plot (delault is False)
        smooth (bool) : apply median filter for smoothing plot
        mf_size (int) : size of median filter kernel for smoothing. default is 1 
        out (bool) : return figure as output (default is False)
        kwargs (dict) : keyword arguments passed to matplotlib""" 
        
        nx, ny = self.grid.nx, self.grid.ny 
        xb, yb = self.grid.xbins, self.grid.ybins
        voigt_tensor = self.get(dname)
        
        # figures layout
        fig, ax = pl.subplots(2,3, figsize=(10,7), sharex=True, sharey=True)
        ax = ax.flatten()
        
        if any(['strain' in dname, 'eps' in dname]):
            titles = 'e11,e22,e33,e23,e13,e12'.split(',')
        elif any(['stress' in dname, 'sigma' in  dname]):
            titles = 's11,s22,s33,s23,s13,s12'.split(',')
        else:
            print('data name not recognized. Should contain either "strain"/"eps" or "stress"/"sigma"')
            return
                  
        # loop through strain / stress components and plot them in map
        for i, (a,t) in enumerate(zip(ax, titles)):
            a.set_aspect('equal')
            a.set_axis_off()
            x = voigt_tensor[:,i].reshape(nx,ny)
            if smooth:
                x = ndi.median_filter(x, size=mf_size)
            x_u = np.unique(voigt_tensor[voigt_tensor != float('inf')])  # select unique values to get distribution across grains  
    
            # plots
            if autoscale:
                low, up = np.percentile(x_u, (percentile_cut[0],percentile_cut[1]))
                norm=pl.matplotlib.colors.CenteredNorm(vcenter=np.median(x_u), halfrange=up)
                im = a.pcolormesh(xb, yb, x, norm=norm, **kwargs)
            else:
                im = a.pcolormesh(xb, yb, x, **kwargs)
            a.set_title(t)
            
            # colorbar
            if not hide_cbar:
                cbar = pl.colorbar(im, ax=a, orientation='vertical', pad=0.04, shrink=0.7)
                cbar.formatter.set_powerlimits((-1, 1)) 
            
        # Adjust layout
        fig.tight_layout()
        dsname = self.h5name.split('/')[-1].split('.h')[0]
        
        fig.suptitle('grainmap '+dname+' - '+dsname, y=1.0)

        if save:
            fname = self.h5name.replace('.h5', '_'+dname+'.png')
            fig.savefig(fname, format='png', dpi=300)
            
        if out:
            return fig
            
        
        
    def hist_voigt_tensor(self, dname, percentile_cut=[2,98], nbins=100, save=False, out=False, **kwargs):
        """ plot histogram for all components of strain / stress tensor (voigt notation)
        Args: 
        --------
        dname (str) : name of data array. data in self.dname must be a Nx6 array with strain / stress components in the following order:
        e11, e22, e33, e23, e13, e12
        percentile_cut ([low,up]): percentile thresholds to cut distribution and adjust histogram width
        save (bool) : save plot (default is False)
        nbins (int): number of bins in the histogram. Default is 100
        out (bool) : return figure as output (default is False)
        kwargs (dict) : keyword arguments """ 
        
        voigt_tensor = self.get(dname)
        # figures layout
        fig, ax = pl.subplots(2,3, figsize=(10,6))
        ax = ax.flatten()
        
        if any(['strain' in dname, 'eps' in dname]):
            titles = 'e11,e22,e33,e23,e13,e12'.split(',')
        elif any(['stress' in dname, 'sigma' in  dname]):
            titles = 's11,s22,s33,s23,s13,s12'.split(',')
        else:
            print('data name not recognized. Should contain either "strain"/"eps" or "stress"/"sigma"')
            return
          
        for i, (a,t) in enumerate(zip(ax, titles)):
            x = voigt_tensor[:,i]
            x_c = x[x != float('inf')]
            low, up = np.percentile(x_c, (percentile_cut[0],percentile_cut[1]))
            bins = np.linspace(low, up, nbins)
            h = a.hist(x_c, label=lab, bins=bins, **kwargs)
            a.vlines(x=np.median(x_c), ymin=0, ymax=h[0].max(), colors='r', label='median')
            a.set_title(t)
            a.set_xlim(low,up)
            a.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            a.legend(loc='upper left', fontsize=7)
                
        fig.tight_layout()
        dsname = self.h5name.split('/')[-1].split('.h')[0]
        fig.suptitle('hist '+dname+' - '+dsname, y=1.0)
            
        if save:
            fname = self.h5name.replace('.h5', '_'+dname+'_hist.pdf')
            fig.savefig(fname, format='pdf')
        if out:
            return fig
            
    
    
    
    def plot_ipf_map(self, phase, dname='U',ipfdir = [0,0,1], ellipsoid = False, save=False, hide_cbar=False, out=False, **kwargs):
        """ Plot inverse pole figure color map of orientation 
        
        Args:
        --------
        phase (str): name of the phase to plot. must be in self.phases
        dname (str) : name of orientation data array. Default is 'U'. Must be a ndarray with shape (N,3)
        ipfdir (array): direction for the ipf colorkey in the laboratory reference frame. must be a 3x1 vector [x,y,z]. Default: z-vector [0,0,1]
        ellipsoid (bool) : set symmetry to one of a triaxial ellipsoid (orthorombic, mmm). For ipf color map of strain stress principal components orientation
        save (bool) : save plot (default is False)
        hide_cbar (bool) : hide colorbar from plot (delault is False)
        out (bool) : return figure as output (default is False)
        kwargs (dict) : keyword arguments passed to matplotlib"""
    
        assert phase in self.phases.pnames
        
        #map grid
        nx, ny = self.grid.nx, self.grid.ny
        xb, yb = self.grid.xbins, self.grid.ybins
    
        # phase symmetry + color key
        cs =  self.phases.get(phase)
        if ellipsoid:
            sym = oq.symmetry.D2h
            ipf_key = opl.IPFColorKeyTSL(sym, direction=ovec.Vector3d(ipfdir))
        else:
            sym = cs.orix_phase.point_group.laue
            cs.get_ipfkey(direction = ovec.Vector3d(ipfdir))
            ipf_key = cs.ipfkey
            
        # convert orientation data to color map
        U = self.get(dname)
        ori = oq.Orientation.from_matrix(U, symmetry=sym)
        rgb = ipf_key.orientation2color(ori)
        m  = self.phase_id == cs.phase_id
        indx = self.nindx > 0
        rgb[~m,:] = 0 # black pixels for phases other than the one selected
        rgb[~indx,:] = 0  # keep unindexed pixels black

        # plot orientation map 
        fig = pl.figure(figsize=(8,8))
        ax = fig.add_subplot(111, aspect ='equal')
        ax.set_axis_off()
        
        im = ax.pcolormesh(xb, yb, rgb.reshape(nx,ny,3), **kwargs)
        ax.set_title(f'{phase} - ipf map {str(ipfdir)}')
        ax.add_artist(self.grid.scalebar())
    
        # plot color key
        pl.matplotlib.rcParams.update({'font.size': 4})
        fig.subplots_adjust(right=0.7)
        ax1 = fig.add_axes([0.8, 0.25, 0.15, 0.15], projection='ipf',  symmetry=sym)
        ax1.plot_ipf_color_key(show_title=False)
        pl.matplotlib.rcParams.update({'font.size': 10})
            
        
        fig.suptitle(self.h5name.split('/')[-1].split('.h')[0], y=.9)    
    
        if save:
            ipfd_str = ''.join(map(str, ipfdir))
            fname = self.h5name.replace('.h5', f'_{phase}_ipf_{ipfd_str}.png')
            fig.savefig(fname, format='png', dpi = 300) 
        if out:
            return fig
    
    
            
    def save_to_hdf5(self, h5name=None, save_mode='minimal', debug=0):
        """ 
        save pixelmap to hdf5 format
        
        Args:
        --------
        h5name : hdf5 file name. If None, name in self.h5name will be used. default is None
        save_mode : minimal / full. If minimal, drops all columns computed from grains and columns related to strain and stress
        
        """
        # save path
        if h5name is None:
            try:
                h5name = self.h5name
                h5name[0]
            except:
                print("please enter a path for the h5 file")
        
        with h5py.File(h5name, 'w') as f:
            
            f.attrs['h5path'] = h5name
            
            # Save grid in 'grid' group
            grid_group = f.create_group('grid')
            
            attr = 'pixel_size', 'pixel_unit'
            for item in self.grid.__dict__.keys():
                if item in attr:
                    grid_group.attrs[item] = self.grid.__getattribute__(item)
                else:
                    data = self.grid.__getattribute__(item)
                    grid_group.create_dataset(item, data = data, dtype = int) 
            
            # Save phases in 'phases' group
            phases_group = f.create_group('phases')
            for pname, pid in zip(self.phases.pnames, self.phases.pids):
                # create a new group for each phase
                phase = phases_group.create_group(pname)
                cs = self.phases.get(pname)
                phase.attrs.update({ 'pid':pid})
                phase.attrs.update({'cif_path':str(cs.cif_path)})
                try:
                    phase.create_dataset('cif_file', data=cs.cif_file, dtype=h5py.string_dtype())
                except:
                    print('error in saving cif file for phase', pname)
                
            # save grains
            save_grains_dict(self.grains.dict, h5name)

            # Save other data
            skip = ['grid', 'xi', 'yi', 'phases', 'pksind', 'h5name', 'grains']  # things to skip
            
            for item in self.__dict__.keys():
                if item in skip:
                    continue
                if save_mode == 'minimal' and any([s in item for s in ['_g','strain','stress','eps','sigma']]):
                    continue
                data = self.__getattribute__(item)
                if debug:
                    print(item) 
                f.create_dataset(item, data = data, dtype = type(data.flatten()[0]))

        print("Pixelmap saved to:", h5name)
        


# Save / load functions
##########################    
        
    
def load_from_hdf5(h5name, debug=0):
    """ load pixelmap for hdf5 file"""
    with h5py.File(h5name, 'r') as f:
        # Load grid information 
        xbins  = f['grid/xbins'][()]
        ybins  = f['grid/xbins'][()]
        pxsize = f['grid'].attrs['pixel_size']
        pxunit = f['grid'].attrs['pixel_unit']

        # Load phases information : names ids, cif_path, cif_file (stored as list of strings)
        pnames, pids, cif_paths, cif_files = [], [], [], []
        for pname, phase in f['phases'].items():
            pid = phase.attrs['pid']
            cif_path = phase.attrs['cif_path']
            
            try:    
                cif_file_bytes = phase['cif_file'][()]
                cif_file = [c.decode('utf-8') for c in cif_file_bytes]   # strings encoded, need to be decoded
            except:
                cif_file = '_'
                
            pnames.append(pname)
            pids.append(pid)
            cif_paths.append(cif_path)
            cif_files.append(cif_file)
        
        if debug:
            print('cif paths', cif_paths,'\n','files',cif_files)
            
        # Load grains
        if 'grains' in list(f.keys()):
            grainsdict = load_grains_dict(h5name)
        else:
            grainsdict = {}
    
        # Load other data
        skip = ['grid', 'phases', 'grains']
        data = {}
        for item in f.keys():
            if item in skip:
                continue
            data[item] = f[item][()]

            
    # Create a new Pixelmap object 
    pixelmap = Pixelmap(xbins, ybins, h5name=h5name)
    
    # update grid
    pixelmap.grid.pixel_size = pxsize
    pixelmap.grid.pixel_unit = pxunit
    
    # Add phases to Pixelmap
    for pname, pid, cpath, cfile in zip(pnames, pids, cif_paths, cif_files):
        if pname == "notIndexed":
            continue
        
        # if cif_path is valid, load crystal structure from there; otherwise, try to load it from saved cif file
        if cif_path.endswith('cif'):
            try:
                cs = crystal_structure.CS(pname,pid,cpath)
            except:
                crystal_structure.list_to_cif(cfile, 'tmp')
                cs = crystal_structure.CS(pname,pid,'tmp')
                
        pixelmap.phases.add_phase(pname, cs)
                
    # Add data
    for d in data.keys():
        pixelmap.add_data(data[d], d)
    # Add grainsdict
    pixelmap.grains.dict = grainsdict
    pixelmap.grains.gids = list(grainsdict.keys())
    pixelmap.grains.glist = list(grainsdict.values())
    
    # remove tmp files possibly created when loaing phases
    if os.path.exists('tmp') and not os.path.isdir('tmp'):
            os.remove('tmp')

    return pixelmap




def save_grains_dict(grainsdict, h5name, debug=0):
    """ save grain dictionnary to hdf5. Append data to existing hdf5 file"""
    
    with h5py.File( h5name, 'a') as hout:
        # Delete the existing 'grains' group if it already exists
        if 'grains' in hout:
            del hout['grains']
            
        grains_grp = hout.create_group('grains')

        for i,g in grainsdict.items():
            gr = grains_grp.create_group(str(i))    
            gprops = [p for p in list(g.__dict__.keys()) if not p.startswith('_')]  # list grain properties, skip _U, _UB etc. (dependent)
            
            if debug:
                print(gprops)
            
            for p in gprops:
                attr = g.__getattribute__(p)
                if attr is None:   # skip empty attributes
                    continue
                # find data type + shape
                if np.issubdtype(type(attr), np.integer):
                    dtype = 'int'
                    shape = None
                elif np.issubdtype(type(attr), np.floating):
                    dtype = 'float'
                    shape = None
                elif isinstance(attr, str):
                    dtype = str
                    shape = None
                else:
                    attr = np.array(attr)
                    shape = attr.shape
                    try:
                        dtype = type(attr.flatten()[0])
                    except:    # occurs if attr is empty
                        dtype = float
                
                if debug:
                    print(p,dtype)
                # save arrays as datasets and othr data as attributes
                if shape is not None: 
                    gr.create_dataset(p, data = attr, dtype = dtype) 
                else:
                    gr.attrs.update({ p : attr})


def load_grains_dict(h5name):
    grainsdict = {}
    with h5py.File(h5name,'r') as f:
        if 'grains' in list(f.keys()):
            grains = f['grains']
        else:
            grains = f
            
        gids = list(grains.keys())
        gids.sort(key = lambda i: int(i))
        
        # loop through grain ids and load data
        for gi in gids:
            gr = grains[gi]
            # create grain from ubi
            g = ImageD11.grain.grain(gr['ubi'])
            # load other properties
            for prop, vals in gr.items():
                if prop == 'ubi':
                    continue
                ary = vals[()]
                setattr(g, prop, ary)
            # add grain attributes
            for attr, val_attr in gr.attrs.items():
                setattr(g, attr, val_attr)
                        
            # add grain to grainsdict
            grainsdict[int(gi)] = g
            
    return grainsdict
        
