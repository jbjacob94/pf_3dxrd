import os, sys, h5py, tqdm
import numpy as np, pylab as pl, math as m

import fast_histogram
import skimage.transform
import scipy.spatial, scipy.signal
from scipy.stats import gaussian_kde

from ImageD11 import unitcell, blobcorrector, columnfile, transform, cImageD11, refinegrains, parameters
from ImageD11.blobcorrector import eiger_spatial


""" general functions, mostly to work on ImageD11 columnfiles."""

# columnfile conversion: to/from dict export to hdf5, etc.. Mostly customized functions derived from ImageD11.columnfile.py
########################################################################################
def colf_to_dict(cf):
    "converts ImageD11 columnfile to python dictionnary"""
    keys = cf.titles
    cols = [cf[k] for k in keys]
    cf_dict = dict(zip(keys, cols))
    return cf_dict

def colf_from_dict(pks, parfile=None):
    """Convert a dictionary of numpy arrays to columnfile """
    titles = list(pks.keys())
    nrows = len(pks[titles[0]])
    for t in titles:
        assert len(pks[t]) == nrows, t
    colf = columnfile.newcolumnfile( titles=titles )
    colf.nrows = nrows
    colf.set_bigarray( [ pks[t] for t in titles ] )
    
    if parfile is not None:
        colf.parameters.loadparameters(parfile)
    return colf

def colf_to_hdf( colfile, hdffile, save_mode='minimal', name=None, compression='lzf', compression_opts=None):
    
        """
        Saves columnfile to hdf5. Modified from ImageD11.columnfile.colfile_to_hdf
        
        Args:
        ---------
        colfile : columnfile
        hdffile : hdf5 file path name
        save mode : minimal / full. 
        Minimal mode saves only necessary information that cannot be computed with updateGeometry() (default), while full mode saves all columns 
        """  
        # LIST OF COLUMNS TO SAVE AS INTEGERS. UPDATED FROM IMAGED11
        INTS = [
        "Number_of_pixels",
        "IMax_f",
        "IMax_s",
        "Min_f",
        "Max_f",
        "Min_s",
        "Max_s",
        "spot3d_id",
        "fp_id",
        "phase_id",
        "h", "k", "l",
        "onfirst", "onlast", "labels",
        "labels",
        "Grain",
        "grainno",
        "grain_id",
        "IKEY",
        "npk2d"
        ]
        
        
        if isinstance(colfile, columnfile.columnfile):
            c = colfile
        else:
            c = columnfile.columnfile( colfile )

        h = h5py.File( hdffile , 'w') # will overwrite if exists
        opened = True
        
        if name is None:
            # Take the file name
            try:
                name = os.path.split(c.filename)[-1]
            except:
                name = 'peaks'
        if name in list(h.keys()):
            g = h[name]
        else:
            g = h.create_group( name )
        g.attrs['ImageD11_type'] = 'peaks'
        
        # col to exclude in "minimal" saving mode (can be recomputed with info from other columns, but takes a bit longer) 
        exclude = ['xl', 'yl', 'zl', 'tth', 'tthc', 'eta', 'gx', 'gy', 'gz', 'ds', 'dsc', 'xs', 'ys', 'r_dist', 'xi', 'yi']
        if save_mode == 'minimal':
            cols = [col for col in c.titles if col not in exclude]
        else:
            cols = c.titles
        
        for t in cols:
            if t in INTS:
                ty = np.int32
            else:
                ty = np.float32
            # print "adding",t,ty
            dat = getattr(c, t).astype( ty )
            if t in list(g.keys()):
                if g[t].shape != dat.shape:
                    g[t].resize( dat.shape )
                g[t][:] = dat
            else:
                g.create_dataset( t, data = dat,
                                  compression=compression,
                                  compression_opts=compression_opts )
        if opened:
            h.close()
            



# Geometric corrections (using detector info), updates on columnfile parameters, load structure data
########################################################################################
def correct_distorsion_eiger( cf, parfile,
              dxfile="/data/id11/nanoscope/Eiger/spatial_20210415_JW/e2dx.edf",
              dyfile="/data/id11/nanoscope/Eiger/spatial_20210415_JW/e2dy.edf"):
    
    """ 
    FOR EIGER DATA. Apply detector distortion correction for eiger data, using pre-computed disortion files. 
    Adds on the geometric computations (tth, eta, gvector, etc.)
    
    Args: 
    ---------
    cf : ImageD11 columnfile
    parfile (str) : parameter file
    dxfile, dyfile : detector distortion. Default files are valid for the eiger detector on the nanofocus station at ID11 
    """
    
    spat = blobcorrector.eiger_spatial( dxfile = dxfile, dyfile = dyfile )
    cf = columnfile.colfile_from_dict( spat( {t:cf.getcolumn(t) for t in cf.titles} ) )
    cf.parameters.loadparameters(parfile)
    cf.updateGeometry()
    
    return cf



def correct_disorsion_frelon( cf, parfile, splinefile, detector_dim = [2048,2048]):
    """ 
    FOR FRELON DATA. Apply detector distortion correction using a pixel look up table computed from a splinefile. 
    Adds on the geometric computations (tth, eta, gvector, etc.)
    
    Args: 
    ---------
    cf : ImageD11 columnfile
    parfile (str) : parameter file
    splinefile    : detector distortion splinefile
    detector_dim  : (X,Y) detector dimensions  
    """
    
    spat = blobcorrector.correctorclass(splinefile)
    
    # make pixel_lut  + substract xy grid coordinate (i,j) to keep only dx and dy arrays.
    spat.make_pixel_lut((detector_dim[0], detector_dim[1]))
    i, j = np.mgrid[ 0:detector_dim[0], 0:detector_dim[1] ]
    dx = spat.pixel_lut[0] - i
    dy = spat.pixel_lut[1] - j
    
    # get integer pixel index (si,fi) of each peak
    si = np.round(cf['s_raw']).astype(int)
    fi = np.round(cf['f_raw']).astype(int)
    
    # apply dx dy correction on s_raw / f_raw
    sc = (dx[ si, fi ] + cf.s_raw).astype(np.float32)
    fc = (dy[ si, fi ] + cf.f_raw).astype(np.float32)
    
    # add corrected arrays as new columns
    cf.addcolumn(sc, 'sc')
    cf.addcolumn(fc, 'fc')
    
    # load parameters and update geometry
    cf.parameters.loadparameters(parfile)
    cf.updateGeometry()
    
    return cf



def fix_flt( cf, splinefile, parfile ):
    """ 
    spline correction for ImageD11 columnfile with stadard method. Slow...
    
    Args:
    --------
    cf : columnfile
    splinefile : detector splinefile
    parfile : imageD11 parameter file
    """
    spat = blobcorrector.correctorclass(splinefile)
    
    if any(['s_raw' in cf.titles, 'f_raw' in cf.titles]):
        cf.addcolumn( cf.s_raw.copy(), 'sc' )
        cf.addcolumn( cf.f_raw.copy(), 'fc' )
    
    for i in tqdm.tqdm(range( cf.nrows )):
        cf.sc[i], cf.fc[i] = spat.correct( cf.s_raw[i], cf.f_raw[i] )
    cf.parameters.loadparameters(parfile)
    cf.updateGeometry()




def get_uc(cf):
    """ computes unitcell and hkl rings using parameters in cf.parameters """ 
    wl = cf.parameters.get('wavelength')
    spg = cf.parameters.get('cell_sg')

    # compute unit cell
    uc = unitcell.unitcell_from_parameters(cf.parameters)
    uc.makerings(cf.ds.max())
    
    ds = uc.ringds
    hkls = uc.ringhkls
    ds = np.unique(ds)

    tth_calc = [np.arcsin( wl*d/2 )*360/np.pi for d in ds]
    
    return uc, ds, hkls, tth_calc, wl


def gethkl(cell,spg, sym, wl, dsmax=1.):
    """ return unique ds + hkl rings for a given space group + wavelength """
    u = unitcell.unitcell(cell,sym)
    hkls = u.gethkls_xfab(dsmax, spg) 
    d = [hkls[i][0] for i in range(len(hkls))]
    d = np.unique(d)
    return d, hkls


def update_colf_cell(cf, cell, spg, lattice_type, mute=False):
    """ update cf.parameters with new cell parameters and crystal symmetry (a, b, c, alpha, beta, gamma, sg, lattice) """
    uc = cell
    pars = [uc[0], uc[1], uc[2], uc[3], uc[4], uc[5], spg, lattice_type]
    parnames = 'cell__a', 'cell__b', 'cell__c', 'cell_alpha', 'cell_beta', 'cell_gamma', 'cell_sg', 'cell_lattice_[P,A,B,C,I,F,R]'

    for p, n in zip(pars, parnames):
        cf.parameters.parameters[n] = p
    if not mute:
        print('updated colfile parameters')
        

def get_Xray_energy(wl):
    """ return x-ray energy (kev) from wavelength """
    E_kev = 6.62607015e-34*2.99792e8/(wl*1e-10) / 1.60218e-19 / 1e3
    return E_kev



# Operations on columnfiles: drop column, merge two columnfiles, get columnfile size in memory , etc.
########################################################################################
def merge_colf(c1, c2):
    """ merge two columnfiles with same columns (same ncols + colnames)"""
    titles = list(c1.keys())
    c_merged = columnfile.newcolumnfile(titles=titles)
    c_merged.setparameters(c1.parameters)
    
    assert c1.ncols==c2.ncols
    for i in range(c1.ncols):
        item1, item2 = list(c1.keys())[i], list(c2.keys())[i]
        assert item1 == item2
    c_merged.set_bigarray( [ np.append(c1[t], c2[t]) for t in titles ] )    
    return c_merged


def dropcolumn(cf, colname):
    """ remove column from colfile """
    assert colname in cf.titles
    
    titles = [t for t in cf.titles if t != colname]
    c_out = columnfile.newcolumnfile(titles=titles)
    c_out.setparameters(cf.parameters)
    c_out.set_bigarray( [ cf[t] for t in titles ] )
    del cf
    return c_out


def get_colf_size(cf, out=False):
    """ returns memory taken by the columnfile when loaded"""
    size_MB = sum([sys.getsizeof(cf[item]) for item in cf.keys()]) / (1024**2)
    print('Total size = ', '%.2f' %size_MB, 'MB')
    if out:
        return size_MB
    

def select_subset(cf, selection_type = 'rectangle',
                  xmin=0, ymin=0, xmax=1, ymax=1,
                  xcenter=0, ycenter=0, r=1):
    """
    select subset of peaks based on position on the map. Columnfile must contain xs, ys columns, obtained from Friedel pairing
    
    Args:
    --------
    cf : columnfile
    selection_type : either 'rectangle' or 'circle'
    xmin, ymin, xmax, ymax : rectangle vertices for rectangle selection
    xcenter, ycenter, r : center coords and radius for circle selection
    
    Returns:
    ---------
    mask : peak selection as a Boolean mask
    """
    assert selection_type in ['rectangle', 'circle']
    
    if selection_type == 'rectangle':
        assert all([xmin < xmax, ymin < ymax])
        mask = np.all([c.xs <= xmax, c.xs >= xmin, c.ys <= ymax, c.ys >= ymin], axis=0)
        
    else:
        mask = (c.xs - xcenter)**2 + (c.ys - ycenter)**2 <= r**2

    return mask


def recast(ary):
    """ 
    given an array [x1, x2,...,xn] of len n, returns recast_array [x1, x1, x2, x2, ..., xn, xn] of len 2n. Useful to work with friedel pairs labels"""
    
    return np.concatenate((ary,ary)).reshape((2,len(ary))).T.reshape((2*len(ary)))     


def select_tth_rings(cf, tth_calc, tth_tol, tth_max=20, is_sorted=False):
    """ select all peaks within tth_tol distance from a list of hkl rings. useful to select a specific phase from computed hkl rings positions.
    If corrected tth (tthc) column is present, will try to use these instead of tth
    
    Args:
    ----------
    cf: columnfile
    tth_calc: array of tth position for hkl rings
    tth_tol: tolerance in tth to select peaks around hkl rings
    tth_max: max tth cutoff
    is_sorted: if cf is already sorted on tth, can be set to True to avoid sorting it again
    
    Returns:
    ----------
    mhkl : peak selection as a Boolean mask
    """
    
    # use tth or tthc. Arrays need to be sorted on tth/tthc for indices selection
    if 'tthc' in cf.titles:
        if not is_sorted:
            cf.sortby('tthc')
        tth = cf.tthc
    else:
        if not is_sorted:
            cf.sortby('tth')
        tth = cf.tth

    # initialize indices
    inds = []
    tth_max = min(tth_max, max(tth))
    # scan each tth ring and select peaks
    
    for hkl in tth_calc:
        if hkl >= tth_max:
            break
        imin, imax = np.searchsorted(tth, hkl - tth_tol), np.searchsorted( tth, hkl + tth_tol)
        inds.extend(np.r_[imin:imax])
    inds = np.asarray(inds)

    # transform indices to a bool array of len cf.nrows
    mhkl = np.zeros(cf.nrows, dtype=bool)
    mhkl[inds] = True

    return mhkl



def compute_kde(cf, ds, tthmin=0, tthmax=20, npks_max = 1e6, tth_step = 0.001, bw = 0.001, usetthc = True,
                uself = True,  doplot=True, save = True, fname=None):
    """ compute kernel density estimate of tth column, mimmicking a powder XRD spectrum.
    
    Args:
    ----------
    cf : columnfile
    ds : dataset file
    tthmin, tth_max: two-theta range over which kde is computed
    tth_step : sampling density over the computed kde
    npks_max : max number of peaks in selection to compute kde. For large peakfile, a random subset of peaks is selected, which contains at most npks_max peaks. Necessary for large peakfiles because kde function from scipy is extremely slow when npeaks becomes very large. 
    bw : kde bandwidth
    usetthc (bool) : use corrected tth column (tthc) instead of tth (sharper peaks on the kde). default is True.
    uself (bool)   : use Lorentz scaling factor for intensity:  L( theta,eta ) = sin( 2*theta )*|sin( eta )| (Poulsen 2004) """
    
    # downsample cf if nrows > npks_max
    rnd_msk = np.full(cf.nrows, True)  # initialize random mask
    if cf.nrows > npks_max:
        p = npks_max/cf.nrows
        rnd_msk = np.random.choice(a=[True, False], size=cf.nrows, p=[p, 1-p])  
    
    # select tth col + range
    if usetthc is True:
        msk = np.all([cf.tthc <= tthmax, cf.tthc >= tthmin, rnd_msk], axis=0)
        tth = cf.tthc[msk]
    else: 
        msk = np.all([cf.tth <= tthmax, cf.tth >= tthmin, rnd_msk], axis=0)
        tth = cf.tth[msk]
    
    #Lorentz factor for intensity correction
    if uself is True:
        cor_I = cf.sum_intensity[msk] * (np.exp( cf.ds[msk]*cf.ds[msk]*0.2 ) )
        lf = refinegrains.lf(tth, cf.eta[msk])
        cor_I *= lf
        
    # compute kde
    print('computing kde. This may take a while...')
    kde = gaussian_kde(tth, bw_method=bw, weights=cor_I)

    # resample tth values with given tth_step
    x = np.arange(tth.min(),tth.max(),tth_step)
    pdf = kde.pdf(x)

    # plot kde
    if doplot:
        fig = pl.figure(figsize=(10,5))
        fig.add_subplot(111)
        pl.hist(tth, bins=x, weights=cor_I,density=True);
        pl.plot(x, pdf, '-', linewidth=1., label='bw = ' +str(bw))
        pl.xlim(tthmin, tthmax)
        pl.xlabel('tth (deg)')
        pl.legend()
        pl.title('Integrated intensity profile - '+ds.dsname)
        pl.show()
        if save:
            fig.savefig(ds.analysispath+'_kde.png', format='png')
        
    if save:
        if fname is None:
            fname = os.path.join(os.getcwd(), ds.dsname, ds.dsname+'_bw'+str(bw)+'_kde.txt')
        f = open(fname,'w')
        for l in range(len(x)):
            if format(pdf[l],'.4f') == '0.0000':    # replace zeros by 0.0001 to avoid crashes with profex
                f.write(format(x[l],'.4f')+' '+'0.0001'+'\n')
            else:
                f.write(format(x[l],'.4f')+' '+format(pdf[l],'.4f')+'\n')
        f.close()
    
    return x, pdf



def split_xy_chunks(cf,ds, nx, ny, doplot=True):
    """ chunk columnfile by (xs,ys) coordinates in the sample frame. 
    
    Args:
    ----------
    cf : columnfile
    ds : dataset file
    nx, ny: number of chunks along x and y directions
    doplot : plot chunk grid over sample reconstruction image
    
    Returns:
    updates columnfile with a new chunk_id column
    chunks : dictionnary with rectangle verticies of each chunk 
    """
    # bins for chunking
    xbins = np.linspace(ds.ybinedges.min(), ds.ybinedges.max(), nx+1)
    ybins = np.linspace(ds.ybinedges.min(), ds.ybinedges.max(), ny+1)
    
    cf.setcolumn(-1*(np.ones_like(cf.fp_id)), 'chunk_id')
    chunk_labels = np.arange(nx*ny)
    
    # chunk splitting
    chunks = {}
    for chk in tqdm.tqdm(chunk_labels):
        row = chk // nx
        col = chk % nx
        
        # x-y coords of chunk. return them as a dict
        xmin, xmax, ymin, ymax = xbins[col], xbins[col+1], ybins[row], ybins[row+1]
        chunks[chk] = [( xmin, xmax, ymin, ymax )]

        msk = select_subset(cf, 'rectangle', xmin, ymin, xmax, ymax)
        cf.chunk_id[msk] = chk
    
    if doplot:
        fpr = friedel_recon(cf, ds.ybinedges, ds.ybinedges,doplot=False, mask=None, weight_by_intensity=True, norm=True)
        pl.figure()
        pl.pcolormesh(ds.ybinedges, ds.ybinedges,fpr, cmap='Greys_r')
        pl.hlines(y = ybins[1:-1], xmin = min(xbins), xmax = max(xbins), colors='red', alpha=.8)
        pl.vlines(x = xbins[1:-1], ymin = min(ybins), ymax = max(ybins), colors='red', alpha=.8)
        pl.xlabel('x mm')
        pl.ylabel('y mm')
    return chunks

    
    
# Image reconstruction: do filtered back projection or use friedel pairs to reconstruct a 2D image of the sample
########################################################################################
def iradon_recon(cf, obins, ybins, mask=None, doplot=True, weight_by_intensity=True,circle=True, norm=False, **kwargs):
    """
    compute sinogram + filtered back_projection reconstruction
    Args:
    ---------
    cf    : imageD11 columnfile
    obins : bins array for omega  # if input from ds, use ds.obinedges
    ybins : bins array for dty,   # if input from ds, use ds.ybinedges
    mask  : boolean array of length cf.nrows to filter data (default = None)
    plot  : choose whether to plot the data (True) or just return the output as a 2D array without plotting (False)
    weights_by_intensity : weights peak peaks by intensity in the 2D-histogram. Intensity defined as log(e+sumI). default is True
    norm : normalize reconstruction and sinogram. Default is False
    **kwargs : keyword arguments passed to matplotlib 
    
    Returns:
    sino : 2D sinogram
    r    : 2D image obtained by filtered back-projection of the sinogram
    """
    if mask is None :
        mask=np.full(cf.nrows,True)
        
    if weight_by_intensity is True:
        weights = np.log(np.exp(1)+cf.sum_intensity[mask])
    else:
        weights = np.ones(cf.sum_intensity[mask].shape)
    
    sino= fast_histogram.histogram2d( cf.dty[mask], 
                                      cf.omega[mask],
                                      weights = weights, 
                                      range = [[ybins[0], ybins[-1]],
                                               [obins[0], obins[-1]]],
                                      bins = (len(ybins)-1, len(obins)-1) )
    
    outsize = sino.shape[0]
    
    if norm is True:
        sino = (sino.T/np.nanmax(sino,axis=1)).T
        sino = np.nan_to_num(sino, nan=0, copy=False)
    
    r = skimage.transform.iradon(sino,theta=obins[:-1],output_size = outsize, circle=circle)
    
    
    if norm is True:
        rmax = np.percentile(r,99.95)
        r = np.where(r <= rmax, r/rmax, 1)

    print(sino.shape, r.shape)
    
    if doplot is True:
        f, a = pl.subplots(1,2, figsize=(10,5), constrained_layout=True)
        a[0].pcolormesh( obins, ybins, sino, **kwargs);
        a[0].set_ylabel('dty')
        a[0].set_xlabel('omega')
        a[0].set_title('sinogram')

        a[1].imshow(r, **kwargs);
        a[1].set_xlabel('x')
        a[1].set_ylabel('y')
        a[1].set_title('iradon reconstruction')

    return sino, r



def friedel_recon(cf, xbins, ybins, doplot=True, mask=None, weight_by_intensity=True, norm=False, **kwargs):
    """
    Image reconstruction Friedel pair position density
    Args:
    --------
    cf    : imageD11 columnfile with friedel pairs labeled
    mask  :  boolean array of length cf.nrows to filter data (default = None)
    plot  : choose whether to plot the data (True) or just return the output as a 2D array without plotting (False)
    weights_by_intensity : weights peak peaks by intensity in the 2D-histogram. Intensity defined as log(e+sumI). default is True
    norm : normalize reconstruction and sinogram. Default is False
    **kwargs : keyword arguments passed to matplotlib 
    
    Returns:
    r : Image reconstruction
    """
    if mask is None :
        mask=np.full(cf.nrows,True)
        
    if weight_by_intensity is True:
        weights = np.log(np.exp(1)+cf.sum_intensity[mask])
    else:
        weights = np.ones(cf.sum_intensity[mask].shape)
        
    r = fast_histogram.histogram2d( cf.xs[mask], 
                                      cf.ys[mask],
                                      weights = weights,   # gives more weight to peaks with high intensity
                                      range = [[xbins[0], xbins[-1]],
                                               [ybins[0], ybins[-1]]],
                                      bins = (len(xbins), len(ybins)) );
    if norm is True:
        r = r / r.max()
    
    if doplot is True:
        f = pl.figure(figsize=(5,5))
        ax = f.add_subplot(111)
        ax.pcolormesh(xbins, ybins, r, **kwargs);
        ax.set_ylabel('y mm')
        ax.set_xlabel('x mm')
        ax.set_title('Friedel pairs reconstruction')

    return r



def random_color_map(ncolors):
    """ return random color map with specified number of colors. Useful for plotting phase or grain id maps"""
    colors = np.concatenate( (np.array([[0,0,0]]), np.random.rand(ncolors,3) ), axis=0)  
    cmap = pl.matplotlib.colors.ListedColormap(colors)
    return cmap
