""" This module contains functions to identify symmetric (h,k,l), (-h,-k,-l) reflections (aka. Friedel pairs) in a 3DXRD dataset. It works the best with scanning-3DXRD data acquired using a pencil-beam and y-translation scanning procedure, but seems to provide also good results with regular 3DXRD from a single 360° scan in omega acquiered with a box bam or a letter-box beam"""

import os, sys, h5py
import numpy as np
import pylab as pl
from tqdm import tqdm
import fast_histogram
from multiprocessing import Pool

import scipy.spatial
from scipy.sparse import csr_matrix

from ImageD11 import unitcell, blobcorrector, columnfile, transform, sparseframe, cImageD11
from pf_3dxrd import utils


# pre-processing: form pairs of y-scans (-dty,+dty), which should contain symmetric information
##############################################################

def form_y_pairs(cf, ds, disp=True):
    """ form pairs of symmetric scans (-dty,+dty) and return them as a list of tuples. Central scan (dty = 0) is paired with itself. It also
    checks that dty central scan is at position dty = 0 in ds. If not, ymin, ymax and dty in ds are updated and ybins are recomputed
    
    Args: 
    --------
    cf          : ImageD11 columnfile. Must contain dty column
    ds          : ImageD11.sinograms.dataset.Dataset class object, containing info on y-bins and omega-bins
    disp (bool) : display y-pairs
    """
    
    # check dty bins are centered on zero
    
    central_bin = len(ds.ybincens)//2
    c0 = ds.ybincens[central_bin]
    
    print('dty_center = ', '%.2f'%c0, ' on bin n°', central_bin)
    
    if abs(c0) > 0.01 * ds.ystep:
        print('dty not centered on zero. Updating dty scan positions in dataset and peakfile...')
        shift  = (ds.ymax + ds.ymin) / 2
        print('shift=', '%.4f'%shift)
        
        ds.dty = ds.dty - shift
        cf.dty = cf.dty - shift
        ds.guessbins()
    
    # form y-pairs
    hi_side = ds.ybincens[central_bin:].tolist()
    lo_side = ds.ybincens[:central_bin+1].tolist()
    
    hi_side.sort()
    lo_side.sort(reverse=True)
    
    ypairs = [(y1,y2) for (y1,y2) in zip(hi_side, lo_side)]
    ds.ypairs = ypairs
    
    if disp:
        print('dty pairs: \n ===========')
        for yp in ypairs:
            print('%.4f'%yp[0], ';', '%.4f'%yp[1])
            

    

def check_y_symmetry(cf, ds, saveplot=False, fname_plot = None):
    """check that paired scans contain about the same number of peaks + total peak intensity. For each pair, computes total intensity + total nb of peaks and plot them in function of abs(dty). If the dty and -dty plots do not fit, it is likely that the sample was not correctly aligned on the rotation center, or that dty is not centered on zero.
    
    Args:
    ---------
    cf : ImageD11 columnfile. Must contain dty column
    ds : ImageD11.sinograms.dataset.Dataset object. contains metadata
    """
    
    Nrows, Sum_I = [],[]
    
    if not hasattr(ds, 'ypairs'):
        form_y_pairs(cf, ds, disp=False)
    
    for i,j in tqdm(ds.ypairs):
        sum_I_hi = cf.sum_intensity[abs(cf.dty-i) < 0.5*ds.ystep] 
        sum_I_lo = cf.sum_intensity[abs(cf.dty-j) < 0.5*ds.ystep] 
        Sum_I.append((np.sum(sum_I_hi) , np.sum(sum_I_lo)) )
        Nrows.append((len(sum_I_hi), len(sum_I_lo)))
    
    central_bin = len(ds.ybincens)//2
    hi_side = ds.ybincens[central_bin:]

    f = pl.figure(figsize=(10,5), layout='constrained')
    f.add_subplot(121)
    pl.plot(hi_side, np.cbrt(Sum_I),'.', label=['dty','-dty'])
    pl.xlabel('|dty|')
    pl.ylabel('I^1/3')
    pl.legend()

    f.add_subplot(122)
    pl.plot(hi_side,Nrows,'.', label=['dty','-dty'])
    pl.xlabel('|dty|')
    pl.ylabel('n peaks')
    pl.legend()
    pl.show()
    
    f.suptitle('dty alignment – '+str(ds.dsname))
    if saveplot is True:
        if fname_plot is None:
            fname_plot = os.path.join(os.getcwd(), ds.dsname, ds.dsname+'_dty_alignment.png')
        f.savefig(fname_plot, format='png')
            
    
    
def select_ypair(cf, ds, pair_id):
    """ select peaks from two symmetric scans and return them as two separate columnfiles.
    
    Args : 
    ---------
    cf     : ImageD11 columnfile with complete data (all dty scans)
    ds     : ImageD11.sinograms.dataset.Dataset object
    pair_id: index of ypair to select in ds.ypairs
    
    Outputs:
    ---------
    c1, c2 : symmetric columnfiles containing respectively peaks from the hi-side (+dty) and the lo-side (-dty) of the selected ypair
    """
    
    ypair = ds.ypairs[pair_id]
    y1, y2 = ypair[0], ypair[1]
    
    c1 =  cf.copy()
    c1.filter( abs(c1.dty-y1) < 0.49 * ds.ystep )  # take a bit less than 1/2 * ystep to avoid selecting same peak multiple times
    
    c2 = cf.copy()
    c2.filter( abs(c2.dty-y2) < 0.49 * ds.ystep )
    
    return c1, c2



# functions used for pair matching
##############################################################
        
def normalized_search_space(cf, mask, flip_eta_omega=True):
    """
    returns normalized variables used to build the 4D search space (eta_n, omega_n, tth_n, I_n) in which distance matrix between peaks is computed. 
    Normalization scheme is as follows:
    eta_n   = (eta/360) mod 1
    omega_n = (omega/360) mod 1
    I_n     = (log(I) - log(I).min) / (log(I).max - log(I).min)
    tth_n   = (ln(tan_tth) - ln(tan_tth).min / (ln(tan_tth).min - ln(tan_tth).max)    where tan_tth = tan(tth)
    
    Thus, the spread in each dimension is restrained to the [0,1] interval. Intensity and 2-theta are re-scaled before normalization. 
    For intensity, logarithmic scaling is used because peak intensity spans several orders of magnitude. Thus, significant intensity variations are  
    better represented on a logarithmic scale than on a linear scale. For two-theta, the need for rescaling is related to the fact that 2-theta variations
    related to geometrical offset dx increases with the scattering angle: Δtan(2θ) ∝ dx.tan(2θ). In contrast, the difference in logarithm of tan(2θ)
    only depends on the detector distance and geometrical offset. 
    
    Args:
    ----------
    cf   : input columnfule
    mask : boolean mask to select subset in cf
    flip_eta_omega (Bool). If True, eta and omega coordinate are flipped to match fridel pairs in symmetrical scans.
    """
    if mask is None:
        mask = np.full(cf.nrows, True)
    
    # eta - omega
    if flip_eta_omega:
        eta_n = ((180 - cf.eta[mask])/360) % 1
        omega_n = ((180+cf.omega[mask])/360) % 1
    else:
        eta_n = (cf.eta[mask]/360) % 1
        omega_n = (cf.omega[mask]/360) % 1
    # sum_intensity
    logI = np.log10(cf.sum_intensity[mask])
    I_n = (logI - logI.min()) / (logI.max() - logI.min())
    # two-theta
    logtth = np.log(np.tan(np.radians(cf.tth[mask])))
    tth_n = (logtth - logtth.min()) / (logtth.max() - logtth.min())
    
    return eta_n, omega_n, tth_n, I_n
    

    
def compute_csr_dist_mat(c1, c2, dist_cutoff=1.):
    """
    Core function for friedel pair matching. 
    
    This function uses the cKDTree algorithm from scipy.spatial to construct a distance matrix between peaks from c1 (+dty) and c2 (-dty) in a
    computationally efficient way. All values above a given distance threshold (dist_cutoff) are reset to zero, so we mostly end up witb a sparse matrix 
    containing mostly zeros, except when two peaks are close to each other. This sparse matrix is then cleaned to keep only one non-zero value
    per row and column, so that each peak in c1 is associated to at most one single peak in c2, and conversely (func clean_csr_dist_mat below). 
    
    Peak distance is computed in a 4D-space defined using (2-theta, eta, omega, sum_intensity). c2 peaks are flipped in eta and omega in order to make
    them match with peaks from c1 for these coordinates. eta, omega, two-theta and sum_Intensity need to be re-scaled and normalized to vary within comparable range. 
    
    For eta and omega, simple normalization to [0,1] interval is done by dividing angle in degree by 360 and applying a modulo 1 operation
    For 2-theta and intensity, a re-scaling is done prior to normalization: f(tth) = 1/tan(tth) and f(sumI) = log(sumI). 
    
    For 2-theta (tth), f(tth) = a / [tan(tth) + b] where b is arbitrarily set to 0.04 and a is defined using the mult_fact_tth parameter
    For sum_Intensity, g(sum_intensity) = k*sum_I**1/3, where k is a constant defined using the mult_fact_I parameter.
    
    The sparse matrix could also be computed in the reciprocal space, using g-vector coordinates (gx,gy,gz) + another dimension for intensity. It seems 
    to work, but is is maybe less intuitive and I have not evaluated yet how consistent it is with pair matching in (tth, eta, omega, I) space.
    
    Args:
    --------
    c1, c2        : pair of ImageD11 columnfiles 
    dist_cutoff   : distance threshold for csr_matrix. All values above are reset to zero
    mult_fact_tth : scale_factor for 2-theta
    mult_fact_I   : scale_factor for sum_intensity
    
    Outputs:
    ---------
    dij (csr mat) : sparse distance matrix between peaks from c1 and c2. shape (c1.nrows,c2.nrows)
    """
    
    # mask to select non-paired data. fp_id contains the labels for friedel_pairs. It is initialized to -1, and then updated iteratively
    msk1 = c1.fp_id == -1
    msk2 = c2.fp_id == -1
    
    # form KDTrees and compute distance matrix
    g1 = np.transpose(normalized_search_space(c1, msk1, flip_eta_omega=False))
    a = scipy.spatial.cKDTree( g1 )
    g2 = np.transpose(normalized_search_space(c2, msk2, flip_eta_omega=True))
    b = scipy.spatial.cKDTree( g2 )
    
    dij = csr_matrix( a.sparse_distance_matrix( b, dist_cutoff ) )
                      
    return dij    


def compute_csr_dist_mat_old(c1, c2, dist_cutoff, mult_fact_tth, mult_fact_I):
    """
    DEPRECATED !!!!  Old version using non-normalized search space
   
    Peak distance is computed in a 4D-space defined using (2-theta, eta, omega, sum_intensity). c2 peaks are back-rotated in eta and omega in order to make
    them match with peaks from c1 for these coordinates. eta and omega are angles of comparable values, but intensity and 2-theta need to be rescaled 
    so that the distance between two paks along these two dimensions is similar to those of in eta and omega diensions. 
    
    For 2-theta (tth), f(tth) = a / [tan(tth) + b] where b is arbitrarily set to 0.04 and a is defined using the mult_fact_tth parameter
    For sum_Intensity, g(sum_intensity) = k*sum_I**1/3, where k is a constant defined using the mult_fact_I parameter.
    
    The sparse matrix could also be computed in the reciprocal space, using g-vector coordinates (gx,gy,gz) + another dimension for intensity. It seems 
    to work, but is is maybe less intuitive and I have not evaluated yet how consistent it is with pair matching in (tth, eta, omega, I) space.
    
    Args:
    --------
    c1, c2        : pair of ImageD11 columnfiles 
    dist_cutoff   : distance threshold for csr_matrix. All values above are reset to zero
    mult_fact_tth : scale_factor for 2-theta
    mult_fact_I   : scale_factor for sum_intensity
    
    Outputs:
    ---------
    dij (csr mat) : sparse distance matrix between peaks from c1 and c2. shape (c1.nrows,c2.nrows)
    """
    
    # mask to select non-paired data. fp_id contains the labels for friedel_pairs. It is initialized to -1, and then updated iteratively
    msk1 = c1.fp_id == -1
    msk2 = c2.fp_id == -1
    
    # rescale tth + sum_intensity to have a spread comparable to omega and eta 
    tth_1 = 1./(np.tan(np.radians(c1.tth[msk1])) + 0.04) * mult_fact_tth
    tth_2 = 1./(np.tan(np.radians(c2.tth[msk2])) + 0.04) * mult_fact_tth

    sI1 = pow( c1.sum_intensity[msk1], 1/3 ) * mult_fact_I
    sI2 = pow( c2.sum_intensity[msk2], 1/3 ) * mult_fact_I
    
    # form KDTrees and compute distance matrix
    g1 = np.transpose((c1.eta[msk1]%360, c1.omega[msk1]%360, tth_1, sI1))
    a = scipy.spatial.cKDTree( g1 )
    g2 = np.transpose(((180-c2.eta[msk2])%360, (180+c2.omega[msk2])%360, tth_2, sI2))
    b = scipy.spatial.cKDTree( g2 )
    
    dij = csr_matrix( a.sparse_distance_matrix( b, dist_cutoff ) )
                      
    return dij


                      
def clean_csr_dist_mat(dij_csr, verbose=True):
    """ clean the csr distance matrix to avoid pairing a peak from c1 with multiple peaks from c2 and conversely. Keep only the minimal non-zero value 
    in each row and column
    
    Args:
    ---------
    dij_csr (csr mat) : scipy sparse distance matrix of shape M*N, where M = c1.nrows ad M = c2.nrows
    verbose (bool)    : print some information about pairing
    
    Outputs:
    ----------
    dij_best.data   : distance for selected friedel pairs
    c1_indx, c2_indx: indices of paired peaks in c1[msk1] and c2[msk2] (msk1 and msk2 defined in compute_csr_dist_mat).
    """
    
    n_pairs_all = dij_csr.nnz  # number of non-zero elements in the sparse matrix, ie number of possible pairs      
    
    # We want minimum non zero values in each row and column. However, there seems to be no efficient way to find minimum non-zero values in the matrix
    # So here we first do element-wise inversion of non-zero data and find the position of maximum values using np.argmax()
    
    # computes the inverse of non-zero elements in the dij_csr.data.
    dij_best = dij_csr.copy()
    dij_best.data = np.divide(np.ones_like(dij_csr.data), dij_csr.data, out=np.zeros_like(dij_csr.data), where=dij_csr.data!=0)
    
    # Find max values and row index in each column, ie: find the closest match in c2 for data in c1
    maxvals  = np.max(dij_best, axis=0).toarray()[0]
    row_indx = np.argmax(dij_best, axis=0).A[0]     # m.A[0] to convert matrix to 1D numpy array
    col_indx = np.arange(len(row_indx))
    # update csr matrix to keep only maxvals
    dij_best = csr_matrix((maxvals, (row_indx, col_indx)), shape = dij_best.shape )
    
    # do the same as above, but working on rows, ie: find the closest match in c1 for data in c2
    maxvals = np.max(dij_best, axis=1).toarray()[:,0]
    col_indx = np.argmax(dij_best, axis=1).A[:,0]    
    row_indx = np.arange(len(col_indx))
    dij_best = csr_matrix((maxvals, (row_indx, col_indx)), shape = dij_best.shape )
    
    # drop all zero values
    dij_best.eliminate_zeros()  
    
    # inverse data to get back to initial distance
    dij_best.data = np.divide(np.ones_like(dij_best.data), dij_best.data, out=np.zeros_like(dij_best.data), where=dij_best.data!=0)
    
    n_pairs_cleaned = dij_best.nnz   # number of pairs eventually retained
    
    if verbose:
        print(n_pairs_cleaned, 'pairs kept out of ', n_pairs_all, ' possible matches')
    
    c1_indx, c2_indx = dij_best.nonzero()  # friedel pairs indices in c1 and c2 
    
    return dij_best.data, c1_indx, c2_indx
  


def label_friedel_pairs(c1, c2, dist_max=1., dist_step=0.1, verbose=True, doplot=False):
    """
    Big scary function to find Friedel pairs in symmetric columnfiles c1 and c2. It is organized in different steps:
    
    INITIALIZATION 
    - initialize friedel pairs labels (fp_id) and friedel pair distance (fp_dist) in c1 and c2 as arrays containing -1
    - initialize some lists to store all friedel pair labels (useful in the end)
    
    FRIEDEL PAIR SEARCH LOOP
     - Run 'compute_csr_dist_mat' and 'clean_csr_dist mat -> returns a list of indices in c1 and c2 corresponding to friedel pairs
     - Update the labels fp_id and fp_dist in c1 and c2 with newly identified pairs
     - Repeat iteratively the procedure above on remaining non-paired peaks using a larger dist_cutoff threshold, and continue these iterations until maximum
     distance threshold is reached
     
     dist_cutoff increase and max distance are controled with the two parameters dist_step and dist_max: 
     search_steps = np.arange(dist_step, dist_max+dist_step, dist_step)
    
    MERGE PAIRED DATA
     After the search loop is finished, screen out unpaired peaks (fp_id == -1) in c1 and c2 and merge the two columnfiles. 
     Sort the merged columnfile by fp_id, so c_merged.fp_id = [0,0,1,1,2,2,...n,n]
     
    PLOT
     plot some statstics about the distance between paired peaks. 4D distance in normalized 4D search space 
     and distance along each individual dimension, expressed in more meaningful quantities (angles for eta, omega and two-theta, normalized intensity for I)
     
     
    Args:
    --------
    c1, c2    : set of columnfiles corresonding to symmetric scans [dty, -dty]
    dist_max  : float parameter controlling the max distance threshold to apply for the Friedel pair search loop.
                Take something close to 1 is usually a good guess. 
    dist_step : float parameter controlling dist_cutoff increase at each iteration. Must be << dist_max. Start with something close to 0.1 first, and adjust if needed
    
    verbose : (bool) print information about pairing process
    doplot  : (bool) plot some statistics to evaluate quality of pairing
    
    
    Outputs:
    ---------
    c_merged  : merged columnfile containing paired peaks in c1 and c2, with friedel pair labels (fp_id) and distance between paired peaks (fp_dist)
    """
    
    # INITIALIZATION
    ###############################################################################################
    # sort c1 and c2 on spot3d_id at the begining. No sure whether it is useful, but does not harm
    c1.sortby('spot3d_id')
    c2.sortby('spot3d_id')
    
    # create new friedel pair label + fp_dist for c1 and c2,and initialize all values to -1 (non paired)
    c1.addcolumn(np.full(c1.spot3d_id.shape, -1, dtype=int), 'fp_id')
    c2.addcolumn(np.full(c2.spot3d_id.shape, -1, dtype=int), 'fp_id')
    
    c1.addcolumn(np.full(c1.spot3d_id.shape, -1, dtype = np.float64), 'fp_dist')
    c2.addcolumn(np.full(c2.spot3d_id.shape, -1, dtype = np.float64), 'fp_dist')
    
   
    fp_labels = []   # friedel pair labels list, updated at each iteration with newly found set of friedel pairs
    npkstot = min(c1.nrows, c2.nrows)  # maximum number of pairs to find. Used to compute proportion of paired peaks
    sumI_tot = sum(c1.sum_intensity) + sum(c2.sum_intensity)
    
    # FRIEDEL PAIR SEARCH LOOP
    ###############################################################################################
    dist_steps = np.arange(dist_step, dist_max+dist_step, dist_step)  # list of dist_cutoff steps over which we will iterate
    for it,dist_cutoff in enumerate(dist_steps):
        
        # sort colfiles by fp_id, to put all unpaired peaks at the begining of the columnfile
        c1.sortby('fp_id')
        c2.sortby('fp_id')        
        
        # find Friedel pairs. c1_indx and c2_indx are indices of paired peaks in c1[msk1] and c2[msk2] (msk1 and msk2 defined in compute_csr_dist_mat)
        # Since c1 and c2 have been sorted by fp_id, putting all unpaired peaks at the begining, indices to select are the same in c1 and c2
        try:
            dij = compute_csr_dist_mat(c1, c2, dist_cutoff)
            dist, c1_indx, c2_indx = clean_csr_dist_mat(dij, verbose=verbose)
        except Exception as e:
            print(f'Pairing error at dist_step {dist_cutoff:.2f}. Skip it')
            continue
                
        # update the list of friedel pair labels
        if not fp_labels:  # list is empty
            newlabels = np.arange(len(c1_indx))
        else:
            newlabels = np.arange(max(fp_labels)+1, max(fp_labels)+len(c1_indx)+1)
        
        fp_labels.extend(newlabels)
        
        
        # sanity check. make sure we are not overwriting already indexed pairs
        assert np.all([i == -1 for i in c1.fp_id[c1_indx]])
        assert np.all([j == -1 for j in c2.fp_id[c2_indx]])
        
        # update fp_id and fp_dist in c1, c2
        c1.fp_id[c1_indx]   = c2.fp_id[c2_indx]   = newlabels
        c1.fp_dist[c1_indx] = c2.fp_dist[c2_indx] = dist
        
     
    # MERGE PAIRED DATA
    ###############################################################################################
    # keep only paired peaks
    c1.filter(c1.fp_id != -1)
    c2.filter(c2.fp_id != -1)
    
    #merged the two columnfiles and sort again by fp label
    c_merged = utils.merge_colf(c1, c2)
    c_merged.sortby('fp_id')
    
    if verbose: 
        print(f'==============================\nFriedel pair matching Completed.')
        print(f'N pairs = {len(fp_labels)} out of {npkstot} possible candidates')
        print(f'Fraction of peaks matched = {len(fp_labels)/npkstot:.2f}')
        print(f'Fraction of intensity matched = {sum(c_merged.sum_intensity)/sumI_tot:.2f}')
   
    

    # PLOT SOME STATISTICS: distance between paired peaks along each dimension
    ###############################################################################################

    if doplot:
        m1 = c1.fp_id>=0
        m2 = c2.fp_id>=0
        
        if verbose:
            print(f'dstep_max = {dist_steps[-1]:.3f}')
        
        # coordinates in norrmalized search space
        e1,o1,tth1,sI1 = normalized_search_space(c1, mask=m1, flip_eta_omega=False)
        e2,o2,tth2,sI2 = normalized_search_space(c2, mask=m2, flip_eta_omega=True)
        
        # distance along each dimension. Those are rescaled to some meaningful range: real angles in eta, omega, tth; sumI is kept in normalized space
        eta_dist   = (e2 - e1) * 360
        omega_dist = (o2 - o1) * 360
        tth_dist   = c2.tth[m2] - c1.tth[m1]
        sumI_dist  = sI2 - sI1
        
        # for scaling axes
        def x_lim(x):
            return np.percentile(x,2), np.percentile(x,98)
        def x_bins(x):
            return np.linspace(np.percentile(x,2), np.percentile(x,98),200)
        
        
        fig = pl.figure(figsize=(8,10))
        
        ax1 = fig.add_subplot(311)
        ax1.hist(c1.fp_dist[m1], bins=x_bins(c1.fp_dist[m1]), density=True);
        ax1.set_xlabel('Normalized distance between pairs in 4D search-space')
        ax1.set_ylabel('density of pairs')
        ax1.set_xlim(x_lim(c1.fp_dist[m1]))
        
        ax2 = fig.add_subplot(323)
        ax2.hist(eta_dist, bins=x_bins(eta_dist), density=True);
        ax2.set_xlabel('eta mismatch (deg)')
        ax2.set_xlim(x_lim(eta_dist))
        
        ax3 = fig.add_subplot(324)
        ax3.hist( omega_dist , bins=x_bins(omega_dist), density=True);
        ax3.set_xlabel('omega mismatch (deg)')
        ax3.set_xlim(x_lim(omega_dist))
        
        ax4 = fig.add_subplot(325)
        ax4.hist(tth_dist , bins=x_bins(tth_dist), density=True);
        ax4.set_xlabel('2θ mismatch (deg)')
        ax4.set_xlim(x_lim(tth_dist))
        
        ax5 = fig.add_subplot(326)
        ax5.hist(sumI_dist, bins=x_bins(sumI_dist),density=True);
        ax5.set_xlabel('normalized intensity mismatch')
        ax5.set_xlim(x_lim(sumI_dist))
                     
        fig.suptitle('Mismatch between paired peaks')
        
    return c_merged




def process_ypair(args):
    """ group processing of ypair in a single function, which takes a list of arguments args as input 
    Aimed to parallelize pairing process for all ypairs """
    cf, ds, pair_id, dist_max, dist_step, doplot = args
    c1, c2 = select_ypair(cf, ds, pair_id)
    c_merged = label_friedel_pairs(c1, c2, dist_max, dist_step, verbose=False, doplot=doplot)
    c_merged.sortby('fp_id')
    return c_merged




def find_all_pairs(cf, ds, dist_max=.1, dist_step=0.01, doplot=True, verbose=True, saveplot=False):
    """
    process successively all ypairs [-dty; +dty ] in ds.ypairs, and concatenate all the output into a new columnfile
    with friedel pairs index (fp_id) and distance between paired peaks (fp_dist). Make sure each pair get a unique label in fp_id
    
    Args:
    ----------
    cf   : ImageD11 columnfile to pair
    ds   : ImageD11.sinogram.dataset.Dataset metadata, which contains the list of ypairs
    dist_max, dist_step : parameters to control distance threshold for the friedel pair search loop (see label_friedel_pairs for explanation)
    
    doplot (bool) : plot fp_dist statistics
    verbose (bool): print some informaton oabout indexing
    saveplot (bool) : save fp_dist statistics plot 
    
    Returns:
    --------------
    cf_paired : filtered peakfile with matched peaks sorted by friedel pair id
    fp_labels : list of fridel pair labels
    stats     : matching statistics. [frac_pks_matched, frac_intensity_matched]
    """
    
    # check if ds contains ypairs. if not, compute them
    if not hasattr(ds, 'ypairs'):
        form_y_pairs(cf, ds, disp=True)
    
    
    # Run friedel pair search on each pair.
    ##########################################################################
    # list of arguments to be passed to process_ypair
    args = []
    ypairs = sorted(ds.ypairs)
    for pair_id in range(len(ypairs)):
        args.append((cf, ds, pair_id, dist_max, dist_step, False))
    
    print('Friedel pair search...')

    out = []
    for arg in tqdm(args):
        o = process_ypair(arg)
        out.append(o)
    
    ###########################
    # pool object for multithreading. Does not split jobs in different threads, I don't know why. To fix
    #t0 = timeit.default_timer()
    #nthreads = os.cpu_count() -1
    
    #if __name__ == '__main__':
    #    with Pool(nthreads) as pool:
    #        out = list(tqdm(pool.map(process_ypair, args), total = len(args)))
    ###########################

    if verbose:
        print('Friedel pair search completed.')
    
    
    # group all outputs into one single columnfile, and update fp_labels to make sure each pair has a unique label
    ##########################################################################
    #initialization
    c_cor = out[0]
    c_cor.sortby('fp_id')
    fp_labels = np.unique(c_cor.fp_id)
    
    # update fp_labels
    if verbose:
        print('Updating new Friedel pair labels')
    for colf in out[1:]:    
        newlabels = np.arange(max(fp_labels)+1, max(fp_labels)+len(np.unique(colf.fp_id))+1)
        fp_labels = np.concatenate((fp_labels, newlabels))
        colf.sortby('fp_id')
        colf.setcolumn(utils.recast(newlabels), 'fp_id')
    
    # merge columnfiles
    if verbose:
        print('Merging peakfiles...')
    for colf in tqdm(out[1:]):
        c_cor = utils.merge_colf(c_cor, colf)
    
    # matching stats
    frac_pks_matched = c_cor.nrows / cf.nrows
    frac_ints_matched = sum(c_cor.sum_intensity) / sum(cf.sum_intensity)
    
    if verbose:
        print('==============================\nFriedel pair matching Completed.')
        print('N pairs = ', int(c_cor.nrows/2))
        print(f'Fraction of peaks matched = {frac_pks_matched:.2f}')
        print(f'Fraction of total intensity matched = {frac_ints_matched:.2f}')
    
    
    if doplot:
        fig = pl.figure(figsize=(7,4))
        
        ax1 = fig.add_subplot(111)
        ax1.hist(c_cor.fp_dist, bins = np.linspace(0, np.percentile(c_cor.fp_dist,99),200), density=True);
        ax1.set_xlabel('4D distance (tth, eta, oega, I)')
        ax1.set_ylabel('prop. of pairs')
        ax1.set_xlim(0, np.percentile(c_cor.fp_dist,99))            
        fig.suptitle('4D distance between identified pairs')
        fig.show()
        
        if saveplot:
            fig.savefig(os.path.join(os.getcwd(), ds.dsname, ds.dsname+'_fp_dist.png'), format='png')
                         
    return c_cor, fp_labels, [frac_pks_matched,frac_ints_matched]




# Geometry correction
##############################################################

           
        
def update_geometry_s3dxrd(cf, detector = 'eiger', update_gvecs=True):
    """ update geometry using friedel pairs, for a scanning 3dxrd acquisition (pencil-beam). 
    
    Details:
    If a grain is not precisely centered on the rotation axis but instead shifted by a certain translation vector (dx, dy, dz) from the center, it will cause an offset on the detector resulting in inaccurate measurements of 2-theta and eta values. Traditionally, these translation parameters, along with eta and 2-theta, are adjusted after the indexing process. However, with Friedel pairs, this adjustment can be made without the need for indexing.
    
    In a scanning 3dxrd experiment, the size of the thin pencil beam in y and z is small, and thus we can consider that grain offset from the
    rotation center only occurs along the beam, ie in the x-direction, and offset in y and z is negligible. This implies that only the 2-theta angle (tth)
    is affected by this offset, not eta.
    
    We also know the offset in the y-direction dy, which is basically the translation dty of a given scan, stored in the dty column of the peakfile. 
    With these assumptions, we find that for two peaks (p1,p2) forming a Friedel pair:
    
    * tth_cor = 1/2.(tan1 + tan2) 
    * dx = L * (tan1-tan2)/(tan1+tan2)
    
    where tan1 and tan2 are respectively tan(tth1) and tan(tth2) of p1 and  p2 and L is the distance of the detector from the rotation center. 
    Thus, tth_cor provides the "true" 2-theta angle, which reflects solely the d-spacing in the crystal, excluding any offset from the rotation center.
    
    To obtain dx and dy in the sample reference frame, they need to be back-rotated by an angle omega. This rotation places them in the sample's reference frame, allowing determination of the grain's position (xs, ys) in the sample, which is then utilized for point-by-point fitting.
    
    Args:
    -------
    cf       : ImageD11 colunfile with with Friedel pair labels 'fp_id' and friedel pair distance 'fp_dist'
    detector : either' eiger' or 'frelon'. The two stations on ID11 use different distance units (µm/mm), this is just to get the distances right
    update_gvecs : (bool). If True, recompute g-vectors using to corrected tth
    """
    
    assert detector in ['eiger', 'frelon'], 'detector must be either "eiger" or "frelon"' 
    
    # check that friedel pairs have been labeled and that the columnfile is not corrupted, extract fp_ids and fp_dist and reshape them 
    ################################################################################
    assert np.all(['fp_id' in cf.titles, 'fp_dist' in cf.titles]), 'friedel pairs have not been labeled'
    assert np.all([cf.fp_id.min() >= 0, cf.fp_dist.min() != -1])  # check that all peaks in cf have been labeled           
    assert cf.nrows % 2 == 0
    
    cf.sortby('fp_id') # sort by fp label
    
    # define masks to split data into two twin columnfiles, each containing one item of each pair
    m1 = np.arange(0, cf.nrows, 2)
    m2 = np.arange(1, cf.nrows, 2)
    
    # check all fp_ids and fp_dist match
    assert np.all(np.equal(cf.fp_id[m1],cf.fp_id[m2]))
    assert np.all(np.equal(cf.fp_dist[m1], cf.fp_dist[m2]))
    
    
    # compute corrected tth + (xs,ys) peak position in sample space
    #################################################################################
    
    wl = cf.parameters.get('wavelength')
    L = cf.parameters.get('distance')
    
    # tth correction
    #################
    tan1  = np.tan(np.radians(cf.tth[m1]))
    tan2  = np.tan(np.radians(cf.tth[m2]))
    tth_cor = np.degrees(np.arctan( (tan1 + tan2)/2 ) ) 
    ds_cor = 2 * np.sin(np.radians(tth_cor)/2) / wl      # 1/d-spacing
    
    # compute (dx, dy): distance of peak from rot center along x and y axes, in lab reference frame
    ##################
    dy = (cf.dty[m1] - cf.dty[m2]) / 2
    
    if detector == 'frelon':  # dty is given in mm with the Frelon, so we convert distance to mm 
        L  = L/1000    
    dx = L * (tan1-tan2)/(tan1+tan2)  
    
    # rearrange dx, dy arrays
    dx = utils.recast(dx)
    dy = utils.recast(dy)
    r_dist = np.sqrt(dx**2 + dy**2)  # euclidian distance from rotation center in sample frame
    
    # compute (xs,ys): peak origin in sample reference frame
    ##################
    o = np.radians(cf.omega)
    o[m2] = (o[m2]-np.pi)%(2*np.pi)  # rotate omega by 180° for the second half of the data
    co,so = np.cos(o), np.sin(o)

    # omega can be slightly different between two paired peaks -> would lead to different xs,ys, which causes issues later in the processing
    # for each pair, take average value of omega and assign it to both peaks 
    co = utils.recast((co[m1]+co[m2])/2)
    so = utils.recast((so[m1]+so[m2])/2)
    xs = co*dx + so*dy
    ys = -so*dx + co*dy
    
    
    # recast arrays and add them as new columns in cf
    ##################
    cf.addcolumn(utils.recast(tth_cor), 'tthc')
    cf.addcolumn(utils.recast(ds_cor), 'dsc')
    cf.addcolumn(xs, 'xs')
    cf.addcolumn(ys, 'ys')
    cf.addcolumn(r_dist, 'r_dist')
    
    # update gvectors
    if update_gvecs:
        cf.gx, cf.gy, cf.gz = transform.compute_g_vectors(cf.tthc, cf.eta, cf.omega,
                                                          wvln  = wl,
                                                          wedge = cf.parameters.get('wedge'),
                                                          chi   = cf.parameters.get('chi'))
    


    
    

    

def update_geometry_boxbeam_3dxrd(cf):
    """
     update geometry using friedel pairs, in the case of a regular 3dxrd acquisition (box beam or letter-box beam)
     
    Details:
    If a grain is not precisely centered on the rotation axis but instead shifted by a certain translation vector (dx, dy, dz) from the center,
    it will cause an offset on the detector resulting in inaccurate measurements of 2-theta and eta values. Traditionally, these translation parameters,
    along with eta and 2-theta, are adjusted after the indexing process. However, with Friedel pairs, this adjustment can be made without the need for 
    indexing.
    
    In a standard 3DXRD acquisition (unlike scanning 3DXRD), the dimensions of the beam in y and z directions cannot be neglected. Consequently, the translation vector (dx, dy, dz) has three unknowns, and both 2-theta and eta require correction. Unfortunately, it's not possible to solve all these parameters with just one Friedel pair. While 2-theta and eta can be determined, solving for the translation vector (dx, dy, dz) leaves the system of equations underdetermined, with one degree of freedom remaining. Therefore, the positions of grains in the sample are adjusted later, using all the indexed peaks for each grain.
    
    Visualizing the problem with cartesian coordinates in the lab reference frame (xl, yl, zl) while considering the full experimental setup (beam + detector) rotating around the sample during a scan makes it easier to understand. In this setup, the peaks forming a Friedel pair (p1 and p2) and the grains they originate from are aligned, regardless of the grain's position in the sample. The orientation of the line (p1p2) is only determined by the lattice spacing and orientation of the grain. Therefore, the coordinates of the "true" diffraction vector are obtained by halving the vector from p2 to p1. Considering the actual experimental setup where the detector remains fixed while the sample rotates, this yields the following coordinates for the corrected diffraction vector (xl,yl,zl) in the laboratory reference frame:
    
    xl = 1/2 . (xl1 + xl2)
    yl = 1/2 . (yl1 + yl2)
    zl = 1/2 . (zl1 - zl2)
    
    the symmetric reflection (-h,-k,-l) is then (xl, yl, -zl) When these values have been calculated, tth and eta can be computed using
    ImageD11.transform.compute_tth_eta_from_xyz
     
    """
    cf.sortby('fp_id')
    
    # get lab coordinates without detector tilt
    #############################################
    
    # replace raw pixel coordinates by corrected coordinates
    cf2 = cf.copy()
    cf2.s_raw = cf2.sc
    cf2.f_raw = cf2.fc
    
    # reset tilt and wedge parameters and run updadeGeometry()
    pars = deepcopy(cf_paired.parameters)
    pars.parameters['tilt_x'] = 0
    pars.parameters['tilt_y'] = 0
    pars.parameters['tilt_z'] = 0
    pars.parameters['wedge']  = 0
    
    cf2.setparameters(pars)
    cf2.updateGeometry()
    
    c1, c2 = friedel_pairs.split_fpairs(cf2)  # peak coordinates (xl,yl,zl) assuming no detector tilt
    
    del cf2  # not needed anymore
    
    
    # compute corrected diffraction vector v
    ##############################################
    # apparent diffraction vectors (v1,v2) in lab coordinates. Both can be described as the sum of the true diffraction vector +  some offset (dx,dy,dz)
    xl1, yl1, zl1 = c1.xl, c1.yl, c1.zl
    xl2, yl2, zl2 = c2.xl, c2.yl, c2.zl
    
    # corrected diffraction vector in lab coordinates
    x = (xl1 + xl2) / 2  #  take average for sanity, but xl1 and xl2 should be equal since xli,yli,zli have been corrected for detector tilt
    y = (yl1 + yl2) / 2
    z = (zl1 - zl2) / 2
    
    # build lab coordinate arrays for both peaks in each friedel pairs: p(hkl)-> (x,y,z); p'(-h-k-l) -> (x,y,-z) 
    x_c = utils.recast(x)
    y_c = utils.recast(y)
    z_c = np.concatenate((z,-z)).reshape((2,len(z))).T.reshape((2*len(z)))
    
    # compute corrected values in peakfile: lab coordinates, (tth,eta) and g-vectors
    ###############################################
    
    # lab coordinates
    cf.addcolumn(x_p, 'xl_c')
    cf.addcolumn(y_p, 'yl_c')
    cf.addcolumn(z_p, 'zl_c')
    
    # tth-eta coordinates
    tth_c, eta_c = ImageD11.transform.compute_tth_eta_from_xyz(np.array([x_c, y_c, z_c]), omega=cf.omega)
    cf.addcolumn(tth_c, 'tth_c')
    cf.addcolumn(eta_c, 'eta_c')
    
    # g-vectors
    gx, gy, gz = ImageD11.transform.compute_g_vectors(cf.tth_c, cf.eta_c, cf.omega,
                                                                          wvln  = cf.parameters.get('wavelength'),
                                                                          wedge = cf.parameters.get('wedge'),
                                                                          chi   = cf.parameters.get('chi'))
    cf.addcolumn(gx, 'gx')
    cf.addcolumn(gy, 'gy')
    cf.addcolumn(gz, 'gz')
    
    
    



    
# other functions
##############################################################
    
def split_fpairs(cf):
    """ split columnfile with friedel pairs into two columnfiles, each containing one peak of each pair
    Args:  cf : columnfile with fp_id column
    Outputs : c1, c2: splitted columnfiles
    """
    
    cf.sortby('fp_id') # sort by fp label
    
    # define masks to split data into two twin columnfiles, each containing one item of each pair
    m = np.arange(cf.nrows)%2 == 0
    
    # check that splitting is ok
    assert np.all(np.equal(cf.fp_id[m],cf.fp_id[~m]))
    assert np.all(np.equal(cf.fp_dist[m],cf.fp_dist[~m]))
    
    # filter peaks
    c1 = cf.copy()
    c1.filter(m)
    c2 = cf.copy()
    c2.filter(~m)
    
    return c1, c2




def exclude_singles(cf):
    # filter peakfile to remove all peaks which do not have a Friedel pair associated
    assert 'fp_id' in cf.titles, 'Friedel pairs have not been identified. Run friedel pair search before'
    
    cf_filtered = cf.copy()
    cf_filtered.sortby('fp_id')
    
    # find fp_id values occurring twice
    uniq, ind, cnt = np.unique(cf.fp_id, return_index=True, return_counts=True)
    twins_inds = np.concatenate([(i,i+1) for i in ind[cnt==2]])

    # define mask and keep only twins 
    mask = np.full(cf_filtered.nrows, False)
    mask[twins_inds] = True
    
    cf_filtered.filter(mask)
    
    # sanity check 
    assert np.all(np.equal(cf_filtered.fp_id[::2], cf_filtered.fp_id[1::2]))
    
    return cf_filtered




def find_missing_twins(cf, selection, restrict_search=False, restrict_subset=[]):
    """ Identifies single peaks (unique fpids) in a subset of a paired columnfile and find their friedel twin in the full columnfile or in a second subset.
    Inputs:
    cf: columnfile with friedel pair identified (fp_id column). 
    It has an even number of rows and exactly two peaks with same fp_id for all  values in fp_id
    restrict_search: limit search for missing peaks to a subset of cf (faster if cf is large)
    restrict_subset: subset to search in
    
    Output: new_selection: updated selection of peaks with missing peaks added
 """
    
    fpids = cf.fp_id[selection]  # fpids in selection
    
    # use np.unique to find "single" peaks: peaks missign their friedel twin
    _, ind, cnt = np.unique(fpids, return_counts=True, return_index=True)
    fp_single = fpids[ind[cnt!=2]]
    
    if len(fp_single)==0:
        return selection
    
    # list of peaks ids to select in columnfile
    if restrict_search is True:
        pks_to_add = np.concatenate([np.argwhere(cf.fp_id[restrict_subset] == i) for i in fp_single])[:,0]   # pks indices to select in subset
        newpks = restrict_subset[pks_to_add]  # pks indices to select in full subset
    else:
         newpks = np.concatenate([np.argwhere(cf.fp_id == i) for i in fp_single])[:,0]
    
    new_selection = np.unique( np.concatenate((selection, newpks)) )  # concatenate with former selection
    return new_selection
