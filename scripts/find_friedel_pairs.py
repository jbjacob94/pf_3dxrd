"""
Run Friedel pair search for all scans in a peakfile. Make the code parallel over each pair of scans (dty,-dty) to match
"""

import os, sys, time
import argparse
import subprocess
import numpy as np
import pylab as pl
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

import ImageD11.columnfile
import ImageD11.sinograms.dataset
import ImageD11.sinograms.properties
import ImageD11.parameters
import ImageD11.sparseframe
import ImageD11.blobcorrector

if '/home/esrf/jean1994b' not in sys.path:
    sys.path.append('/home/esrf/jean1994b')
from pf_3dxrd import utils, friedel_pairs


# UPDATE OPTIONS HERE
######################################################################

class Options():
    ## PAIRING OPTIONS
    dist_max  = 1.5      # max distance for nearest-neighbour search
    dist_step = 0.1      # distance step incrrase at each iteration
    sf_tth    = 1.       # scale factor two-theta
    sf_I      = 1.       # scale factor sum intensity
    
    ## PEAKS FILTERING
    # before Friedel pair matching: remove weak peaks (background noise) + suspiciously strong peaks
    Npx_min   = 3        # min nb of pixels
    sumI_min  = 30       # min sum_intensity 
    sumI_max  = 1e7      # max sum_intensity
    
    # after Friedel pairs identification. screen out dodgy friedel pairs from the output
    max_tth_dist   = 1.     # max two-theta angle between two peaks in a pair (degree) 
    max_eta_dist   = 1.     # max eta angle between two peaks in a pair (degree) 
    max_omega_dist = 1.     # max omega angle between two peaks in a pair (degree) 
    max_logI_dist  = 1.     # max intensity difference (in log units) between two peaks in a pair
    
    # MULTIPROCESSING OPTIONS
    ncpu = len(os.sched_getaffinity( os.getpid() ))    # ncpu. by default, use all cpus available
    chunksize = 4                                      # size of chunks passed to ProcessPoolExecutor
    
# END EDITABLE PART
######################################################################
    
    
    def __str__(self):
        attrdict = {i: self.__getattribute__(i) for i in dir(self) if not i.startswith('__')}
        return '\n'.join([f'{k}:{v}' for (k,v) in attrdict.items()])
        
######################################################################


def reformat_args(parser):
    args = parser.parse_args()
    if args.use2Dpeaks is None:
        args.use2Dpeaks = False
    return args



def load_data(pksfile, dsfile, parfile, return_2D_peaks=False):
    """ 
    load data from peakfile or peaktable 
    
    Args:
    ----------
    dsname (str) : dataset name
    parfile (str): parameter file name
    return_2D_peaks (bool): returns 2D peaks instead of 3D peaks (merged in omega). Only relevant for data loaded from a peak table
    
    Returns:
    cf : ImageD11 columnfile
    ds : ImageD11 dataset
    """
        
    # load ds file
    ####################################
    ds = ImageD11.sinograms.dataset.load(dsfile)

    print(f'Loading data for dataset {ds.dsname}: \n==============================')    
    items = 'n_ystep,n_ostep,ymin,ymax,ystep,omin,omax,ostep'.split(',')
    vals  = [ds.shape[0], ds.shape[1], ds.ymin, ds.ymax, ds.ystep, ds.omin, ds.omax, ds.ostep]
    for i,j in zip(items, vals):
        print(f'{i}: {j:.1f}')
        
    print('==============================')
    
    
    # load peaks
    ####################################
    print('loading peaks...')
    
    # Try from peakfile first
    try:
        cf = ImageD11.columnfile.columnfile(pksfile)
        print('loaded peaks from peakfile')
        print(f'nrows = {cf.nrows}')
        utils.get_colf_size(cf)
        return cf, ds
    except:
        pass
    
    # If not a peakfile, try from peaks table
    try:
        pkst = ImageD11.sinograms.properties.pks_table.load(pksfile)
        print('loaded peaks peaks table')
        
    except:
        raise NameError('pksfile format not recognized')
        
    print('merging peaks...')
    if return_2D_peaks:      
        pkd = pkst.pk2d( ds.omega, ds.dty )         # for 2D peaks
    else:
        pkd = pkst.pk2dmerge( ds.omega, ds.dty )  # for 3D peaks
            
    cf = ImageD11.columnfile.colfile_from_dict(pkd)
    print(f'loaded peakfile from peaks table')
    print(f'nrows = {cf.nrows}')
    utils.get_colf_size(cf)
    return cf, ds

        

def correct_distortion_eiger( cf, parfile,
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
    
    spat = ImageD11.blobcorrector.eiger_spatial( dxfile = dxfile, dyfile = dyfile )
    cf = ImageD11.columnfile.colfile_from_dict( spat( {t:cf.getcolumn(t) for t in cf.titles()} ) )
    cf.parameters.loadparameters(parfile)
    cf.updateGeometry()
    return cf


def correct_distortion_frelon( cf, parfile, splinefile):
    """ 
    FOR FRELON DATA. Apply detector distortion correction using a pixel look up table computed from a splinefile. 
    Adds on the geometric computations (tth, eta, gvector, etc.)
    
    Args: 
    ---------
    cf : ImageD11 columnfile
    parfile (str) : parameter file
    splinefile    : detector distortion splinefile
    """
    
    spat = ImageD11.blobcorrector.correctorclass(splinefile)
    
    # make pixel_lut  + substract xy grid coordinate (i,j) to keep only dx and dy arrays.
    spat.make_pixel_lut(spat.dim)
    i, j = np.mgrid[ 0:spat.dim[0], 0:spat.dim[1] ]
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



@contextmanager
def suppress_stdout():
    saved_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout = saved_stdout


    
#####################################################################
#####################################################################
    
def main():
    
    # extract arguments & options
    #################################################################
    args = reformat_args(parser)
    OPTS = Options()
    
    print('\n---------------------------------')
    print(f'PARAMETERS FOR FRIEDEL PAIR SEARCH:')
    print(OPTS)
    print('use2Dpeaks', args.use2Dpeaks)
    print('---------------------------------')
    
    # load peaktable, export to colfile, correct distortion
    #################################################################
    cf, ds = load_data(args.pksfile, args.dsfile, args.parfile, args.use2Dpeaks)
    

    if 'frelon' in ds.detector:
        cf = correct_distortion_frelon( cf, args.parfile, args.splinefile)

    elif 'eiger' in ds.detector:
        cf = correct_distorsion_eiger(cf, parfile, dxfile = args.dxfile, dyfile=args.dyfile)
    else:
        raise NameError('detector type not recognized')
    
    # folder for saving output: same as folder containing the pksfile
    savedir = os.path.dirname(args.pksfile) 
    subprocess.run(f'mkdir -p {savedir}'.split(' '), check=True)
    
    # filter dodgy peaks
    if 'Number_of_pixels' in cf.titles:
        cf.filter(cf.Number_of_pixels >= OPTS.Npx_min)
    elif 'number_of_pixels' in cf.titles:
        cf.filter(cf.number_of_pixels >= OPTS.Npx_min)
    else:
        raise AttributeError('columnfile object has no attribute "Number_of_pixels" or "number_of_pixels"')
    
    cf.filter(cf.sum_intensity  <= OPTS.sumI_max)
    cf.filter(cf.sum_intensity  >= OPTS.sumI_min)
    
    
    # Prepare for pairing
    #################################################################
    # form pairs of dty scans
    print('\n====================\n====================\nForm pairs of dty scans and check symmetry')
    friedel_pairs.form_y_pairs(cf, ds, disp=True)
    friedel_pairs.check_y_symmetry(cf, ds, saveplot=True, fname_plot = os.path.join(savedir, ds.dsname+'_dty_alignment.png'))
    
    # build argument list for parallel processing
    argslist = []
    for pair_id,_ in enumerate(ds.ypairs):
        argslist.append((cf, ds, pair_id, OPTS.dist_max, OPTS.dist_step, OPTS.sf_tth, OPTS.sf_I, False))
    
    
    # match friedel pairs in each pair of scans and write outputs in temporary files
    #################################################################
    print('\nMatch Friedel pairs\n====================')
    subprocess.run(f'mkdir -p tmp'.split(' '), check=True)  # tmp folder for outputs
    
    # simple loop without parallelization. For debugging purposes
    #for i,a in tqdm(enumerate(argslist), total=len(argslist)):
    #    r = friedel_pairs.process_ypair(a)
    #    utils.colf_to_hdf(r, f'tmp/cfp_{str(i)}.h5', save_mode='full')
    
    with ProcessPoolExecutor() as pool:
        pool.max_workers=OPTS.ncpu
        pool.mp_context=multiprocessing.get_context('fork')
        
        for i,r in tqdm( enumerate(pool.map(friedel_pairs.process_ypair, argslist, chunksize = OPTS.chunksize)),
                      total = len(argslist), desc = 'scan pairs completed'):
            
            outname = f'tmp/cfp_{str(i)}.h5'
            utils.colf_to_hdf(r, outname, save_mode='full')

            
    # Extract outputs and merge
    #################################################################
    outputs = []
    with suppress_stdout():
        for f in tqdm(os.listdir('tmp')):
            pks = ImageD11.columnfile.columnfile(f'tmp/{f}')
            outputs.append(pks)

    # merge outputs in a single peakfile
    cf_paired, labels, stats = friedel_pairs.merge_outputs(outputs, cf, doplot=False)
    cf_paired.parameters.loadparameters(args.parfile)
    
    # update geometry: tth correction + relocate Friedel pairs in sample coordinates
    print('Correct geometry using Friedel pairs\n====================')
    friedel_pairs.update_geometry_s3dxrd(cf_paired, ds, update_gvecs=True)
    
    # filter out dodgy pairs 
    print('Clean up paired peakfile\n====================')   
    m_tth = np.abs(cf_paired.tth[::2] - cf_paired.tth[1::2]) < OPTS.max_tth_dist
    m_I   = np.abs( np.log10(cf_paired.sum_intensity[::2] / cf_paired.sum_intensity[1::2]) ) < OPTS.max_logI_dist
    m_eta = np.abs(cf_paired.eta[::2]%360   - (180-cf_paired.eta[1::2])%360) < OPTS.max_eta_dist
    m_om  = np.abs(cf_paired.omega[::2]%360 - (180+cf_paired.omega[1::2])%360) < OPTS.max_omega_dist
    m_r   = cf_paired.r_dist[::2] <= cf_paired.dty.max()
    
    cleanpks = np.all([m_tth, m_I, m_eta, m_om, m_r], axis=0)
    cleanpks = utils.recast(cleanpks)
    print(f'Fraction of peaks retained: {np.count_nonzero(cleanpks)/cf_paired.nrows:.4f}')
    print(f'Fraction of total intensity retained: {np.sum(cf_paired.sum_intensity[cleanpks])/np.sum(cf_paired.sum_intensity):.4f}')      
    cf_paired.filter(cleanpks)  
    
    
    # Make plots and save
    #################################################################
    # tth histogram for corrected vs. uncorrected peaks
    fig = pl.figure(figsize=(10,5))
    fig.add_subplot(111)
    h,b,_ = utils.compute_tth_histogram(cf_paired, use_tthc = False, tthmin=cf.tth.min(), tthmax = cf.tth.max(), tthstep = 0.001,
                                    uself = True, doplot=False, density=True)
    hc,bc,_ = utils.compute_tth_histogram(cf_paired, use_tthc = True, tthmin=cf.tth.min(), tthmax = cf.tth.max(), tthstep = 0.001,
                                      uself = True, doplot=False, density=True)

    pl.plot(b,h,'-', lw=.6, label='non-corrected 2-theta')
    pl.plot(bc,hc,'-',lw=.6, label='corrected 2-theta')
    pl.legend()
    pl.xlim(cf.tth.min(), cf.tth.max())
    pl.xlabel('2-theta (deg)')
    pl.ylabel('pdf')
    fig.savefig(os.path.join(savedir, ds.dsname+'_tth_hist.pdf'), format='pdf')
    
    
    # sample reconstruction
    kw = {'cmap':'Greys_r','vmax':1.}

    fig, _ = utils.friedel_recon(cf_paired,
                    xbins = ds.ybinedges,
                    ybins = ds.ybinedges,
                    doplot=True,
                    mask = None,
                    weight_by_intensity=True,
                    norm = True,
                    **kw );

    fig.savefig(os.path.join(savedir, ds.dsname+'_fp_recon.png'), format='png')
    
    
    # save data. The 'minimal' save mode means that only the necessary data columns are kept.
    print('\nSave paired peaks...')
    utils.colf_to_hdf(cf_paired, os.path.join(savedir, os.path.basename(args.pksfile).replace('.h5','_p.h5')), save_mode='minimal')
    subprocess.run(f'rm -r tmp'.split(' '), check=True)  # delete tmp folder
    
    print('DONE\n==============================================\n')
    

#####################################################################
#####################################################################

    
parser = argparse.ArgumentParser(description='Friedel pairs matching')
parser.add_argument('-pksfile', help='absolute path to peakfile/peaks_table', required=True)
parser.add_argument('-dsfile', help='absolute path to datset file', required=True)
parser.add_argument('-parfile', help='absolute path to parameters file', required=True)
parser.add_argument('-splinefile', help='spline file for detector distortion correction (frelon)', required=False)
parser.add_argument('-dxfile', help='dx file for detectrot distortion correction (eiger)', required=False)
parser.add_argument('-dyfile', help='dy file for detectrot distortion correction (eiger)', required=False)
parser.add_argument('-use2Dpeaks', default=False, help='If loading from a peak table, do not merge peaks in omega but use raw 2D peaks', required=False)   
  
    
    
if __name__ == "__main__":
    main()
        
        
  