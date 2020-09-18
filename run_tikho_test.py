import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits
from scipy.sparse import identity
import matplotlib.pyplot as plt
# Local imports
from extract.overlap import TrpzOverlap
from extract.utils import get_soss_grid, grid_from_map, oversample_grid
from extract.convolution import WebbKer, get_c_matrix


DEFAULT_FILE_ROOT = 'tikho_test'
DEFAULT_FILE_EXT = '.n_os_{}.c_thresh_{:1.0e}.tikho_os_{}'
DEFAULT_PATH = '../jwst-mtl/SOSS/extract/'
DEFAULT_REF_FILES = {'wv_1': 'Ref_files/wavelengths_m1.fits',
                     'wv_2': 'Ref_files/wavelengths_m2.fits',
                     'P_1': 'Ref_files/spat_profile_m1.fits',
                     'P_2': 'Ref_files/spat_profile_m2.fits'}


def run_tikho_tests(p_list, lam_list, scidata, f_th_c,
                    n_os_list, c_thresh_list, t_mat_n_os_list,
                    factors=None, file_root=None, file_ext=None, path=None):

    # Unpack some lists
    P1, P2 = p_list
    wv_1, wv_2 = lam_list

    # Default kwargs
    if factors is None:
        factors = 10.**(-1*np.arange(10, 25, 0.3))

    if file_root is None:
        file_root = 'tikho_test'

    if file_ext is None:
        file_ext = '.n_os_{}.c_thresh_{:1.0e}.tikho_os_{}'

    if path is None:
        path = ''

    # Message to print
    status = 'n_os={}, c_thresh={:1.0e}, t_mat_n_os={}'

    # Iterate on grid oversampling
    for n_os in n_os_list:
        # Generate grid
        lam_grid = get_soss_grid([P1, P2], [wv_1, wv_2], n_os=n_os)

        # Iterate on convolution kernel wings threshold
        for c_thresh in c_thresh_list:
            # Init extraction object
            extra = TrpzOverlap([P1, P2], [wv_1, wv_2], scidata=scidata,
                                lam_grid=lam_grid, thresh=1e-5,
                                c_kwargs={'thresh': c_thresh})
            # Project injected spectrum on grid
            f_k_th = {'f_k_th_1': f_th_c[0](extra.lam_grid_c(0)),
                      'f_k_th_2': f_th_c[1](extra.lam_grid_c(1))}

            # Save values that do not need to be recomputed
            wv_range = [extra.lam_grid.min(), extra.lam_grid.max()]
            # Iterate on resolution of the tikhonov matrix
            for t_mat_n_os in t_mat_n_os_list:
                # Print status
                print(status.format(n_os, c_thresh, t_mat_n_os))

                # Generate a fake wv_map to cover all wv_range with a
                # resolution `t_mat_n_os` times the resolution of order 2.
                wv_map = grid_from_map(wv_2, P2, wv_range=wv_range)
                wv_map = oversample_grid(wv_map, n_os=t_mat_n_os)
                # Build convolution matrix
                conv_ord2 = get_c_matrix(WebbKer(wv_map[None, :]),
                                         extra.lam_grid, thresh=1e-5)
                # Build tikhonov matrix
                t_mat = conv_ord2 - identity(conv_ord2.shape[0])

                # Test factors
                test_conv = extra.get_tikho_tests(factors, t_mat=t_mat)

                # Save results
                file_name = path + file_root
                file_name += file_ext.format(n_os, c_thresh, t_mat_n_os)
                to_save = {**test_conv, **f_k_th, 'grid': extra.lam_grid}
                np.savez(file_name, **to_save)

if __name__ == '__main__':

    # Read relevant files
    wv_1 = fits.open(DEFAULT_PATH + DEFAULT_REF_FILES["wv_1"])[0].data
    wv_2 = fits.open(DEFAULT_PATH + DEFAULT_REF_FILES["wv_2"])[0].data
    P1 = fits.open(DEFAULT_PATH + DEFAULT_REF_FILES["P_1"])[0].data.squeeze()
    P2 = fits.open(DEFAULT_PATH + DEFAULT_REF_FILES["P_2"])[0].data.squeeze()

    # Convert to float (fits precision is 1e-8)
    wv_1 = wv_1.astype(float)
    wv_2 = wv_2.astype(float)
    P1 = P1.astype(float)
    P2 = P2.astype(float)

    # Load simulation
    simu = load_simu('../Simulations/phoenix_teff_02300_scale_1.0e-05.fits')

    # Unpack
    scidata = simu['data']

    # Compute injected convolved flux for each orders
    f_th_c = [interp1d(simu['grid_c1'], simu['f_c1'],
                       kind='cubic', fill_value="extrapolate")]
    f_th_c.append(interp1d(simu['grid_c2'], simu['f_c2'],
                           kind='cubic', fill_value="extrapolate"))

    # Run tests
    file_root = 'phoenix_teff_02300_scale_1.0e-05'
    file_ext = '.n_os_{}.c_thresh_{:1.0e}.tikho_os_{}'
    path = ''

    # Parameters to test
    n_os_list = [1]
    c_thresh_list = [1e-3]
    t_mat_n_os_list = [1]

    run_tikho_tests([P1, P2], [wv_1, wv_2], scidata, f_th_c,
                    n_os_list, c_thresh_list, t_mat_n_os_list,
                    file_root=file_root, file_ext=file_ext, path=path)

    print('Done')
