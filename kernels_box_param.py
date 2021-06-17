from extract.overlap import TrpzOverlap, unsparse
from extract.utils import grid_from_map, oversample_grid
from extract.convolution import WebbKer
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits

width_list = [14, 15, 16]
fwhm_list = [1, 0.5]
n_os = 5


def test_ker_params(width_list, fwhm_list, n_os):

    # List of orders to consider in the extraction
    order_list = [1]

    path = "../jwst-mtl/SOSS/extract/Ref_files/"

    #### Wavelength solution ####
    wave_maps = []
    wave_maps.append(fits.getdata(path + "wavelengths_m1.fits"))

    #### Spatial profiles ####
    spat_pros = []
    spat_pros.append(fits.getdata(path + "spat_profile_m1.fits").squeeze())

    # Convert data from fits files to float (fits precision is 1e-8)
    wave_maps = [wv.astype('float64') for wv in wave_maps]
    spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

    # no tilt
    wave_maps = [np.tile(wv_map[50], (wv_map.shape[0], 1)) for wv_map in wave_maps]

    #### Throughputs ####
    def fct_ones(x): return 1/x
    thrpt_list = [fct_ones for order in order_list]

    ### Save original kernel
    webbker_file = "spectral_kernel_matrix_os_10_width_21pixels.fits"
    webbker_file = "../jwst-mtl/SOSS/extract/Ref_files/spectral_kernel_matrix/" + webbker_file
    webbker = fits.open(webbker_file)

    ### Set WebbKer class parameters to local test
    dummy_file = "spectral_kernel_matrix_test.fits"
    WebbKer.file_frame = dummy_file
    WebbKer.path = ''

    ### Init figure ###
    fig, ax = plt.subplots(len(width_list), 1, sharex=True,
                           figsize=(12, 3 * len(width_list)))
    try:
        ax[0]
    except:
        ax = [ax]

    for i_ax, width in enumerate(width_list):
        print(f"width {i_ax+1}/{len(ax)}")
        for fwhm in fwhm_list:

            # Generate new kernel file
            kernels = cut_ker_box(webbker[0].data[0], width=width, n_os=10, fwhm=fwhm)
            hdu = fits.PrimaryHDU(np.array([kernels, webbker[0].data[1]]),
                                  header=webbker[0].header)
            hdu.writeto(dummy_file, overwrite=True)

            #### Convolution kernels ####
            # wv_map_ker = wave_maps[0][50]#grid_from_map(wave_maps[0], spat_pros[0])
            # ker_list = [WebbKer(wv_map_ker[None, :])]
            ker_list = [WebbKer(wave_maps[0])]
            # ker_list = [GaussKer(np.linspace(0.5, 3.0, 10000), res=800) for wv_map in wave_maps]

            # Put all inputs from reference files in a list
            ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

            def flux_fct(wv): return 1e5 - 1e4 * wv

            # Grid for simulation
            lam_simu = grid_from_map(wave_maps[0][50:51], spat_pros[0][50:51], n_os=10, wv_range=[0.8, 3.0])
            # lam_simu = grid_from_map(wave_maps[0], spat_pros[0], n_os=10, wv_range=[0.8, 3.0])

            # Init simu
            simu = TrpzOverlap(*ref_files_args, lam_grid=lam_simu, thresh=1e-8,
                               orders=[1], c_kwargs={'n_out':[5*10, 8*10], 'length':21*10+1})

            f_c_th = simu.c_list[0].dot(flux_fct(simu.lam_grid))
            wv_th = simu.lam_grid_c(0)
            fct_f_c_th = interp1d(wv_th, f_c_th, bounds_error=False, fill_value=np.nan, kind='cubic')

            scidata = simu.rebuild(flux_fct, orders=[0])


            # Grid
            lam_simu = grid_from_map(wave_maps[0][50:51], spat_pros[0][50:51], n_os=n_os, wv_range=[0.8, 3.0])

            # Init simu
            length_ker = 21*n_os + ((21*n_os)%2 == 0)
            simu = TrpzOverlap(*ref_files_args, lam_grid=lam_simu, thresh=1e-8,
                               orders=[1], lam_bounds=[[0.88, 2.8]],
                               c_kwargs={'thresh_out': 1e-12, 'length':length_ker})

            f_c = simu.c_list[0].dot(flux_fct(lam_simu))


            ax[i_ax].plot(simu.lam_grid_c(0),
                          (f_c - fct_f_c_th(simu.lam_grid_c(0)))/f_c,
                          label=fwhm)

        ax[i_ax].set_title(f"Oversampling: {n_os}, box width: {width}")
        y_lim = ax[i_ax].get_ylim()
        ax[i_ax].vlines(ker_list[0].wv_center, *y_lim, alpha=0.2, linestyle="--")
        ax[i_ax].set_ylim(*y_lim)
        ax[i_ax].set_ylabel("convolution rel error (f_c - f_c_th)")
        ax[i_ax].legend(title="FWHM")

    ax[-1].set_xlabel("Wavelength [um]")
    plt.tight_layout()
#     fig.savefig(f"Convolution_problem/conv_error_n_os_{n_os}_webb_ker_negative.png")
    plt.show()


def cut_ker_box(kernels, width=10, n_os=10, fwhm=5):
    """
    Cut kernel with a gaussian smoothed box.
    width and fhwm in pixels
    """
    kernels = kernels.copy()
    ker_width, n_ker = kernels.shape

    # Define box around kernel center
    width = width * n_os
    ker_hwidth = ker_width // 2
    pixel_os = np.arange(-ker_hwidth, ker_hwidth + 1)
    box = np.abs(pixel_os) <= (width / 2)

    # Define gaussian kernel to smooth the box
    x0 = 0.0
    sigma = fwhm2sigma(fwhm * n_os)
    g_ker = gaussians(pixel_os, x0, sigma)
    g_ker = g_ker / np.sum(g_ker)

    # Convolve
    box = np.convolve(box, g_ker, mode='same')
    box = box / box.max()

    # Apply to kernels
    kernels *= box[:, None]
    # Re-norm
    kernels /= kernels.sum(axis=0)

    return kernels


def gaussians(x, x0, sig, amp=None):
    """
    Gaussian function
    """

    # Amplitude term
    if amp is None:
        amp = 1/np.sqrt(2 * np.pi * sig**2)

    return amp * np.exp(-0.5*((x - x0) / sig)**2)


def fwhm2sigma(fwhm):
    """
    Convert a full width half max to a standard deviation, assuming a gaussian
    """
    return fwhm / np.sqrt(8 * np.log(2))


def cut_ker_mins(kernels, n_mins):
    """
    Cut the kernels at the nth minimum around the center.
    """
    kernels = kernels.copy()
    d_ker = np.diff(kernels, axis=0)
    i_switch = np.diff((d_ker >= 0), axis=0)
    i_switch = [np.where(x)[0] for x in i_switch.T]

    d2_ker = np.diff(d_ker, axis=0)

    i_min_list, i_max = [], []

    for i_sw, d2 in zip(i_switch, d2_ker.T):
        bool_index = (d2[i_sw] < 0)
        i_max.append(i_sw[bool_index])

        bool_index = (d2[i_sw] > 0)
        i_min_list.append(i_sw[bool_index])

    central_peaks = np.argmax(kernels, axis=0)

    for i_ker, i_min in enumerate(i_min_list):
        # Cut left wing
        bool_index = (i_min - central_peaks[i_ker]) < 0
        left_mins = i_min[bool_index]
        i_cut = left_mins[-n_mins[0]] + 1
        kernels[:i_cut, i_ker] = 0.0
        # Cut right wing
        bool_index = (i_min - central_peaks[i_ker]) > 0
        right_mins = i_min[bool_index]
        i_cut = right_mins[n_mins[1]] + 1
        kernels[i_cut:, i_ker] = 0

    return kernels

if __name__ == "__main__":
    test_ker_params(width_list, fwhm_list, n_os)
