## Imports from standard packages
import numpy as np
from scipy.interpolate import interp1d
from astropy.io import fits

## Local imports
from extract.overlap import TrpzOverlap
from extract.utils import get_soss_grid, grid_from_map, oversample_grid
from extract.convolution import WebbKer

# Plots imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm #for better display of FITS images

plt.rc('figure', figsize=(9,3))
plt.rcParams["image.cmap"] = "inferno"

# #############
# Read ref files

# List of orders to consider in the extraction
order_list = [1, 2]

path = "../jwst-mtl/SOSS/extract/Ref_files/"

#### Wavelength solution ####
wave_maps = []
wave_maps.append(fits.getdata(path + "wavelengths_m1.fits"))
wave_maps.append(fits.getdata(path + "wavelengths_m2.fits"))

#### Spatial profiles ####
spat_pros = []
spat_pros.append(fits.getdata(path + "spat_profile_m1.fits").squeeze())
spat_pros.append(fits.getdata(path + "spat_profile_m2.fits").squeeze())

# Convert data from fits files to float (fits precision is 1e-8)
wave_maps = [wv.astype('float64') for wv in wave_maps]
spat_pros = [p_ord.astype('float64') for p_ord in spat_pros]

#### Throughputs ####
def fct_ones(x): return np.ones_like(x)
thrpt_list = [fct_ones for order in order_list]

#### Convolution kernels ####
ker_list = [WebbKer(wv_map) for wv_map in wave_maps]

# Put all inputs from reference files in a list
ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

# ########
# Setup for simulation

## Wavelength grid
lam_simu = get_soss_grid(spat_pros, wave_maps, n_os=15)

# Initiate simulation

# Choose a small threshold for the spatial profile cut
# (less than for a normal extraction)
simu = TrpzOverlap(*ref_files_args, lam_grid=lam_simu, thresh=1e-8, c_kwargs={'thresh':1e-8})


# ##############
# Test on linear function
def flux_fct(wv): return 1e5*(5/wv - 1)

# Inject
scidata = simu.rebuild(flux_fct)

# Extraction
sig = np.sqrt(np.abs(scidata) + 20**2)
extract = TrpzOverlap(*ref_files_args, sig=sig, data=scidata, n_os=2, thresh=1e-4)

f_k = extract.extract()
rebuilt = extract.rebuild(f_k)

plt.figure()
plt.imshow((rebuilt-scidata)/scidata)
plt.colorbar()
plt.show()

for i_ord, order in enumerate(order_list):
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].set_title(f"Order {order}")
    grid_bin = wave_maps[i_ord][100,:]
    grid_bin = grid_bin[grid_bin > 0]
    _, f_bin = extract.bin_to_pixel(f_k=f_k,
                                    grid_pix=grid_bin,
                                    i_ord=i_ord)
    _, f_bin_th = simu.bin_to_pixel(f_k=flux_fct(lam_simu), grid_pix=grid_bin, i_ord=i_ord)

    ax[0].plot(grid_bin, f_bin)
    ax[0].plot(grid_bin, f_bin_th, ":")

    # f_th_interp = interp1d(simu['grid_c2'], simu['f_c2'], kind='cubic', bounds_error=False, fill_value=np.nan)
    ax[1].plot(grid_bin, (f_bin-f_bin_th)/f_bin_th*1e6)
    plt.show()
