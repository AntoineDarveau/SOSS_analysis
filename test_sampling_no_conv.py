from extract.overlap import TrpzOverlap
from extract.utils import grid_from_map
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


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

#### Throughputs ####
def fct_ones(x): return 1/x
thrpt_list = [fct_ones for order in order_list]

#### Convolution kernels ####
ker_list = [np.array([0,0,1,0,0]) for wv_map in wave_maps]

# Put all inputs from reference files in a list
ref_files_args = [spat_pros, wave_maps, thrpt_list, ker_list]

def flux_fct(wv): return 1e5 * wv

# Grid
lam_simu = grid_from_map(wave_maps[0], spat_pros[0], n_os=15)

# Init simu
simu = TrpzOverlap(*ref_files_args, lam_grid=lam_simu, thresh=1e-8, orders=[1])

scidata = simu.rebuild(flux_fct, orders=[0])

fig, ax = plt.subplots(4, 2, figsize=(6, 10))

n_os_list = [1, 2, 3, 5, 8, 11, 14, 15]

ax = ax.ravel()
for i_os, oversample in enumerate(n_os_list):
    #
    ### Simulation ###
    #

    # Grid
    lam_simu = grid_from_map(wave_maps[0], spat_pros[0], n_os=oversample)
    
    # Init simu
    simu = TrpzOverlap(*ref_files_args, lam_grid=lam_simu, thresh=1e-8, orders=[1])
    
    rebuilt = simu.rebuild(flux_fct, orders=[0])
    
    err = ((rebuilt-scidata)/scidata)
    err = err[np.isfinite(err)]

    ax[i_os].hist(err, bins=50)
    ax[i_os].legend([f"Oversampling: {oversample}"])

plt.tight_layout()
plt.show()
