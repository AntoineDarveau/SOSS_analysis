#### To lower sampling of PHOENIX models ######

from convolution import get_c_matrix, WebbKer

path = "/Users/antoinedb/Models/PHOENIX_HiRes/"
file = "Z-0.0/lte02300-0.00-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
hdu = fits.open(path + file)
flux = hdu[0].data

file = "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
hdu = fits.open(path + file)
wv = hdu[0].data / 10000.  # Angstrom to microns

# Keep only relevant wv range
i_good = (0.5 < wv) & (wv < 3.0)
wv, flux = wv[i_good], flux[i_good]

# First convolution and resampling to reduce the length 
# Build estimate of the convolution kernel
wv_grid = grid_from_map(wv_2, P2, wv_range=[0.5, 3.0], n_os=20)
# Build convolution matrix
conv_ord2 = get_c_matrix(WebbKer(wv_grid[None, :]), wv[500000:500500], thresh=1e-5)

# Take the same convolution kernel 
# (approximation, but we are still at high oversampling)
kernel = conv_ord2[250,180:320].toarray().squeeze()
flux = np.convolve(kernel, flux, mode='same')
flux_fct = interp1d(wv, flux, kind='cubic', bounds_error=False)
# Resample
wv = oversample_grid(wv_grid, n_os=4)
flux= flux_fct(wv)

# Build accurate convolution matrix
wv_grid = grid_from_map(wv_2, P2, wv_range=[0.5, 3.0], n_os=15)
conv_ord2 = get_c_matrix(WebbKer(wv_grid[None, :]), wv, thresh=1e-5)

path = "/Users/antoinedb/Documents/Doctorat/SOSS/"
file = "Z-0.0-lte02300-0.00-0.0.PHOENIX-ACES-AGSS-COND-2011-n_os-15"
np.savez(path + file, wave=wv, flux=conv_ord2.dot(flux))
