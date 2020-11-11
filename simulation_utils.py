import numpy as np
from astropy.io import fits
from extract.custom_numpy import is_sorted


def gaussian(x, x0=0, sig=1, amp=None):
    
    # Amplitude term
    if amp is None:
        amp = 1/np.sqrt(2 * np.pi * sig**2)
    
    return amp * np.exp(-0.5*((x - x0) / sig)**2)


def gauss_ker(x=None, mean=None, sigma=None, FWHM=None, nFWHM=7, oversample=None):
    
    if mean is None:
        mean = np.array([0])
    
    if sigma is None:
        sigma = np.array([1])
        
    if oversample is None:
        oversample = 1
    
    if x is None:
        if FWHM is None:
            FWHM = sigma2fwhm(sigma)
        # Length of the kernel is 2 x nFWHM times FWHM
        x = np.linspace(0, nFWHM*FWHM, int(nFWHM*FWHM*oversample + 1))
        x = np.concatenate([mean - x, mean + x])
        x = np.unique(x)
       
    if FWHM is not None:
        # Convert to sigma
        sigma = fwhm2sigma(FWHM)  # FWHM / (2 * np.sqrt(2 * np.log(2)))
    
    #
    # Compute gaussian ----------
    #
    # Exponential term
    G = np.exp(-0.5 * ((x - mean[:,None]) / sigma[:,None])**2)
    # Amplitude term
    G /= np.sqrt(2 * np.pi * sigma[:,None]**2)

    # Normalization
    G /= G.sum(axis=-1)[:,None]  

    return G.squeeze()


def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))

def sigma2fwhm(sigma):
    return sigma * np.sqrt(8 * np.log(2))

def get_d_lam_poly(wv, sampling=2, order=5):
    """
    Returns the function of the width of the resolution
    kernel (delta lambda) in the wavelength space.
    The function is a polynomial fit.
    """
    
    # Use only valid columns
    wv = wv[wv > 0]
    
    # Get d_lam of the convolution kernel assuming sampling
    d_lam = np.abs(sampling*np.diff(wv))
    
    # Fit polynomial
    coeffs = np.polyfit(wv[1:], d_lam, order)
    
    # Return as a callable
    return np.poly1d(coeffs)

def add_noise(data, bkgrd=20):
    
    data_noisy = data.copy()

    # Add Poisson noise
    data_noisy[np.isnan(data_noisy)] = 0  # Nans to zero
    data_noisy = np.random.poisson(data_noisy)    

    # Add background noise
    data_bkgrd = np.random.normal(scale=bkgrd, size=data_noisy.shape)
    data_noisy = data_noisy + data_bkgrd

    return data_noisy

def load_simu(filename, order=None, noisy=True):
    
    hdu = fits.open(filename)
    out = {'grid': hdu['FLUX'].data['lam_grid'],
           'f_k': hdu['FLUX'].data['f_lam'],
           'grid_c1': hdu['FLUX_C1'].data['lam_grid'],
           'f_c1': hdu['FLUX_C1'].data['f_lam'],
           'grid_c2': hdu['FLUX_C2'].data['lam_grid'],
           'f_c2': hdu['FLUX_C2'].data['f_lam']}
    
    if order is None:
        key = "FULL"
    else:
        key = f"ORD {order}"
        
    if noisy:
        key += " NOISY"
        
    out['data'] = hdu[key].data
    
    return out

############################

from extract.convolution import fwhm2sigma, gaussians, get_c_matrix

LIGHT_SPEED = 2.9979246e8  # m / s

class KerVsini:
    
    def __init__(self, vsini, wv_grid, n_os=2):
        """
        vsini: m/s
        """
    
        # Determine the resolution based on the sampling
        d_grid = np.diff(wv_grid)
        res = wv_grid[:-1]/ d_grid / n_os
        
        # Get fwhm given the vsini
        fwhm = res * vsini / LIGHT_SPEED * wv_grid[:-1]
        
        # What we really want is sigma, not FWHM
        sig = fwhm2sigma(fwhm)
        
        # interpolate fwhm as a function of wavelength
        fct_sig = interp1d(wv_grid[:-1], sig, bounds_error=False,
                            fill_value='extrapolate')
        
        self.fct_sig = fct_sig
        
    def __call__(self, x, x0):
        """
        Parameters
        ----------
        x: 1d array
            position where the kernel is evaluated
        x0: 1d array (same shape as x)
            position of the kernel center for each x.

        Returns
        -------
        Value of the gaussian kernel for each sets of (x, x0)
        """

        # Get the sigma of each gaussians
        sig = self.fct_sig(x0)

        return gaussians(x, x0, sig)
        

