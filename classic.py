import numpy as np

from extract.utils import get_lam_p_or_m, grid_from_map


def get_trace_center(aperture):
    """
    *** To use when the trace centroid is not available ***
    Return the center of the trace given an aperture estimate.
    The aperture estimate could use the data directly.
    Parameter
    ---------
    aperture: 2d array
        estimate of the aperture.
    Output
    ------
    (columns, aperture center)
    """
    # Valid columns
    col = np.where((aperture > 0.).any(axis=0))[0]
    
    # Get rows position for valid columns
    rows = np.indices(aperture.shape)[0][:, col]
    
    # Convert aperture to weights for valid columns
    weights = aperture[:, col]
    weights /= weights.sum(axis=0)
    
    # Compute center of mass to find the center
    center = (rows*weights).sum(axis=0)
    
    return col, center

def get_box_weights(cols, centroid, n_pix, shape):
    """
    Return the weights of a box aperture given the centroid
    and the width of the box in pixels.
    All pixels will have the same weights except at the
    ends of the box aperture.
    Parameters
    ----------
    cols: 1d array, integer
        Columns index positions. Useful if the centroid is defined for 
        specific columns or a subrange of columns.
    centroid: 1d array
        Position of the centroid (in rows). Same shape as `cols`
    n_pix: float
        full width of the extraction box in pixels.
    shape: 2 integers tuple
        shape of the output image. (n_row, n_column)
    Ouput
    -----
    2d image of the box weights
    """
    # Row centers of all pixels
    rows = np.indices((shape[0], len(cols)))[0]
    
    # Pixels that are entierly inside the box are set to one
    cond = (rows <= (centroid - 0.5 + n_pix/2))
    cond &= ((centroid + 0.5 - n_pix/2) <= rows)
    weights = cond.astype(float)
    
    # Upper bound
    cond = (centroid - 0.5 + n_pix/2) < rows
    cond &= (rows < (centroid + 0.5 + n_pix/2))
    weights[cond] = (centroid + n_pix/2 - (rows - 0.5))[cond]
    
    # Lower bound
    cond = (rows < (centroid + 0.5 - n_pix/2))
    cond &= ((centroid - 0.5 - n_pix/2) < rows)
    weights[cond] = (rows + 0.5 - (centroid - n_pix/2))[cond]
    
    # Return with the specified shape
    # with zeros where the box is not define
    out = np.zeros(shape, dtype=float)
    out[:, cols] = weights
    
    return out


def box_extract(data, box_weights, lam_col=None, cols=None, mask=None):
    '''
    Make a box extraction
    Parameters
    ----------
    data: 2d array of shape (n_row, n_columns)
        scidata
    box_weights: 2d array, same shape as data
        pre-computed weights for box extraction.
    lam_col: 1d array of shape (n_columns)
        wavelength associated with each columns. If not given,
        the column position is taken as ordinates.
    cols: numpy valid index
        Which columns to extract
    mask: 2d array, boolean, same shape as data
        masked pixels
    Output
    ------
    (wavelengths or column position, spectrum)
    '''
    # Use all columns if not specified
    if cols is None:
        cols = slice(None)

    # Define mask if not given
    if mask is None:
        # False everywhere
        mask = np.zeros(data.shape, dtype=bool)

    # Use pixel position if no wavelenghts are given
    if lam_col is None:
        lam_col = np.arange(data.shape[1])

    # Make a copy of arrays with only needed columns
    # so it is not modified outside of the function
    data = data[:, cols].copy()
    box_weights = box_weights[:, cols].copy()
    mask = mask[:, cols].copy()

    # Initialize the output with nans
    out = np.ones_like(lam_col) * np.nan

    # Mask potential nans in data
    mask_nan = np.isnan(data)

    # Combine with user specified mask
    mask = (mask_nan | mask)

    # Apply to weights
    box_weights[mask_nan] = np.nan

    # Normalize only valid columns
    out = np.nansum(box_weights*data, axis=0)    
    
    # Return sorted with the associated wavelength
    idx_sort = np.argsort(lam_col)
    out = (lam_col[idx_sort], out[idx_sort])

    return out


class OptimalExtract:

    def __init__(self, scidata, t_ord, p_ord, lam_ord, lam_grid=None,
                 lam_bounds=None, sig=None, mask=None, thresh=1e-5):

        # Use `lam_grid` at the center of the trace if not specified
        if lam_grid is None:
            lam_grid, lam_col = grid_from_map(lam_ord, p_ord, out_col=True)
        else:
            lam_col = slice(None)

        # Save wavelength grid
        self.lam_grid = lam_grid.copy()
        self.lam_col = lam_col

        # Compute delta lambda for the grid
        self.d_lam = -np.diff(get_lam_p_or_m(lam_grid), axis=0)[0]

        # Basic parameters to save
        self.N_k = len(lam_grid)
        self.thresh = thresh

        if sig is None:
            self.sig = np.ones_like(scidata)
        else:
            self.sig = sig.copy()

        # Save PSF
        self.p_ord = p_ord.copy()

        # Save pixel wavelength
        self.lam_ord = lam_ord.copy()

        # Throughput
        # Can be a callable (function) or an array
        # with the same length as lambda grid.
        try:  # First assume it's a function
            self.t_ord = t_ord(self.lam_grid)  # Project on grid
        except TypeError:  # Assume it's an array
            self.t_ord = t_ord.copy()

        # Build global mask
        self.mask = self._get_mask(mask)

        # Assign other trivial attributes
        self.data = scidata.copy()
        # TODO: try setting to np.nan instead?
        self.data[self.mask] = 0

    def _get_mask(self, mask):

        # Get needed attributes
        thresh = self.thresh
        p_ord = self.p_ord
        lam = self.lam_ord
        grid = self.lam_grid

        # Mask according to the global troughput (spectral and spatial)
        mask_p = (p_ord < thresh)

        # Mask pixels not covered by the wavelength grid
        lam_min, lam_max = grid.min(), grid.max()
        mask_lam = (lam <= lam_min) | (lam >= lam_max)

        # Combine all masks including user's defined mask
        if mask is None:
            mask = np.any([mask_p, mask_lam], axis=0)
        else:
            mask = np.any([mask_p, mask_lam, mask], axis=0)

        return mask

    def extract(self):

        # Get needed attributes
        psf, sig, data, ma, lam, grid, lam_col =  \
            self.getattrs('p_ord', 'sig', 'data',
                          'mask', 'lam_ord', 'lam_grid', 'lam_col')

        # Define delta lambda for each pixels
        d_lam = -np.diff(get_lam_p_or_m(lam), axis=0)[0]

        # Optimal extraction (weighted sum over columns)
        # ------------------
        # Define masked array (easier to sum with the mask)
        # First, compute normalisation factor at each columns
        norm = np.ma.array(psf**2/sig**2, mask=ma).sum(axis=0)
        # Second, define numerator
        num = np.ma.array(psf*data/sig**2 / (d_lam), mask=ma)
        # Finally compute flux at each columns
        out = (num / norm).sum(axis=0)

        # Return flux where lam_grid is defined
        out = out[lam_col]
        i_good = ~(ma[:, lam_col]).all(axis=0)
        out = out[i_good].data

        # Return sorted according to lam_grid
        i_sort = np.argsort(grid[i_good])
        return grid[i_sort], out[i_sort]

    def f_th_to_pixel(self, f_th):

        # Get needed attributes
        grid, lam, ma =  \
            self.getattrs('lam_grid', 'lam_ord', 'mask')

        # Project f_th on the detector valid pixels
        lam_pix = lam[~ma]
        f_k = f_th(lam_pix)

        # Compute extremities of the bins
        # (Assuming grid is the center)
        lam_p, lam_m = get_lam_p_or_m(grid)

        # Make sur it's sorted
        lam_p, lam_m = np.sort(lam_p), np.sort(lam_m)

        # Special treatment at the end of the bins
        lam_bin = np.concatenate([lam_m, lam_p[-1:]])

        # Compute bins
        f_out = np.histogram(lam_pix, lam_bin, weights=f_k)[0]
        # Normalise (result is the mean in each bins)
        f_out /= np.histogram(lam_pix, lam_bin)[0]

        return f_out

    def getattrs(self, *args):

        out = [getattr(self, arg) for arg in args]

        if len(out) > 1:
            return out
        else:
            return out[0]
