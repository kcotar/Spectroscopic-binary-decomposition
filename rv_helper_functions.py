import numpy as np

from astropy.io import fits


def get_RV_ref_spectrum(spectrum_full_path):
    ref_data = fits.open(spectrum_full_path)
    ref_flx_orig = ref_data[0].data
    ref_wvl_orig = ref_data[0].header['CRVAL1'] + np.arange(len(ref_flx_orig)) * ref_data[0].header['CDELT1']
    # resample data to a finer resolution
    ref_wvl_d = ref_data[0].header['CDELT1']/5.
    ref_wvl = np.arange(ref_wvl_orig[0], ref_wvl_orig[-1], ref_wvl_d)
    ref_flx = np.interp(ref_wvl, ref_wvl_orig, ref_flx_orig)
    # output processed spectrum
    return ref_flx, ref_wvl
