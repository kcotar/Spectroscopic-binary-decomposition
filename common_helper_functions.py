import numpy as np
import astropy.constants as const

# speed of light in km/s
c_val = const.c.value / 1000.


def _valid_orders_from_keys(key_list):
    """
    
    :param key_list: 
    :return: 
    """
    key_list = list(key_list)
    # remove keys whose type string
    return [eo for eo in key_list if not isinstance(eo, type(''))]


def correct_wvl_for_rv(wvl, rv):
    """

    :param wvl:
    :param rv:
    :return:
    """
    # perform Doppler shift of wavelength values using the supplied RV value
    # possitive RV -> shift tu bluer wavelengths
    # negative RV -> shift to redder wavelengths
    return wvl * (1. - rv / c_val)


def _spectra_resample(spectra, wvl_orig, wvl_target):
    """

    :param spectra:
    :param wvl_orig:
    :param wvl_target:
    :param k:
    :return:
    """
    idx_finite = np.isfinite(spectra)
    min_wvl_s = np.nanmin(wvl_orig[idx_finite])
    max_wvl_s = np.nanmax(wvl_orig[idx_finite])
    idx_target = np.logical_and(wvl_target >= min_wvl_s,
                                wvl_target <= max_wvl_s)
    new_flux = np.interp(wvl_target[idx_target], wvl_orig[idx_finite], spectra[idx_finite])
    nex_flux_out = np.ndarray(len(wvl_target))
    nex_flux_out.fill(np.nan)
    nex_flux_out[idx_target] = new_flux
    return nex_flux_out


def _combine_orders(exposure_data, target_wvl,
                    use_flx_key='flx', use_rv_key='RV_s1'):
    
    exposure_new_flx = np.full_like(target_wvl, fill_value=np.nan)

    if use_rv_key is None:
        # use a rv of 0 km/s, effectivelly not moving a spectrum
        rv_val = 0.
    else:
        if use_rv_key not in exposure_data.keys():
            # requested RV value was not determined for this exposure
            # TODO: repair return, might be dangerous in certain cases
            return exposure_new_flx
        rv_val = exposure_data[use_rv_key]   

    echelle_orders = _valid_orders_from_keys(exposure_data.keys()) 

    for echelle_order_key in echelle_orders:
        order_orig_flx = exposure_data[echelle_order_key][use_flx_key]
        order_orig_wvl = exposure_data[echelle_order_key]['wvl']
        # apply determined radial velocity to the wavelengths
        order_new_wvl = correct_wvl_for_rv(order_orig_wvl, rv_val)
        # resample wvl to the new supplied grid
        order_new_flx = _spectra_resample(order_orig_flx, order_new_wvl, target_wvl)
        # insert interpolated values into the final spectrum array
        idx_flx_order = np.isfinite(order_new_flx)
        exposure_new_flx[idx_flx_order] = order_new_flx[idx_flx_order]

    # return new filled array that spatially matches reference spectrum
    return exposure_new_flx