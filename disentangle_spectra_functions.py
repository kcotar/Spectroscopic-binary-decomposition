import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from os import scandir, path
from rv_helper_functions import correct_wvl_for_rv

norm_suffix = '_normalised.txt'
sigma_norm_suffix = '_sigma_normalised.txt'


def get_reduced_exposures(in_dir):
    """
    
    :param in_dir: 
    :return: 
    """
    return [f.name for f in scandir(in_dir) if f.is_dir()]


def get_normalised_orders(spec_dir, spectrum):
    """
    
    :param spec_dir: 
    :param spectrum: 
    :return: 
    """
    existing_orders = []
    for i_o in range(35):
        spec_path = spec_dir + spectrum + '/' + spectrum + '_' + str(i_o)
        if path.isfile(spec_path + norm_suffix) and path.isfile(spec_path + norm_suffix):
            existing_orders.append(spectrum + '_' + str(i_o))
    return existing_orders


def get_orderdata_by_wavelength(spec_dir, orders, in_wvl):
    """
    
    :param spec_dir: 
    :param orders: 
    :param in_wvl: 
    :return: 
    """
    for order in orders:
        flux_data = np.loadtxt(spec_dir + order + norm_suffix)
        sigma_data = np.loadtxt(spec_dir + order + norm_suffix)
        if np.nanmin(flux_data[:, 0]) <= in_wvl <= np.nanmax(flux_data[:, 0]):
            return flux_data[:, 0], flux_data[:, 1], sigma_data[:, 1]
    return None, None, None


def get_spectral_data(star, wvl_orders, in_dir):
    """
    
    :param star: 
    :param wvl_orders: 
    :param in_dir: 
    :return: 
    """
    input_dir = in_dir + star + '/spec/'
    list_exposures = get_reduced_exposures(input_dir)
    # create a dictionary of all exposures with their belonging data
    star_data_all = {}
    for exposure in list_exposures:
        if 'joined' in exposure:
            # skipp exposures/spectra that are created by joining multiple exposures
            continue
        # get all possible orders
        print('Exploring exposures of:', exposure)
        all_norm_orders = get_normalised_orders(input_dir, exposure)
        if len(all_norm_orders) > 0:
            # create new dictionary that will hold the data of selected order for a given exposure
            star_data_all[exposure] = {}
            for get_wvl_order in wvl_orders:
                order_data = get_orderdata_by_wavelength(input_dir + exposure + '/',
                                                         all_norm_orders, get_wvl_order)
                if order_data[0] is not None:
                    star_data_all[exposure][get_wvl_order] = {
                        'wvl': order_data[0],
                        'flx': order_data[1],
                        'sig': order_data[2]
                    }
    return star_data_all


def spectra_resample(spectra, wvl_orig, wvl_target):
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


def create_new_reference(exposures_all, target_wvl,
                         use_flx_key='flx', use_rv_key='RV_s1',
                         plot_combined=False, plot_path='plot_combined.png'):
    flx_new = list([])
    for exposure_id in exposures_all.keys():
        exposure_data = exposures_all[exposure_id]

        if use_rv_key not in exposure_data.keys():
            # requested RV value was not determined for this exposure
            continue
        rv_val = exposure_data[use_rv_key]

        echelle_orders = list(exposure_data.keys())
        # remove keys whose type string
        echelle_orders = [eo for eo in echelle_orders if not isinstance(eo, type(''))]
        
        exposure_new_flx = np.full_like(target_wvl, fill_value=np.nan)
        for echelle_order_key in echelle_orders:
            order_orig_flx = exposure_data[echelle_order_key][use_flx_key]
            order_orig_wvl = exposure_data[echelle_order_key]['wvl']
            # apply determined radial velocity to the wavelengths
            order_new_wvl = correct_wvl_for_rv(order_orig_wvl, rv_val)
            # resample wvl to the new supplied grid
            order_new_flx = spectra_resample(order_orig_flx, order_new_wvl, target_wvl)
            # insert interpolated values into the final spectrum array
            idx_flx_order = np.isfinite(order_new_flx)
            exposure_new_flx[idx_flx_order] = order_new_flx[idx_flx_order]
            
        # combine all resampled and moved spectra
        flx_new.append(exposure_new_flx)

    # compute median of all considered exposures
    flx_new = np.array(flx_new)
    flx_new_median = np.nanmedian(flx_new, axis=0)
    idx_median = np.isfinite(flx_new_median)
    wvl_range = (np.min(target_wvl[idx_median]) - 2.,
                 np.max(target_wvl[idx_median]) + 2.)

    if plot_combined:
        fig, ax = plt.subplots(1, 1, figsize=(80, 3.))
        for i_ex in range(flx_new.shape[0]):
            ax.plot(target_wvl, flx_new[i_ex, :], lw=0.5, alpha=0.33)
        ax.plot(target_wvl, flx_new_median, c='black', lw=0.8)
        ax.set(xlim=wvl_range, ylim=np.nanpercentile(flx_new_median, [0.4, 99.6]),
               xlabel='Wavelength', ylabel='Normalized flux')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

    # return rv corrected and computed median combination of individual exposures
    return flx_new_median
