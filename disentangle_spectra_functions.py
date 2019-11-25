import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from os import scandir, path
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter, argrelextrema
from common_helper_functions import _valid_orders_from_keys, correct_wvl_for_rv, _combine_orders, _spectra_resample

norm_suffix = '_normalised.txt'
sigma_norm_suffix = '_sigma_normalised.txt'


def _get_reduced_exposures(in_dir):
    """
    
    :param in_dir: 
    :return: 
    """
    return [f.name for f in scandir(in_dir) if f.is_dir()]


def _get_normalised_orders(spec_dir, spectrum):
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


def get_spectral_data(star, wvl_orders, in_dir,
                      new_only=False):
    """
    
    :param star: 
    :param wvl_orders: 
    :param in_dir:
    :param new_only:
    :return: 
    """
    input_dir = in_dir + star + '/spec/'
    list_exposures = _get_reduced_exposures(input_dir)
    # create a dictionary of all exposures with their belonging data
    star_data_all = {}
    for exposure in list_exposures:
        if 'joined' in exposure:
            # skipp exposures/spectra that are created by joining multiple exposures
            continue
        if new_only:
            if 'ec.vh' not in exposure:
                # skipp older exposures that might be of worse quality
                continue
        # get all possible orders
        print('Exploring exposures of:', exposure)
        all_norm_orders = _get_normalised_orders(input_dir, exposure)
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


def create_new_reference(exposures_all, target_wvl,
                         use_flx_key='flx', use_rv_key='RV_s1',
                         plot_combined=False, plot_path='plot_combined.png',
                         plot_shifted=False):
    flx_new = list([])
    for exposure_id in exposures_all.keys():
        exposure_data = exposures_all[exposure_id]    
        
        # combine all resampled and RV moved spectra
        exposure_new_flx = _combine_orders(exposure_data, target_wvl,
                                           use_flx_key=use_flx_key, use_rv_key=use_rv_key)            
        flx_new.append(exposure_new_flx)

    # compute median of all considered exposures
    flx_new = np.array(flx_new)
    flx_new_median = np.nanmedian(flx_new, axis=0)
    idx_median = np.isfinite(flx_new_median)
    wvl_range = (np.min(target_wvl[idx_median]) - 2.,
                 np.max(target_wvl[idx_median]) + 2.)

    n_spectra = flx_new.shape[0]
    x_ticks = range(4500, 7000, 20)
    x_ticks_str = [str(xt) for xt in x_ticks]
    # plot combined spectra - all around normalized level of 1
    if plot_combined:
        fig, ax = plt.subplots(1, 1, figsize=(135, 3.))
        for i_ex in range(n_spectra):
            ax.plot(target_wvl, flx_new[i_ex, :], lw=0.5, alpha=0.33)
        ax.plot(target_wvl, flx_new_median, c='black', lw=0.8)
        ax.set(xlim=wvl_range, ylim=np.nanpercentile(flx_new_median, [0.4, 99.6]),
               xlabel='Wavelength [A]', ylabel='Normalized flux',
               xticks=x_ticks, xticklabels=x_ticks_str)
        ax.grid(ls='--', alpha=0.2, color='black')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

    # plot combined and shifted spectra - every spectrum shifted for a certain flux offset level
    if plot_shifted:
        flx_offset = 0.15
        fig, ax = plt.subplots(1, 1, figsize=(100, 3. + 1. * n_spectra))
        for i_ex in range(n_spectra):
            ax.plot(target_wvl, flx_new[i_ex, :] + (flx_offset * (i_ex + 1)), lw=0.5, alpha=0.8)
        ax.plot(target_wvl, flx_new_median, c='black', lw=0.8)
        ax.set(xlim=wvl_range, 
               ylim=np.nanpercentile(flx_new_median, [0.4, 99.6]) + np.array([0, flx_offset * n_spectra]),
               xlabel='Wavelength [A]', ylabel='Normalized and shifted flux',
               xticks=x_ticks, xticklabels=x_ticks_str)
        ax.grid(ls='--', alpha=0.2, color='black')
        fig.tight_layout()
        fig.savefig(plot_path[:-4] + '_shifted.png', dpi=200)
        plt.close(fig)

    # return rv corrected and computed median combination of individual exposures
    return flx_new_median


def _evaluate_norm_fit(orig, fit, idx, sigma_low, sigma_high):
    """

    :param orig:
    :param fit:
    :param idx:
    :param sigma_low:
    :param sigma_high:
    :return:
    """
    # diffence to the original data
    diff = orig - fit
    std_diff = np.nanstd(diff[idx])
    # select data that will be fitted
    idx_outlier = np.logical_or(diff < (-1. * std_diff * sigma_low),
                                diff > (std_diff * sigma_high))
    return np.logical_and(idx, ~idx_outlier)


def _spectra_normalize(wvl, spectra_orig, 
                       steps=5, sigma_low=2., sigma_high=2.5, window=15, order=5, n_min_perc=5.,
                       func='cheb', fit_on_idx=None, fit_mask=None, sg_filter=False,
                       return_fit=False, return_idx=False):
    # perform sigma clipping before the next fitting cycle
    idx_fit = np.logical_and(np.isfinite(wvl), np.isfinite(spectra_orig))
    spectra = np.array(spectra_orig)

    if fit_mask is not None:
        idx_fit = np.logical_and(idx_fit, fit_mask)

    if fit_on_idx is not None:
        idx_fit = np.logical_and(idx_fit, fit_on_idx)
        steps = 1  # no clipping performed, one iteration, forced fitting on selected pixels
    else:
        # filter noisy original spectra, so it is easier to determine continuum levels
        if sg_filter:
            spectra = savgol_filter(spectra_orig, window_length=15, polyorder=5)
        init_fit = np.nanmedian(spectra)
        idx_fit = _evaluate_norm_fit(spectra, init_fit, idx_fit, sigma_low*2.5, sigma_high*2.5)
    data_len = np.sum(idx_fit)
    n_fit_points_prev = np.sum(idx_fit)
    for i_f in range(steps):  # number of sigma clipping steps
        # print i_f
        if func == 'cheb':
            chb_coef = np.polynomial.chebyshev.chebfit(wvl[idx_fit], spectra[idx_fit], order)
            cont_fit = np.polynomial.chebyshev.chebval(wvl, chb_coef)
        if func == 'legen':
            leg_coef = np.polynomial.legendre.legfit(wvl[idx_fit], spectra[idx_fit], order)
            cont_fit = np.polynomial.legendre.legval(wvl, leg_coef)
        if func == 'poly':
            poly_coef = np.polyfit(wvl[idx_fit], spectra[idx_fit], order)
            cont_fit = np.poly1d(poly_coef)(wvl)
        if func == 'spline':
            # if i_f == 1:
            #     chb_coef = np.polynomial.chebyshev.chebfit(wvl[idx_fit], spectra[idx_fit], 5)
            #     cont_fit = np.polynomial.chebyshev.chebval(wvl, chb_coef)
            #     idx_fit = _evaluate_norm_fit(spectra, cont_fit, idx_fit, sigma_low, sigma_high)
            spline_coef = splrep(wvl[idx_fit], spectra[idx_fit], k=order, s=window)
            cont_fit = splev(wvl, spline_coef)
            print(i_f, 'points:', n_fit_points_prev, 'knots:', len(spline_coef[0]))
        idx_fit = _evaluate_norm_fit(spectra, cont_fit, idx_fit, sigma_low, sigma_high)
        n_fit_points = np.sum(idx_fit)
        if 100.*n_fit_points/data_len < n_min_perc:
            break
        if n_fit_points == n_fit_points_prev:
            break
        else:
            n_fit_points_prev = n_fit_points
    if return_fit:
        if return_idx:
            return cont_fit, idx_fit
        else:
            return cont_fit
    else:
        return spectra_orig / cont_fit


def renorm_exposure_perorder(exposure_data, ref_flx, ref_wvl,
                             use_rv_key='RV_s1',
                             input_flx_key='flx',
                             output_flx_key='flx_renorm',
                             plot=False, plot_path=None):
    
    rv_val_star = exposure_data[use_rv_key]
    # shift reference spectrum from stars' to baricentric/observed reference frame - use reversed RV value
    ref_wvl_shifted =  correct_wvl_for_rv(ref_wvl, -1.*rv_val_star)

    echelle_orders = _valid_orders_from_keys(exposure_data.keys())
    
    # loop trough all available echelle orders
    for echelle_order_key in echelle_orders:
        # determine observed data that will be used in the correlation procedure
        order_flx = exposure_data[echelle_order_key][input_flx_key]
        order_wvl = exposure_data[echelle_order_key]['wvl']

        # resample reference spectrum to the observed wavelength pixels
        ref_flx_order = _spectra_resample(ref_flx, ref_wvl_shifted, order_wvl)
        # perform renormalization using the supplied reference spectrum
        # get renormalization curve by comparing reference and observed spectrum
        try:
            ref_flx_norm_curve = _spectra_normalize(order_wvl, order_flx / ref_flx_order,
                                                    steps=10, sigma_low=2.5, sigma_high=2.5, n_min_perc=8.,
                                                    order=4, func='poly', return_fit=True)
            # renorm order
            exposure_data[echelle_order_key][output_flx_key] = order_flx / ref_flx_norm_curve
        except:
            print('   Renormalization problem for:', echelle_order_key)
            exposure_data[echelle_order_key][output_flx_key] = order_flx

    # return original data with addition of a renormed spectrum
    return exposure_data
