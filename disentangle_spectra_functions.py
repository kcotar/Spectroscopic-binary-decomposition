import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
  from os import scandir, path  # scandir introduced in py3.x
except:
    pass
from os import system, chdir
from copy import deepcopy
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter, medfilt
from scipy.optimize import minimize
from astropy.io import fits
from common_helper_functions import _order_exposures_by_key, _valid_orders_from_keys, correct_wvl_for_rv, _combine_orders, _spectra_resample
from rv_helper_functions import get_RV_ref_spectrum, get_RV_custom_corr_perorder, get_RV_custom_corr_combined, add_rv_to_metadata


norm_suffix = '_normalised.txt'
sigma_norm_suffix = '_sigma_normalised.txt'


def _go_to_dir(path):
    try:
        system('mkdir ' + path)
    except:
        pass
    chdir(path)


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
        #sigma_data = np.loadtxt(spec_dir + order + sigma_norm_suffix)
        sigma_data = deepcopy(flux_data)
        sigma_data[:, 1] = 0.
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
        print('Exploring orders of exposure:', exposure)
        all_norm_orders = _get_normalised_orders(input_dir, exposure)
        if len(all_norm_orders) > 0:
            # create new dictionary that will hold the data of selected order for a given exposure
            star_data_all[exposure] = {}
            # if available read vhelio velocity from the original reduction fits file
            vh_key = 'VHELIO'
            vhelio = np.nan
            # open and read original reduced fits file
            orig_fits = fits.open(input_dir + exposure + '.fits')
            header_fits = orig_fits[0].header
            if vh_key in header_fits.keys():
                vhelio = header_fits[vh_key]
            orig_fits.close()
            # add vhelio velocity to the data structure
            star_data_all[exposure][vh_key] = vhelio
            # read data of individual orders and save them into the structure
            for get_wvl_order in wvl_orders:
                order_data = get_orderdata_by_wavelength(input_dir + exposure + '/',
                                                         all_norm_orders, get_wvl_order)
                if order_data[0] is not None:
                    star_data_all[exposure][get_wvl_order] = {
                        'wvl': order_data[0],
                        'flx': order_data[1],
                        'sig': order_data[2],
                        'flx1': np.ones_like(order_data[1]),
                        'flx2': np.zeros_like(order_data[1]),
                    }

            # add RV flags, which determines which of the orders can be used for RV estimation
            #star_data_all['RV_s1_use'] =
            #star_data_all['RV_s2_use'] =

    return star_data_all


def create_new_reference(exposures_all, target_wvl,
                         percentile=None, w_filt=None,
                         use_flx_key='flx', use_rv_key='RV_s1',
                         plot_combined=False, plot_path='plot_combined.png',
                         plot_shifted=False):
    """

    :param exposures_all:
    :param target_wvl:
    :param percentile:
    :param w_filt:
    :param use_flx_key:
    :param use_rv_key:
    :param plot_combined:
    :param plot_path:
    :param plot_shifted:
    :return:
    """
    flx_new = list([])
    all_exp_ids = _order_exposures_by_key(exposures_all, exposures_all.keys(),
                                          sort_key=use_rv_key)
    for exposure_id in all_exp_ids:
        exposure_data = exposures_all[exposure_id]    
        
        # combine all resampled and RV moved spectra
        exposure_new_flx = _combine_orders(exposure_data, target_wvl,
                                           use_flx_key=use_flx_key, use_rv_key=use_rv_key)            
        flx_new.append(exposure_new_flx)

    # compute median of all considered exposures
    flx_new = np.array(flx_new)
    flx_new_median = np.nanmedian(flx_new, axis=0)
    flx_new_std = np.nanstd(flx_new, axis=0)
    idx_median = np.isfinite(flx_new_median)
    wvl_range = (np.min(target_wvl[idx_median]) - 2.,
                 np.max(target_wvl[idx_median]) + 2.)

    n_spectra = flx_new.shape[0]
    x_ticks = range(4500, 7000, 20)
    x_ticks_str = [str(xt) for xt in x_ticks]
    # plot combined spectra - all around normalized level of 1
    if plot_combined:
        fig, ax = plt.subplots(2, 1, figsize=(135, 6.), sharex=True)

        # plot individual spectra and final combined spectrum
        for i_ex in range(n_spectra):
            ax[0].plot(target_wvl, flx_new[i_ex, :], lw=0.5, alpha=0.33)
        ax[0].plot(target_wvl, flx_new_median, c='black', lw=0.8)
        if w_filt is not None:
            ax[0].plot(target_wvl, medfilt(flx_new_median, w_filt), c='green', lw=0.5)
        ax[0].set(xlim=wvl_range, ylim=np.nanpercentile(flx_new_median, [0.4, 99.6]),
                  # xlabel='Wavelength [A]',
                  # xticks=x_ticks, xticklabels=x_ticks_str,
                  ylabel='Normalized flux')

        # plot deviations from the reference spectrum - could be used for RV bad wavelength masking
        flx_new_std = np.nanstd(flx_new - flx_new_median, axis=0)
        for i_ex in range(n_spectra):
            ax[1].plot(target_wvl, flx_new[i_ex, :] - flx_new_median, lw=0.5, alpha=0.33)
        ax[1].set(xlim=wvl_range, ylim=[-0.04, 0.04],
                  # xticks=x_ticks, xticklabels=x_ticks_str,
                  xlabel='Wavelength [A]', ylabel='Flux diff')
        ax[1].plot(target_wvl, flx_new_std, c='black', lw=0.8)
        ax[1].plot(target_wvl, -flx_new_std, c='black', lw=0.8)

        # final plot visual corrections
        ax[0].grid(ls='--', alpha=0.2, color='black')
        ax[1].grid(ls='--', alpha=0.2, color='black')
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    # plot combined and shifted spectra - every spectrum shifted for a certain flux offset level
    if plot_shifted:
        # compute function to be plotted as deviations around median flux value
        fill_1 = np.nanpercentile(flx_new, 15, axis=0)
        fill_2 = np.nanpercentile(flx_new, 85, axis=0)
        idx_fill = np.logical_and(np.isfinite(fill_1), np.isfinite(fill_2))
        # start plotting
        y_range = np.nanpercentile(flx_new_median, [0.4, 99.6])
        flx_offset = 0.75 * (y_range[1] - y_range[0])  # half of expected y range
        fig, ax = plt.subplots(1, 1, figsize=(90, 3. + 0.8 * n_spectra))
        for i_ex in range(n_spectra):
            ax.plot(target_wvl, flx_new[i_ex, :] + (flx_offset * (i_ex + 1)), lw=0.6, alpha=0.8)
            ax.text(wvl_range[0]+5, 1 + + (flx_offset * (i_ex + 1)), all_exp_ids[i_ex].split('.')[0],
                    fontsize=10, va='center')
        # ax.fill_between(target_wvl, fill_1, fill_2,
        #                 color='lightgrey', where=idx_fill)
        ax.fill_between(target_wvl, flx_new_median-flx_new_std, flx_new_median+flx_new_std,
                        color='lightgrey', where=idx_fill)
        ax.plot(target_wvl, flx_new_median, c='black', lw=0.8)
        ax.set(xlim=wvl_range, 
               ylim=y_range + np.array([0, flx_offset * n_spectra]),
               # xticks=x_ticks, xticklabels=x_ticks_str,
               xlabel='Wavelength [A]', ylabel='Normalized and shifted flux')
        ax.grid(ls='--', alpha=0.2, color='black')
        fig.tight_layout()
        fig.savefig(plot_path[:-4] + '_shifted.png', dpi=150)
        plt.close(fig)

    # return rv corrected and computed median combination of individual exposures
    if percentile is None:
         flx_final = flx_new_median  # / np.nanpercentile(flx_new_median, 80)
    else:
        flx_new_perc = np.nanpercentile(flx_new, percentile, axis=0)
        flx_final = flx_new_perc  # / np.nanpercentile(flx_new_median, 80)

    # apply median filtering if requested
    if w_filt is not None:
        flx_final = medfilt(flx_final, w_filt)

    # return new median combined spectrum
    return flx_final, flx_new_std


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
                       return_fit=False, return_idx=False, median_init=True):
    """

    :param wvl:
    :param spectra_orig:
    :param steps:
    :param sigma_low:
    :param sigma_high:
    :param window:
    :param order:
    :param n_min_perc:
    :param func:
    :param fit_on_idx:
    :param fit_mask:
    :param sg_filter:
    :param return_fit:
    :param return_idx:
    :return:
    """
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
        if median_init:
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
            # print(i_f, 'points:', n_fit_points_prev, 'knots:', len(spline_coef[0]))
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
    """

    :param exposure_data:
    :param ref_flx:
    :param ref_wvl:
    :param use_rv_key:
    :param input_flx_key:
    :param output_flx_key:
    :param plot:
    :param plot_path:
    :return:
    """
    print('   Input normalization flux key is', input_flx_key, 'and RV key is', use_rv_key)
    rv_val_star = exposure_data[use_rv_key]
    if not np.isfinite(rv_val_star):
        rv_val_star = 0

    # shift reference spectrum from stars' rest to barycentric/observed reference frame - use reversed RV value
    ref_wvl_shifted = correct_wvl_for_rv(ref_wvl, -1.*rv_val_star)

    echelle_orders = _valid_orders_from_keys(exposure_data.keys())
    
    # loop trough all available Echelle orders
    for echelle_order_key in echelle_orders:
        # determine observed data that will be used in the correlation procedure
        order_flx = exposure_data[echelle_order_key][input_flx_key]
        order_wvl = exposure_data[echelle_order_key]['wvl']

        # resample reference spectrum to the observed wavelength pixels
        ref_flx_order = _spectra_resample(ref_flx, ref_wvl_shifted, order_wvl)
        # perform renormalization using the supplied reference spectrum
        # get renormalization curve by comparing reference and observed spectrum
        try:
            wvl_len = len(order_wvl)
            ref_flx_norm_curve = _spectra_normalize(np.arange(wvl_len), order_flx / ref_flx_order,
                                                    steps=10, sigma_low=2.5, sigma_high=2.5, n_min_perc=8.,
                                                    order=4, func='cheb', return_fit=True)
            # renorm order
            exposure_data[echelle_order_key][output_flx_key] = order_flx / ref_flx_norm_curve

            if plot:
                fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15, 5))
                ax[0].plot(order_wvl, order_flx, lw=0.5, label='Original')
                ax[0].plot(order_wvl, ref_flx_order, lw=0.5, label='Reference')
                ax[0].plot(order_wvl, order_flx / ref_flx_norm_curve, lw=0.5, label='Renormed')
                ax[1].plot(order_wvl, order_flx / ref_flx_order, lw=0.5)
                ax[1].plot(order_wvl, ref_flx_norm_curve, lw=0.5)
                ax[1].set(xlim=[order_wvl[0]-0.2, order_wvl[-1]+0.2])
                ax[0].legend()
                fig.tight_layout()
                fig.subplots_adjust(hspace=0, wspace=0)
                if plot_path is None:
                    fig.show()
                else:
                    fig.savefig(plot_path[:-4] + '_' + str(echelle_order_key) + '.png', dpi=150)
                plt.close(fig)

        except Exception as e:
            print('   Renormalization problem for:', echelle_order_key, e)
            exposure_data[echelle_order_key][output_flx_key] = order_flx

    # return original data with addition of a renormed spectrum
    return exposure_data


def _flx_amp(flx, amp, cont=1.):
    """

    :param flx:
    :param amp:
    :return:
    """
    return cont - amp * (cont - flx)


def remove_ref_from_exposure(exposure_data, ref_flx, ref_wvl,
                             primary=True,
                             use_rv_key='RV_s1',
                             input_flx_key='flx',
                             fit_before_removal=False,
                             output_flx_key='flx_secon',
                             ref_orig=None, w_filt=None,
                             plot=False, plot_path='plot.png',
                             verbose=True):
    """

    :param exposure_data:
    :param ref_flx:
    :param ref_wvl:
    :param primary:
    :param use_rv_key:
    :param input_flx_key:
    :param fit_before_removal:
    :param output_flx_key:
    :param ref_orig:
    :param w_filt:
    :param plot:
    :param plot_path:
    :param verbose:
    :return:
    """
    if use_rv_key not in exposure_data.keys():
        if verbose:
            print('   WARNING: Given RV key (' + use_rv_key + ') not found -> RV = 0. will be used.')
        rv_val_star = 0.
    else:
        rv_val_star = exposure_data[use_rv_key]

    if not np.isfinite(rv_val_star):
        if verbose:
            print('   WARNING: Component removal not possible as RV was not estimated.')
        return exposure_data

    # shift reference spectrum from stars' rest to barycentric/observed reference frame - use reversed RV value
    ref_wvl_shifted = correct_wvl_for_rv(ref_wvl, -1. * rv_val_star)

    echelle_orders = _valid_orders_from_keys(exposure_data.keys())

    # loop trough all available Echelle orders
    for echelle_order_key in echelle_orders:
        # determine observed data that will be used in the primary removal procedure
        order_flx = exposure_data[echelle_order_key][input_flx_key]
        order_wvl = exposure_data[echelle_order_key]['wvl']

        # resample reference spectrum to the observed wavelength pixels
        ref_flx_order = _spectra_resample(ref_flx, ref_wvl_shifted, order_wvl)

        # adjust/force reference flux to have the same amplitude as observed spectrum
        # useful for stars with lower snr and/or reduction problems
        if fit_before_removal:
            # helper function used in the minimization process
            def min_flx_dif_prim(amp):
                # manhattan spectral distance between two spectra
                return np.sum(np.abs((order_flx - 1.) - _flx_amp(ref_flx_order, amp, cont=0.)))

            def min_flx_dif_sec(amp):
                # manhattan spectral distance between two spectra
                return np.sum(np.abs(order_flx - _flx_amp(ref_flx_order, amp, cont=1.)))

            # minimize difference between observed and reference spectrum
            if primary:
                min_res = minimize(min_flx_dif_prim, [1.], bounds=[(0., 2.)])
            else:
                min_res = minimize(min_flx_dif_sec, [1.], bounds=[(0., 2.)])

            # get the best amplitude correction factor
            amp_use = min_res['x'][0]
            if verbose:
                print('   Flx amp modification (order - ' + str(echelle_order_key) + '): {:.3f}'.format(amp_use))
            # correct flux for determined amplitude
            if primary:
                ref_flx_order = _flx_amp(ref_flx_order, amp_use, cont=0.)
            else:
                ref_flx_order = _flx_amp(ref_flx_order, amp_use, cont=1.)

        # remove contribution of a reference spectrum by a simple spectral substraction
        order_flx_diff = order_flx - ref_flx_order
        # order_flx_diff = order_flx / ref_flx_order
        if w_filt is not None:
            exposure_data[echelle_order_key][output_flx_key] = medfilt(order_flx_diff, w_filt)
        else:
            exposure_data[echelle_order_key][output_flx_key] = order_flx_diff

    if plot:
        flx_orig_comb = _combine_orders(exposure_data, ref_wvl_shifted,
                                        use_flx_key=input_flx_key, use_rv_key=None)
        flx_seco_comb = _combine_orders(exposure_data, ref_wvl_shifted,
                                        use_flx_key=output_flx_key, use_rv_key=None)

        y_range = np.nanpercentile(flx_orig_comb, [0.4, 99.6])
        flx_offset = 0.75 * (y_range[1] - y_range[0])
        wvl_range = (np.min(ref_wvl_shifted[np.isfinite(flx_orig_comb)]) - 2.,
                     np.max(ref_wvl_shifted[np.isfinite(flx_orig_comb)]) + 2.)
        x_ticks = range(4500, 7000, 20)
        x_ticks_str = [str(xt) for xt in x_ticks]

        fig, ax = plt.subplots(1, 1, figsize=(120, 5.))

        if primary:
            ax.plot(ref_wvl_shifted, flx_orig_comb, c='C3', lw=0.7, alpha=0.8)
            ax.plot(ref_wvl_shifted, 1. + ref_flx, c='black', lw=0.5, alpha=0.8)
            ax.plot(ref_wvl_shifted, 0.04 + flx_seco_comb, c='C2', lw=0.7, alpha=0.8)
        else:
            ax.plot(ref_wvl_shifted, flx_orig_comb, c='C3', lw=0.7, alpha=0.8)
            ax.plot(ref_wvl_shifted, ref_flx, c='black', lw=0.5, alpha=0.8)
            ax.plot(ref_wvl_shifted, 1.04 + flx_seco_comb, c='C2', lw=0.7, alpha=0.8)

        ax.axhline(1.04, c='black', ls='--', lw=0.5, alpha=0.9)
        if ref_orig is not None:
            ax.plot(ref_wvl_shifted, ref_orig - flx_offset, c='red', lw=0.8)
            y_range[0] -= flx_offset
        ax.set(xlim=wvl_range,
               ylim=[y_range[0], 1.05],
               xlabel='Wavelength [A]', ylabel='Normalized and median removed flux',
               xticks=x_ticks, xticklabels=x_ticks_str)
        ax.grid(ls='--', alpha=0.2, color='black')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    # return original data with addition of a reference corrected per order spectrum
    return exposure_data


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# ------------ Function that runs the whole procedure at once --------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
def _are_orders_renormed(exposures_data, input_key):
    """

    :param exposures_data:
    :param input_key:
    :return:
    """
    exposures_all = list(exposures_data.keys())
    n_renorm = 0
    for exp_id in exposures_all:
        orders_all = _valid_orders_from_keys(exposures_data[exp_id].keys())
        n_orders = 0
        for ord_id in orders_all:            
            if input_key+'_renorm' in list(exposures_data[exp_id][ord_id].keys()):
                n_orders += 1
        if n_orders == len(orders_all):
            n_renorm += 1
    # return True if all orders in all exposures have renormed flux data
    return n_renorm == len(exposures_all)


def run_complete_RV_and_template_discovery_procedure(star_data, obs_metadata,  # datasets and tables
                                                     ref_flx, ref_wvl,  # spectral reference data
                                                     star_id='', in_flx_key='flx', rv_key='RV_s1', # exact data that will be used
                                                     primary=True,  # are we processing the most obvoius spectral component
                                                     combined_rv_spectrum=False,  # processing settings
                                                     save_plots=True, plot_prefix='', plot_suffix='',  # plotting settings
                                                     verbose=True,   # screen verbocity setting
                                                     ):
    """

    :param star_data:
    :param obs_metadata:
    :param ref_flx:
    :param ref_wvl:
    :param star_id:
    :param in_flx_key:
    :param rv_key:
    :param primary:
    :param combined_rv_spectrum:
    :param save_plots:
    :param plot_prefix:
    :param plot_suffix:
    :param verbose:
    :return:
    """
    # some component specific processing and output settings
    if primary:
        c_id = 1
        cont_value = 1.
    else:
        c_id = 2
        cont_value = 0.

    # set flux dataset that will be used in the processing
    use_flx_key = deepcopy(in_flx_key)

    if verbose:
        print('  Spectra used for RV determination:', use_flx_key)

    # get per order RV velocities for every exposure
    for exp_id in star_data.keys():
        if verbose:
            print('  Exposure:', exp_id)

        if combined_rv_spectrum:
            # compute RV from a combined spectrum (stack of individual echelle orders)
            rv_png = plot_prefix + '_' + exp_id + '_rv' + str(c_id) + '-combined' + plot_suffix + '.png'
            rv_med, rv_std = get_RV_custom_corr_combined(deepcopy(star_data[exp_id]), ref_flx, ref_wvl,
                                                         cont_value=cont_value,
                                                         rv_ref_val=None, use_flx_key=use_flx_key,
                                                         plot_rv=True, plot_path=rv_png)
            if verbose:
                print('   Combined RV value:', rv_med, rv_std)
        else:
            # compute mean RV from all considered orders
            rv_png = plot_prefix + '_' + exp_id + '_rv' + str(c_id) + '-orders' + plot_suffix + '.png'
            rv_all, rv_med, rv_std = get_RV_custom_corr_perorder(deepcopy(star_data[exp_id]), ref_flx, ref_wvl,
                                                                 cont_value=cont_value,
                                                                 rv_ref_val=None, use_flx_key=use_flx_key,
                                                                 plot_rv=True, plot_path=rv_png)
            if verbose:
                print('   Median RV value:', rv_med, rv_std)
            star_data[exp_id][rv_key + '_orders'] = rv_all

            # store values to the dictionary
        star_data[exp_id][rv_key] = rv_med
        star_data[exp_id]['e_' + rv_key] = rv_std

    # compute median spectrum of a secondary star and use it as a new and updated RV template
    use_flx_key_median = deepcopy(in_flx_key)

    if verbose:
        print(' Creating median reference spectrum')
    combined_png = plot_prefix + '_s' + str(c_id) + '_combined' + plot_suffix + '.png'
    # get new reference spectrum as median of all alligned spectra, per wvl pixel std is also computed and returned 
    ref_flx_new, _ = create_new_reference(star_data, ref_wvl,
                                          # percentile=85.,
                                          w_filt=7,
                                          use_flx_key=use_flx_key_median, use_rv_key=rv_key,
                                          plot_combined=True, plot_shifted=save_plots,
                                          plot_path=combined_png)

    # Add RV values of a binary star to the observations metadata table and plot phase RV diagram
    rv_phase_plot_png = plot_prefix + '_RV' + str(c_id) + plot_suffix + '.png'
    obs_metadata = add_rv_to_metadata(star_data, star_id,
                                      deepcopy(obs_metadata), rv_key,
                                      # always save this plot as it is the final result of the binary spectral processing
                                      plot=True, plot_path=rv_phase_plot_png)

    # finally return all important structures that hold gathered information and spectra
    return star_data, obs_metadata, ref_flx_new


def show_spectra_heliocentric(star_data, order,
                              tellurics_data=None, prefix=''):
    """

    :param star_data:
    :param order:
    :param tellurics_data:
    :param prefix:
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(85, 5))
    w_min = 10000
    w_max = 0
    for exp_is in star_data.keys():
        exposure_data = star_data[exp_is]
        if order not in exposure_data.keys():
            continue
        y_flx = exposure_data[order]['flx']
        x_wvl = exposure_data[order]['wvl']
        x_wvl = correct_wvl_for_rv(x_wvl, exposure_data['VHELIO'])  # + or - VHELIO??
        w_min = min(w_min, np.nanmin(x_wvl))
        w_max = max(w_max, np.nanmax(x_wvl))
        ax.plot(x_wvl, y_flx, lw=0.5, alpha=0.6)
    # add telluric reference spectrum to the combined plot
    if tellurics_data is not None:
        ax.plot(tellurics_data[:, 0], tellurics_data[:, 1], lw=0.7, alpha=0.75, c='black')
    # additional plotting settings
    ax.set(ylim=(0.8, 1.05), xlim=(w_min, w_max))
    ax.grid(ls='--', alpha=0.2, color='black')
    fig.tight_layout()
    fig.savefig(prefix + 'spec_helio_'+str(order)+'.png', dpi=150)
    plt.close(fig)
    return True


def plot_combined_spectrum_using_RV(exposure_data,
                                    ref_flx_s1, ref_flx_s2, ref_wvl,
                                    prim_rv='RV_s1', sec_rv='RV_s2', input_flx_key='flx',
                                    plot=True, plot_path='plot.png'):
    """

    :param exposure_data:
    :param ref_flx_s1:
    :param ref_flx_s2:
    :param ref_wvl:
    :param prim_rv:
    :param sec_rv:
    :param input_flx_key:
    :param plot:
    :param plot_path:
    :return:
    """

    # loop trough all available orders
    flx_orig_comb = _combine_orders(exposure_data, ref_wvl,
                                    use_flx_key=input_flx_key, use_rv_key=None)
    # shift reference spectra into observed stars frame
    rv_s1 = exposure_data[prim_rv]
    rv_s2 = exposure_data[sec_rv]

    if not np.isfinite(rv_s1):
        rv_s1 = 0
    if not np.isfinite(rv_s2):
        rv_s2 = 0

    use_wvl_s1 = correct_wvl_for_rv(ref_wvl, -1. * rv_s1)
    use_wvl_s2 = correct_wvl_for_rv(ref_wvl, -1. * rv_s2)
    # resample reference spectra to match with observed wavelength spacing
    use_flx_s1 = _spectra_resample(ref_flx_s1, use_wvl_s1, ref_wvl)
    use_flx_s2 = _spectra_resample(ref_flx_s2, use_wvl_s2, ref_wvl)

    # determine plotting settings
    y_range = np.nanpercentile(flx_orig_comb, [0.4, 99.6])
    wvl_range = (np.min(ref_wvl[np.isfinite(flx_orig_comb)]) - 2.,
                 np.max(ref_wvl[np.isfinite(flx_orig_comb)]) + 2.)
    x_ticks = range(4500, 7000, 20)
    x_ticks_str = [str(xt) for xt in x_ticks]

    if plot:
        # plot everything together
        fig, ax = plt.subplots(2, 1, figsize=(120, 8.), sharex=True)
        # individual plots
        ax[0].plot(ref_wvl, flx_orig_comb, c='black', lw=0.8, alpha=0.9, label='Original')
        ax[0].plot(ref_wvl, use_flx_s1, c='C2', lw=0.6, alpha=0.9, label='Spectrum 1')
        ax[0].plot(ref_wvl, 1.0 + use_flx_s2, c='C3', lw=0.6, alpha=0.9, label='Spectrum 2')
        # residual towards original spectrum
        flx_res = flx_orig_comb - (use_flx_s1 + use_flx_s2)
        y_range2 = np.nanpercentile(flx_res, [0.3, 99.7])
        ax[1].plot(ref_wvl, flx_res, c='black', lw=0.7, alpha=1)
        # make nicer looking plot with labels etc
        ax[0].set(ylim=[y_range[0], 1.03],
                  # xticks=x_ticks, xticklabels=x_ticks_str,
                  ylabel='Normalized flux')
        ax[1].set(xlim=wvl_range, ylim=y_range2,
                  # xticks=x_ticks, xticklabels=x_ticks_str,
                  xlabel='Wavelength [A]', ylabel='Residual')
        ax[0].grid(ls='--', alpha=0.2, color='black')
        ax[1].grid(ls='--', alpha=0.2, color='black')
        ax[0].legend()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        return True
    else:
        return False
