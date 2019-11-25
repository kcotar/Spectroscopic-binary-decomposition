import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt

from astropy.io import fits
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel
from scipy.signal import correlate
from copy import deepcopy
from common_helper_functions import _valid_orders_from_keys, correct_wvl_for_rv, _combine_orders

# speed of light in km/s
c_val = const.c.value / 1000.


def get_RV_ref_spectrum(spectrum_full_path,
                        wvl_oversample_factor=5.):
    """

    :param spectrum_full_path:
    :param wvl_oversample_factor:
    :return:
    """
    ref_data = fits.open(spectrum_full_path)
    ref_flx_orig = ref_data[0].data
    ref_wvl_orig = ref_data[0].header['CRVAL1'] + np.arange(len(ref_flx_orig)) * ref_data[0].header['CDELT1']
    # resample data to a finer resolution
    ref_wvl_d = ref_data[0].header['CDELT1'] / wvl_oversample_factor
    ref_wvl = np.arange(ref_wvl_orig[0], ref_wvl_orig[-1], ref_wvl_d)
    ref_flx = np.interp(ref_wvl, ref_wvl_orig, ref_flx_orig)
    # output processed spectrum
    return ref_flx, ref_wvl


def spectra_logspace(flx, wvl):
    """

    :param flx:
    :param wvl:
    :return:
    """
    wvl_new = np.logspace(np.log10(wvl[0]), np.log10(wvl[-1]), num=len(wvl))
    return np.interp(wvl_new, wvl, flx), wvl_new


def correlate_spectra(obs_flx, obs_wvl, ref_flx, ref_wvl,
                      plot=None):
    """

    :param obs_flx:
    :param obs_wvl:
    :param ref_flx:
    :param ref_wvl:
    :param plot:
    :return:
    """

    # convert spectra sampling to logspace
    obs_flux_res_log, _ = spectra_logspace(obs_flx, obs_wvl)
    ref_flux_sub_log, wvl_log = spectra_logspace(ref_flx, ref_wvl)
    wvl_step = ref_wvl[1] - ref_wvl[0]

    # correlate the two spectra
    min_flux = 0.95
    # set near continuum spectral wiggles to the continuum level
    # ref_flux_sub_log[ref_flux_sub_log > min_flux] = 1.
    # obs_flux_res_log[obs_flux_res_log > min_flux] = 1.
    corr_res = correlate(1.-ref_flux_sub_log, 1.-obs_flux_res_log, mode='same', method='fft')

    # create a correlation subset that will actually be analysed
    corr_w_size_wide = 130  # preform rough search of the local peak
    corr_w_size = 35  # narrow down to the exact location of the CC peak
    corr_c_off = np.int64(len(corr_res) / 2.)
    corr_c_off += np.nanargmax(corr_res[corr_c_off - corr_w_size_wide : corr_c_off + corr_w_size_wide]) - corr_w_size_wide
    corr_pos_min = corr_c_off - corr_w_size
    corr_pos_max = corr_c_off + corr_w_size

    # if plot is not None:
    #     plt.plot(corr_res, lw=1)
    #     plt.axvline(corr_pos_min, color='black')
    #     plt.axvline(corr_pos_max, color='black')
    #     plt.savefig(plot+'_1.png', dpi=300)
    #     plt.close()

    # print corr_pos_min, corr_pos_max
    corr_res_sub = corr_res[corr_pos_min:corr_pos_max]
    corr_res_sub -= np.median(corr_res_sub)
    corr_res_sub_x = np.arange(len(corr_res_sub))

    # analyze correlation function by fitting gaussian/voigt/lorentzian distribution to it
    peak_model = GaussianModel()
    additional_model = LinearModel()  # ConstantModel()
    parameters = additional_model.make_params(intercept=np.nanmin(corr_res_sub), slope=0)
    parameters += peak_model.guess(corr_res_sub, x=corr_res_sub_x)
    fit_model = peak_model + additional_model
    corr_fit_res = fit_model.fit(corr_res_sub, parameters, x=corr_res_sub_x)
    corr_center = corr_fit_res.params['center'].value
    corr_center_max = corr_res_sub_x[np.argmax(corr_res_sub)]

    # determine the actual shift
    idx_no_shift = np.int32(len(corr_res) / 2.)
    idx_center = corr_c_off - corr_w_size + corr_center
    idx_center_max = corr_c_off - corr_w_size + corr_center_max
    log_shift_px = idx_no_shift - idx_center
    log_shift_px_max = idx_no_shift - idx_center_max
    log_shift_wvl = log_shift_px * wvl_step

    wvl_log_new = wvl_log - log_shift_wvl
    rv_shifts = (wvl_log_new[1:] - wvl_log_new[:-1]) / wvl_log_new[:-1] * c_val * log_shift_px
    rv_shifts_max = (wvl_log_new[1:] - wvl_log_new[:-1]) / wvl_log_new[:-1] * c_val * log_shift_px_max
    rv_shifts_5 = (wvl_log_new[1:] - wvl_log_new[:-1]) / wvl_log_new[:-1] * c_val * 5

    if plot is not None:
        plt.plot(corr_res_sub, lw=1, color='C0')
        plt.axvline(corr_center_max, color='C0')
        plt.plot(corr_fit_res.best_fit, lw=1, color='C1')
        plt.axvline(corr_center, color='C1')
        plt.title(u'RV max: {:.2f}, RV fit: {:.2f}, $\Delta$RV per 5: {:.2f}, center wvl {:.1f}'.format(np.nanmedian(rv_shifts_max), np.nanmedian(rv_shifts), np.nanmedian(rv_shifts_5), np.nanmean(obs_wvl)))
        plt.savefig(plot+'_2.png', dpi=200)
        plt.close()

    if log_shift_wvl < 2.:
        # return np.nanmedian(rv_shifts_max)
        return np.nanmedian(rv_shifts), np.nanmedian(ref_wvl)
    else:
        # something went wrong
        return np.nan, np.nanmedian(ref_wvl)


def correlate_order(obs_flx, obs_wvl,
                    ref_flx, ref_wvl,
                    order_crop_frac=0.,
                    plot=False, plot_path='rv_corr_plot.png'):
    """

    :param obs_flx:
    :param obs_wvl:
    :param ref_flx:
    :param ref_wvl:
    :param order_crop_frac:
    :param plot:
    :param plot_path:
    :return:
    """
    # determine wavelength subset of a given order on which a correlation will be performed
    wvl_len = len(obs_wvl) - 1
    wvl_beg = obs_wvl[int(wvl_len * order_crop_frac / 100.)]
    wvl_end = obs_wvl[int(wvl_len * (1. - order_crop_frac/100.))]

    # resample data to the same wavelength step
    idx_ref_use = np.logical_and(ref_wvl >= wvl_beg, ref_wvl <= wvl_end)
    if np.sum(idx_ref_use) < 20:
        return np.nan, np.nanmedian(ref_wvl[idx_ref_use])

    ref_wvl_use = deepcopy(ref_wvl[idx_ref_use])
    ref_flx_use = deepcopy(ref_flx[idx_ref_use])
    # linear interpolation of observation to the same wvl bins as reference data
    obs_flx_use = np.interp(ref_wvl_use, obs_wvl, obs_flx)

    # add re-normalization step if needed
    # TODO

    # perform correlation between the datasets
    try:
        if plot:
            return correlate_spectra(obs_flx_use, ref_wvl_use, ref_flx_use, ref_wvl_use, plot=order_txt_file[:-4])
        else:
            return correlate_spectra(obs_flx_use, ref_wvl_use, ref_flx_use, ref_wvl_use)
    except:
        print('    Correlation problem')
        return np.nan, np.nanmedian(ref_wvl[idx_ref_use])


def get_RV_custom_corr_perorder(exposure_data, rv_ref_flx, rv_ref_wvl,
                                rv_ref_val=None, use_flx_key='flx',
                                plot_rv=False, plot_path='rv_plot.png'):
    """

    :param exposure_data:
    :param rv_ref_flx:
    :param rv_ref_wvl:
    :param rv_ref_val:
    :param use_flx_key:
    :param plot_rv:
    :param plot_path:
    :return:
    """
    # prepare list of per order RV determinations
    rv_shifts = list([])
    echelle_orders = _valid_orders_from_keys(exposure_data.keys())
    
    # loop trough all available orders
    for echelle_order_key in echelle_orders:
        # determine observed data that will be used in the correlation procedure
        order_flx = exposure_data[echelle_order_key][use_flx_key]
        order_wvl = exposure_data[echelle_order_key]['wvl']
        # perform correlation with reference spectrum
        rv_order_val, _ = correlate_order(order_flx, order_wvl,  # observed spectrum data
                                          rv_ref_flx, rv_ref_wvl,  # reference spectrum data
                                          plot=False, plot_path=plot_path)
        rv_shifts.append(rv_order_val)

    # nothing can be done if no per order RV values were determined for a given exposure/spectrum
    if len(rv_shifts) <= 0:
        return np.nan, np.nan

    # analyse obtained RV shifts
    rv_shifts = np.array(rv_shifts)
    rv_median = np.nanmedian(rv_shifts)
    # remove gross outliers before computing rv std value
    rv_shifts[np.abs(rv_shifts - rv_median) > 10.] = np.nan
    n_fin_rv = np.sum(np.isfinite(rv_shifts))
    # cumpute final RV values
    rv_median = np.nanmedian(rv_shifts)
    rv_std = np.nanstd(rv_shifts)

    # print(rv_median, rv_std, n_fin_rv)

    if plot_rv and np.isfinite(rv_median):
        plt.scatter(echelle_orders, rv_shifts)
        plt.axhline(rv_median, label='Median RV', c='C3', ls='--')
        plt.ylim(rv_median - 6.,
                 rv_median + 6.)
        plt.xlim(np.nanmin(echelle_orders) - 25.,
                 np.nanmax(echelle_orders) + 25.)
        if rv_ref_val is not None:
            plt.axhline(ref_val, label='Ref RV', c='black', ls='--')
        plt.xlabel('Echelle order [A]')
        plt.ylabel('Radial velocity [km/s]')
        plt.grid(ls='--', alpha=0.2, color='black')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=250)
        plt.close()

    # return all derived velocities and its std values
    return rv_median, rv_std


def get_RV_custom_corr_combined(exposure_data, rv_ref_flx, rv_ref_wvl,
                                rv_ref_val=None, use_flx_key='flx',
                                plot_rv=False, plot_path='rv_plot.png'):
    """

    :param exposure_data:
    :param rv_ref_flx:
    :param rv_ref_wvl:
    :param rv_ref_val:
    :param use_flx_key:
    :param plot_rv:
    :param plot_path:
    :return:
    """

    # combne all used orders into one spectrum that will be used during CCF RV procedure
    # orders will not be wavelenght corrected
    flx_exposure_comb = _combine_orders(exposure_data, rv_ref_wvl,
                                        use_flx_key=use_flx_key, use_rv_key=None)
    idx_flx_exp_valid = np.isfinite(flx_exposure_comb)
    # limit reference data to the observed span
    idx_wvl_use = np.logical_and(rv_ref_wvl >= np.min(rv_ref_wvl[idx_flx_exp_valid]),
                                 rv_ref_wvl <= np.max(rv_ref_wvl[idx_flx_exp_valid]))
    # fill missing flx data with continuum values
    flx_exposure_comb[~idx_flx_exp_valid] = 1.

    # correlate data and get RV value
    rv_combined, _ = correlate_spectra(flx_exposure_comb[idx_wvl_use], rv_ref_wvl[idx_wvl_use], 
                                        rv_ref_flx[idx_wvl_use], rv_ref_wvl[idx_wvl_use])

    # TODO: determine uncertainty of the determined radial velocity
    return rv_combined, np.nan
