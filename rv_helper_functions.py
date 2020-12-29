import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt

from astropy.io import fits
from lmfit.models import GaussianModel, VoigtModel, LinearModel, ConstantModel
from scipy.signal import correlate
from copy import deepcopy
from common_helper_functions import _valid_orders_from_keys, _combine_orders

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
                      plot=False, plot_path='plot.png', cont_value=1.):
    """

    :param obs_flx:
    :param obs_wvl:
    :param ref_flx:
    :param ref_wvl:
    :param plot:
    :param plot_path:
    :param cont_value:
    :return:
    """
    # convert spectra sampling to logspace
    obs_flux_res_log, _ = spectra_logspace(obs_flx, obs_wvl)
    ref_flux_sub_log, wvl_log = spectra_logspace(ref_flx, ref_wvl)
    wvl_step = ref_wvl[1] - ref_wvl[0]

    # fill missing values
    obs_flux_res_log[~np.isfinite(obs_flux_res_log)] = cont_value
    ref_flux_sub_log[~np.isfinite(ref_flux_sub_log)] = cont_value

    # correlate the two spectra
    # min_flux = 0.95
    # set near continuum spectral wiggles to the continuum level
    noRV_flux_mask = np.std(ref_flux_sub_log - cont_value)
    # print('No RV flux level:', noRV_flux_mask)
    ref_flux_sub_log[np.abs(ref_flux_sub_log - cont_value) < noRV_flux_mask] = cont_value
    # obs_flux_res_log[np.abs(obs_flux_res_log - cont_value) < noRV_flux_mask] = cont_value
    corr_res = correlate(cont_value - ref_flux_sub_log, cont_value - obs_flux_res_log,
                         mode='same', method='fft')
    # normalize correlation by the number of involved wavelength bins
    # corr_res /= len(corr_res)

    # create a correlation subset that will actually be analysed
    corr_w_size_wide = 130  # preform rough search of the local peak
    corr_w_size = 60  # narrow down to the exact location of the CC peak
    corr_c_off = np.int64(len(corr_res) / 2.)
    corr_c_off += np.nanargmax(corr_res[corr_c_off - corr_w_size_wide: corr_c_off + corr_w_size_wide]) - corr_w_size_wide
    corr_pos_min = corr_c_off - corr_w_size
    corr_pos_max = corr_c_off + corr_w_size

    if plot:
        w_multi = 6.
        x_corr_res = np.arange(len(corr_res))
        idx = (x_corr_res - corr_c_off) < w_multi*corr_w_size_wide
        plt.plot(x_corr_res[idx], corr_res[idx], lw=1)
        plt.axvline(corr_pos_min, color='black')
        plt.axvline(corr_pos_max, color='black')
        plt.xlim(corr_c_off - w_multi*corr_w_size_wide, corr_c_off + w_multi*corr_w_size_wide)
        plt.ylim(np.min(corr_res[idx])-1, np.max(corr_res[idx])+1)
        plt.tight_layout()
        plt.savefig(plot_path[:-4] + '_corr1.png', dpi=200)
        plt.close()

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

    if plot:
        plt.plot(corr_res_sub, lw=1, color='C0')
        plt.axvline(corr_center_max, color='C0', label='max', ls='--')
        plt.plot(corr_fit_res.best_fit, lw=1, color='C1')
        plt.axvline(corr_center, color='C1', label='fit', ls='--')
        plt.legend()
        plt.title(u'RV max: {:.2f}, RV fit: {:.2f}, $\Delta$RV per 5: {:.2f}, center wvl {:.1f} \n chi2: {:.1f}, sigma: {:.1f}, amplitude: {:.1f}, slope: {:.2f}, $\Delta$y slope: {:.1f}'.format(np.nanmedian(rv_shifts_max), np.nanmedian(rv_shifts), np.nanmedian(rv_shifts_5), np.nanmean(obs_wvl), corr_fit_res.chisqr, corr_fit_res.params['sigma'].value, corr_fit_res.params['amplitude'].value, corr_fit_res.params['slope'].value, corr_fit_res.params['slope'].value*corr_w_size*2))
        plt.tight_layout()
        plt.savefig(plot_path[:-4] + '_corr2.png', dpi=200)
        plt.close()

    if log_shift_wvl < 5.:
        # return np.nanmedian(rv_shifts_max)
        return np.nanmedian(rv_shifts), np.nanmedian(ref_wvl), corr_fit_res.chisqr
    else:
        # something went wrong
        print('    Large wvl shift detected.')
        return np.nan, np.nanmedian(ref_wvl), np.nan


def correlate_order(obs_flx, obs_wvl,
                    ref_flx, ref_wvl,
                    order_crop_frac=0., cont_value=1.,
                    plot=False, plot_path='rv_corr_plot.png'):
    """

    :param obs_flx:
    :param obs_wvl:
    :param ref_flx:
    :param ref_wvl:
    :param order_crop_frac:
    :param cont_value:
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
        print('    Short spectral overlap, skipping order ({:.0f} - {:.0f}).'.format(wvl_beg, wvl_end))
        return np.nan, np.nanmedian(ref_wvl[idx_ref_use])

    ref_wvl_use = deepcopy(ref_wvl[idx_ref_use])
    ref_flx_use = deepcopy(ref_flx[idx_ref_use])
    # linear interpolation of observation to the same wvl bins as reference data
    obs_flx_use = np.interp(ref_wvl_use, obs_wvl, obs_flx)

    # add re-normalization step if needed
    # TODO -> already implemented somewhere else in the pipeline

    # perform correlation between the datasets
    try:
        return correlate_spectra(obs_flx_use, ref_wvl_use, ref_flx_use, ref_wvl_use,
                                 cont_value=cont_value, plot=plot, plot_path=plot_path)
    except Exception as e:
        print('    Correlation problem:', e)
        return np.nan, np.nanmedian(ref_wvl[idx_ref_use]), np.nan


def get_RV_custom_corr_perorder(exposure_data, rv_ref_flx, rv_ref_wvl,
                                rv_ref_val=None, use_flx_key='flx', cont_value=1.,
                                plot_rv=False, plot_path='rv_plot.png'):
    """

    :param exposure_data:
    :param rv_ref_flx:
    :param rv_ref_wvl:
    :param rv_ref_val:
    :param use_flx_key:
    :param cont_value:
    :param plot_rv:
    :param plot_path:
    :return:
    """
    # prepare list of per order RV determinations
    rv_shifts = list([])
    rv_fit_chi2 = list([])
    echelle_orders = _valid_orders_from_keys(exposure_data.keys())
    
    # loop trough all available orders
    for echelle_order_key in echelle_orders:
        # determine observed data that will be used in the correlation procedure
        order_flx = exposure_data[echelle_order_key][use_flx_key]
        order_wvl = exposure_data[echelle_order_key]['wvl']
        # perform correlation with reference spectrum
        rv_order_val, _, chi2_order_fit = correlate_order(order_flx, order_wvl,  # observed spectrum data
                                                          rv_ref_flx, rv_ref_wvl,  # reference spectrum data
                                                          cont_value=cont_value,
                                                          plot=False, plot_path=plot_path[:-4] + '_{:04d}.png'.format(echelle_order_key))
        rv_shifts.append(rv_order_val)
        rv_fit_chi2.append(chi2_order_fit)

    # nothing can be done if no per order RV values were determined for a given exposure/spectrum
    if len(rv_shifts) <= 0:
        return np.nan, np.nan

    # analyse obtained RV shifts
    rv_shifts = np.array(rv_shifts)
    rv_fit_chi2 = np.array(rv_fit_chi2)
    # rv_median = np.nanmedian(rv_shifts)
    # remove gross outliers before computing rv std value
    # rv_shifts[np.abs(rv_shifts - rv_median) > 10.] = np.nan
    # n_fin_rv = np.sum(np.isfinite(rv_shifts))
    # determine points with the lowest correlation fit chi2
    idx_rv_use = rv_fit_chi2 < np.nanpercentile(rv_fit_chi2, 75)  # use 3/4 of the measurements
    # cumpute initial RV guess
    rv_median_init = np.nanmedian(rv_shifts[idx_rv_use])
    # compute final RV using all sensible per order meassuremnts
    idx_rv_use_2 = np.abs(rv_shifts - rv_median_init) <= 5
    rv_median = np.nanmedian(rv_shifts[idx_rv_use_2])
    rv_std = np.nanstd(rv_shifts[idx_rv_use_2])
    print('Median RV computation:', len(idx_rv_use), np.sum(idx_rv_use), np.nanmedian(rv_shifts), np.nanmedian(rv_shifts[idx_rv_use]))
    # print(rv_median, rv_std, n_fin_rv)

    if plot_rv and np.isfinite(rv_median):
        plt.scatter(echelle_orders, rv_shifts, s=12)
        plt.scatter(np.array(echelle_orders)[~idx_rv_use], rv_shifts[~idx_rv_use], s=10, c='black', marker='x', label='Large chi2 fit')
        plt.scatter(np.array(echelle_orders)[idx_rv_use_2], rv_shifts[idx_rv_use_2], s=10, c='C1', marker='+', label='Final used')
        plt.axhline(rv_median_init, label='Init median RV', c='C3', ls='--')
        plt.axhline(rv_median, label='Final median RV', c='C1', ls='--')
        # plt.ylim(rv_median - 6.,
        #          rv_median + 6.)
        plt.xlim(np.nanmin(echelle_orders) - 25.,
                 np.nanmax(echelle_orders) + 25.)
        if rv_ref_val is not None:
            plt.axhline(rv_ref_val, label='Ref RV', c='black', ls='--')
        plt.xlabel('Echelle order [A]')
        plt.ylabel('Radial velocity [km/s]')
        plt.grid(ls='--', alpha=0.2, color='black')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=250)
        plt.close()

    # return all derived velocities and its std values
    return rv_shifts, rv_median, rv_std


def get_RV_custom_corr_combined(exposure_data, rv_ref_flx, rv_ref_wvl,
                                rv_ref_val=None, use_flx_key='flx', cont_value=1.,
                                plot_rv=False, plot_path='rv_plot.png'):
    """

    :param exposure_data:
    :param rv_ref_flx:
    :param rv_ref_wvl:
    :param rv_ref_val:
    :param use_flx_key:
    :param cont_value:
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
    flx_exposure_comb[~idx_flx_exp_valid] = cont_value

    # correlate data and get RV value
    try:
        rv_combined, _, _ = correlate_spectra(flx_exposure_comb[idx_wvl_use], rv_ref_wvl[idx_wvl_use],
                                              rv_ref_flx[idx_wvl_use], rv_ref_wvl[idx_wvl_use],
                                              cont_value=cont_value, plot=False, plot_path=plot_path)
    except Exception as e:
        print('   Combined correlation problem:', e)
        return np.nan, np.nan

    # TODO: determine uncertainty of the determined radial velocity
    return rv_combined, np.nan


def add_rv_to_metadata(star_data, star_id,
                       obs_metadata, rv_col,
                       plot=False, plot_path='plot.png'):
    """

    :param star_data:
    :param star_id:
    :param obs_metadata:
    :param rv_col:
    :param plot:
    :param plot_path:
    :return:
    """
    for exp_id in star_data.keys():
        # extract raw filename from exposure id
        filename = exp_id.split('.')[0]
        # check if we have a RV value for a selected star
        if rv_col in star_data[exp_id].keys():
            idx_meta_row = np.where(obs_metadata['filename'] == filename)[0]
            # is the same filename present in the metadata
            if len(idx_meta_row) == 1:
                # copy velocity and its error to the metadata
                obs_metadata[rv_col][idx_meta_row] = star_data[exp_id][rv_col]
                if 'e_' + rv_col in star_data[exp_id].keys():
                    # as it might not exist for some measurements (as in the case of the telluric spectrum)
                    obs_metadata['e_' + rv_col][idx_meta_row] = star_data[exp_id]['e_' + rv_col]

    if plot:
        # find rows that are relevant for the selected star
        idx_cols_plot = obs_metadata['star'] == star_id.replace('_', ' ').lower()
        # export plot if it was requested
        fig, ax = plt.subplots(1, 1)
        ax.errorbar(obs_metadata['phase'][idx_cols_plot], obs_metadata[rv_col][idx_cols_plot],
                    yerr=obs_metadata['e_' + rv_col][idx_cols_plot],
                    c='black', fmt='o', ms=1, elinewidth=0.2, lw=0)
        # add textural labels to the plot
        for rv_row in obs_metadata[idx_cols_plot]:
            if np.isfinite(rv_row[rv_col]):
                ax.text(rv_row['phase']+0.01, rv_row[rv_col], rv_row['filename'],
                        fontsize=2, va='center')
        ax.set(xlim=(-0.05, 1.05), xlabel='Orbital phase', ylabel='Radial velocity [km/s]')
        ax.grid(ls='--', alpha=0.2, color='black')
        fig.tight_layout()
        fig.savefig(plot_path, dpi=350)
        plt.close(fig)

    # return updated table with metadata about individual exposures
    return obs_metadata


def plot_rv_perorder_scatter(star_data,
                             rv_key='RV_s1', plot_path='plot.png'):
    """

    :param star_data:
    :param rv_key:
    :param plot_path:
    :return:
    """
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), sharex=True)
    y_data_all = list([])
    for exp_id in star_data.keys():
        exposure = star_data[exp_id]
        x_data = np.array(_valid_orders_from_keys(exposure.keys()))
        y_med = exposure[rv_key]
        y_data = np.array(exposure[rv_key + '_orders']) - y_med
        y_data_all.append(y_data)
        # add points to plots
        ax[0].scatter(x_data, y_data, lw=0, s=3, label=exp_id)
        ax[1].scatter(x_data, y_data, lw=0, s=3, label=exp_id)
        ax[2].scatter(x_data, y_data, lw=0, s=3, label=exp_id)
    # median rv displacement of all exposure, add it to plots
    y_data_off = np.nanmedian(y_data_all, axis=0)
    ax[0].scatter(x_data+20, y_data_off, lw=0, s=7, c='black', alpha=0.75)
    ax[1].scatter(x_data+20, y_data_off, lw=0, s=7, c='black', alpha=0.75)
    ax[2].scatter(x_data+20, y_data_off, lw=0, s=7, c='black', alpha=0.75)
    # save and configure plot
    ax[1].set(ylabel='Radial velocity [km/s]', ylim=(-25., 25.))
    ax[2].set(xlabel='Order center [A]', ylim=(-8., 8.))
    ax[0].grid(ls='--', alpha=0.2, color='black')
    ax[1].grid(ls='--', alpha=0.2, color='black')
    ax[2].grid(ls='--', alpha=0.2, color='black')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(plot_path, dpi=250)
    plt.close(fig)

    return True
