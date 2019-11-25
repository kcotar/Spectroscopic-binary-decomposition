import numpy as np
import matplotlib.pyplot as plt
import joblib

from rv_helper_functions import get_RV_ref_spectrum, get_RV_custom_corr_perorder, get_RV_custom_corr_combined
from disentangle_spectra_functions import get_spectral_data, create_new_reference, renorm_exposure_perorder
from copy import deepcopy
from os import system, path
from astropy.table import Table

# --------------------------------------------------------------------------------
# --------------------------- Input data setting ---------------------------------
# --------------------------------------------------------------------------------
ref_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/rv_ref/'
ref_file = 'T05500G40P00V000K2SNWNVR20N.fits'

# --------------------------------------------------------------------------------
# ---------------------- Stars and orders data setting ---------------------------
# --------------------------------------------------------------------------------
root_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/'
stars = ['GZ_Dra', 'TV_LMi']
# order_centers = [4980, 5100, 5210, 5340, 5460, 5610]
order_centers = [5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580]
obs_metadata = Table.read(root_dir + 'star_data_all.csv')

renorm_orders = True
new_spectra_only = False
combined_rv_spectrum = True

dump_input_data = True
# add additional columns to the metadata
for col in ['RV_s1', 'e_RV_s1', 'RV_s2', 'e_RV_s2']:
    obs_metadata[col] = np.nan


# --------------------------------------------------------------------------------
# --------------------------- Output data setting --------------------------------
# --------------------------------------------------------------------------------
results_dir = root_dir + 'RV_disentangle_results'
if new_spectra_only:
    results_dir += '_newonly'
else:
    results_dir += '_all'
if renorm_orders:
    results_dir += '_renorm'
if combined_rv_spectrum:
    results_dir += '_combRVspec'
results_dir += '/'
system('mkdir ' + results_dir)


for star_id in stars:
    print('Star:', star_id)
    # load all requested spectral data for the selected star
    pkl_input_data = results_dir + star_id + '_input_data.pkl'
    if path.isfile(pkl_input_data):
        print('Reading exported input data: ' + pkl_input_data)
        star_data = joblib.load(pkl_input_data)
    else:
        star_data = get_spectral_data(star_id, order_centers, root_dir,
                                      new_only=new_spectra_only)
        # export all input data
        if dump_input_data:
            joblib.dump(star_data, pkl_input_data)
    print(star_data.keys())

    # load initial reference file that will be used for initial RV determination
    ref_flx, ref_wvl = get_RV_ref_spectrum(ref_dir + ref_file)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ------------ Radial velocity and spectrum of a primary star --------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    pkl_input_data_s1 = results_dir + star_id + '_input_data_s1.pkl'
    if path.isfile(pkl_input_data_s1):
        print('Reading exported data of a primary star: ' + pkl_input_data)
        star_data = joblib.load(pkl_input_data_s1)

        use_flx_key = 'flx'
        if renorm_orders:
            # use renormalized orders (per order and not global normalization)
            use_flx_key += '_renorm'

        # recreate new reference spectrum of a primary
        ref_flx = create_new_reference(star_data, ref_wvl,
                                       use_flx_key=use_flx_key, use_rv_key='RV_s1')
    else:
        n_rv_iterations = 6
        for i_it in range(n_rv_iterations):
            print(' Primary star iteration', i_it + 1)

            use_flx_key = 'flx'
            if renorm_orders and i_it > 0:
                # use renormalized orders (per order and not global normalization)
                use_flx_key += '_renorm'

            # get per order RV velocities for every exposure
            for exp_id in star_data.keys():
                print('  Exposure:', exp_id)

                if combined_rv_spectrum:
                    # compute RV from a combined spectrum (stack of individual echelle orders)
                    rv_png = results_dir + star_id + '_' + exp_id + '_rv-combined_{:02d}.png'.format(i_it+1)
                    rv_med, rv_std = get_RV_custom_corr_combined(deepcopy(star_data[exp_id]), ref_flx, ref_wvl,
                                                                rv_ref_val=None, use_flx_key=use_flx_key,
                                                                plot_rv=True, plot_path=rv_png)
                    print('   Combined RV value:', rv_med, rv_std)
                else:
                    # compute mean RV from all considered orders
                    rv_png = results_dir + star_id + '_' + exp_id + '_rv-orders_{:02d}.png'.format(i_it+1)
                    rv_med, rv_std = get_RV_custom_corr_perorder(deepcopy(star_data[exp_id]), ref_flx, ref_wvl,
                                                                 rv_ref_val=None, use_flx_key=use_flx_key,
                                                                 plot_rv=True, plot_path=rv_png)
                    print('   Median RV value:', rv_med, rv_std)

                # store values to the dictionary
                star_data[exp_id]['RV_s1'] = rv_med
                star_data[exp_id]['e_RV_s1'] = rv_std

            # renormalize original echell spectrum orders at every iteration
            if renorm_orders:
                for exp_id in star_data.keys():
                    print('  Renormalizing exposure:', exp_id)

                    # compute mean RV from all considered orders
                    rv_png = results_dir + star_id + '_' + exp_id + '_rv-orders_{:02d}.png'.format(i_it+1)
                    star_exposure_new = renorm_exposure_perorder(deepcopy(star_data[exp_id]), ref_flx, ref_wvl,
                                                                 use_rv_key='RV_s1',
                                                                 input_flx_key='flx',
                                                                 output_flx_key='flx_renorm',
                                                                 plot=False, plot_path=None)
                    star_data[exp_id] = star_exposure_new

            # compute median spectrum of a secondary star and use it as a new and updated RV template
            use_flx_key_median = 'flx'
            if renorm_orders:
                # use renormalized orders (per order and not global normalization)
                use_flx_key_median += '_renorm'
            print(' Creating median spectrum', i_it + 1)
            combined_png = results_dir + star_id + '_s1_combined_{:02d}.png'.format(i_it + 1)
            ref_flx = create_new_reference(star_data, ref_wvl,
                                           use_flx_key=use_flx_key_median, use_rv_key='RV_s1',
                                           plot_combined=True, plot_shifted=True,
                                           plot_path=combined_png)            

            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------
            # ----------- Plot primary radial velocity as a function of a phase --------------
            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------

            # add RV values of a binary star to the observations metadata table
            prim_rv_key = 'RV_s1'
            for exp_id in star_data.keys():
                # extract raw filename from exposure id
                filename = exp_id.split('.')[0]
                # check if we have a RV value for a selected star
                if prim_rv_key in star_data[exp_id].keys():
                    idx_meta_row = np.where(obs_metadata['filename'] == filename)[0]
                    # is the same filename present in the metadata
                    if len(idx_meta_row) == 1:
                        # copy velocity and its error to the metadata
                        obs_metadata[prim_rv_key][idx_meta_row] = star_data[exp_id][prim_rv_key]
                        obs_metadata['e_' + prim_rv_key][idx_meta_row] = star_data[exp_id]['e_' + prim_rv_key]

            idx_cols_plot = obs_metadata['star'] == star_id.replace('_', ' ').lower()
            fig, ax = plt.subplots(1, 1)
            ax.errorbar(obs_metadata['phase'][idx_cols_plot], obs_metadata['RV_s1'][idx_cols_plot],
                        yerr=obs_metadata['e_RV_s1'][idx_cols_plot],
                        c='black', fmt='o', ms=1, elinewidth=0.2, lw=0)
            ax.set(xlim=(-0.05, 1.05), xlabel='Orbital phase', ylabel='Radial velocity')
            ax.grid(ls='--', alpha=0.2, color='black')
            fig.tight_layout()
            fig.savefig(results_dir + star_id + '_RV_primary_{:02d}.png'.format(i_it + 1), dpi=300)
            plt.close(fig)

        if dump_input_data:
            # export all data and products after the extraction of primary spectrum
            joblib.dump(star_data, pkl_input_data_s1)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ------------ Radial velocity and spectrum of a secondary star ------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # TODO:
