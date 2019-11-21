import numpy as np
import matplotlib.pyplot as plt
import joblib

from rv_helper_functions import get_RV_ref_spectrum, get_RV_custom_corr_perorder
from disentangle_spectra_functions import get_spectral_data, create_new_reference
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
order_centers = [4980, 5100, 5210, 5340, 5460, 5610]
obs_metadata = Table.read(root_dir + 'star_data_all.csv')
dump_input_data = True
# add additional columns to the metadata
for col in ['RV_s1', 'e_RV_s1', 'RV_s2', 'e_RV_s2']:
    obs_metadata[col] = np.nan


# --------------------------------------------------------------------------------
# --------------------------- Output data setting --------------------------------
# --------------------------------------------------------------------------------
results_dir = root_dir + 'RV_disentangle_results/'
system('mkdir ' + results_dir)


for star_id in stars[:1]:
    print('Star:', star_id)
    # load all requested spectral data for the selected star
    pkl_input_data = results_dir + star_id + '_input_data.pkl'
    if path.isfile(pkl_input_data):
        print('Reading exported input data: ' + pkl_input_data)
        star_data = joblib.load(pkl_input_data)
    else:
        star_data = get_spectral_data(star_id, order_centers, root_dir)
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
        # recreate new reference spectrum of a primary
        ref_flx = create_new_reference(star_data, ref_wvl,
                                       use_flx_key='flx', use_rv_key='RV_s1')
    else:
        n_rv_iterations = 6
        for i_it in range(n_rv_iterations):
            print(' Primary star iteration', i_it+1)
            # get per order RV velocities for every exposure
            for exp_id in star_data.keys():
                print('  Exposure:', exp_id)
                star_exposure = deepcopy(star_data[exp_id])

                # compute mean RV from all considered orders
                rv_png = results_dir + star_id + '_' + exp_id + '_rv-orders_{:02d}.png'.format(i_it+1)
                rv_med, rv_std = get_RV_custom_corr_perorder(star_exposure, ref_flx, ref_wvl,
                                                             rv_ref_val=None, use_flx_key='flx',
                                                             plot_rv=True, plot_path=rv_png)
                print('   RV value:', rv_med, rv_std)

                # store values to the dictionary
                star_data[exp_id]['RV_s1'] = rv_med
                star_data[exp_id]['e_RV_s1'] = rv_std

            # compute median spectrum of a secondary star and use it as a new and updated RV template
            combined_png = results_dir + star_id + '_s1_combined_{:02d}.png'.format(i_it + 1)
            ref_flx = create_new_reference(star_data, ref_wvl,
                                           use_flx_key='flx', use_rv_key='RV_s1',
                                           plot_combined=True, plot_path=combined_png)

        if dump_input_data:
            # export all data and products after the extraction of primary spectrum
            joblib.dump(star_data, pkl_input_data_s1)

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
    ax.scatter(obs_metadata['phase'][idx_cols_plot], obs_metadata['RV_s1'][idx_cols_plot],
                c='black', lw=0, s=4)
    ax.set(xlim=(-0.05, 1.05), xlabel='Orbital phase', ylabel='Radial velocity')
    fig.tight_layout()
    fig.savefig(results_dir + star_id + '_RV_primary.png', dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ------------ Radial velocity and spectrum of a secondary star ------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # TODO
