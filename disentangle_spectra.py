import numpy as np
import matplotlib.pyplot as plt
import joblib

from rv_helper_functions import get_RV_ref_spectrum, get_RV_custom_corr_perorder, get_RV_custom_corr_combined, add_rv_to_metadata
from disentangle_spectra_functions import get_spectral_data, create_new_reference, renorm_exposure_perorder, remove_ref_from_exposure
from copy import deepcopy
from os import system, path
from astropy.table import Table


# --------------------------------------------------------------------------------
# --------------------------- Input data setting ---------------------------------
# --------------------------------------------------------------------------------
ref_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/rv_ref/'
ref_file = ['T06500G40M05V000K2SNWNVR20N.fits',
            'T05500G40M05V000K2SNWNVR20N.fits']

# --------------------------------------------------------------------------------
# ---------------------- Stars and orders data setting ---------------------------
# --------------------------------------------------------------------------------
root_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/'
stars = ['GZ_Dra', 'TV_LMi']
# order_centers = [4980, 5100, 5210, 5340, 5460, 5610]
order_centers = [5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580]
obs_metadata = Table.read(root_dir + 'star_data_all.csv')

renorm_orders = True
new_spectra_only = True
combined_rv_spectrum = False  # False -> produces one RV measurement per Echelle order
n_rv_iterations = 5  # number of iterations per component
n_rv_star_iterations = 4  # number of iterations per star

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


for i_str, star_id in enumerate(stars):
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
    ref_flx, ref_wvl = get_RV_ref_spectrum(ref_dir + ref_file[i_str])
    ref_flx_orig = deepcopy(ref_flx)
    # initial secondary reference flux
    ref_flx_sec = np.full_like(ref_flx_orig, fill_value=0.)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ------------ Radial velocity and spectrum of a primary star --------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    pkl_input_data_s1 = results_dir + star_id + '_input_data_s1.pkl'
    fits_input_data_s1 = results_dir + star_id + '_input_data_s1.fits'
    if path.isfile(pkl_input_data_s1):
        print('Reading exported data of a primary star: ' + pkl_input_data)
        star_data = joblib.load(pkl_input_data_s1)
        obs_metadata = Table.read(fits_input_data_s1)

        use_flx_key = 'flx'
        if renorm_orders:
            # use renormalized orders (per order and not global normalization)
            use_flx_key += '_renorm'

        # recreate new reference spectrum of a primary
        ref_flx, _ = create_new_reference(star_data, ref_wvl,
                                          #percentile=85.,
                                          w_filt=15,
                                          use_flx_key=use_flx_key, use_rv_key='RV_s1')
    else:
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
                    rv_png = results_dir + star_id + '_' + exp_id + '_rv1-combined_{:02d}.png'.format(i_it+1)
                    rv_med, rv_std = get_RV_custom_corr_combined(deepcopy(star_data[exp_id]), ref_flx, ref_wvl,
                                                                 rv_ref_val=None, use_flx_key=use_flx_key,
                                                                 plot_rv=True, plot_path=rv_png)
                    print('   Combined RV value:', rv_med, rv_std)
                else:
                    # compute mean RV from all considered orders
                    rv_png = results_dir + star_id + '_' + exp_id + '_rv1-orders_{:02d}.png'.format(i_it+1)
                    rv_med, rv_std = get_RV_custom_corr_perorder(deepcopy(star_data[exp_id]), ref_flx, ref_wvl,
                                                                 rv_ref_val=None, use_flx_key=use_flx_key,
                                                                 plot_rv=True, plot_path=rv_png)
                    print('   Median RV value:', rv_med, rv_std)

                # store values to the dictionary
                star_data[exp_id]['RV_s1'] = rv_med
                star_data[exp_id]['e_RV_s1'] = rv_std

            # renormalize original Echell spectrum orders at every iteration
            if renorm_orders:
                for exp_id in star_data.keys():
                    print('  Renormalizing exposure:', exp_id)

                    # compute mean RV from all considered orders
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
            ref_flx, _ = create_new_reference(star_data, ref_wvl,
                                              #percentile=85.,
                                              w_filt=15,
                                              use_flx_key=use_flx_key_median, use_rv_key='RV_s1',
                                              plot_combined=True, plot_shifted=True,
                                              plot_path=combined_png)

            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------
            # ----------- Plot primary radial velocity as a function of a phase --------------
            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------

            # add RV values of a binary star to the observations metadata table and plot phase RV diagram
            rv_phase_plot_png = results_dir + star_id + '_RV_primary_{:02d}.png'.format(i_it + 1)
            obs_metadata = add_rv_to_metadata(star_data, star_id,
                                              deepcopy(obs_metadata), 'RV_s1',
                                              plot=True, plot_path=rv_phase_plot_png)

        if dump_input_data:
            # export all data and products after the extraction of primary spectrum
            joblib.dump(star_data, pkl_input_data_s1)
            obs_metadata.write(fits_input_data_s1, overwrite=True)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ------------ Initial guess for a spectrum of a secondary star ------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    input_flx_key = 'flx'
    if renorm_orders:
        input_flx_key += '_renorm'

    # start recovering flx of a secondary component in a multiple system
    for exp_id in star_data.keys():
        print('  Removing primary component of:', exp_id)

        # remove reference/median flux from all acquired spectra
        ref_removal_png = results_dir + star_id + '_' + exp_id + '_s2_removal.png'
        star_exposure_new = remove_ref_from_exposure(deepcopy(star_data[exp_id]), ref_flx, ref_wvl,
                                                     use_rv_key='RV_s1',
                                                     input_flx_key=input_flx_key,
                                                     output_flx_key='flx_secon',
                                                     ref_orig=ref_flx_orig, w_filt=3,
                                                     plot=True, plot_path=ref_removal_png)
        star_data[exp_id] = star_exposure_new

    print(' Creating secondary RV reference spectrum')
    ref_flx_sec = deepcopy(ref_flx_orig) - 1.

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # -------------------- Radial velocity of a secondary star -----------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    pkl_input_data_s2 = results_dir + star_id + '_input_data_s2.pkl'
    fits_input_data_s2 = results_dir + star_id + '_input_data_s2.fits'
    if path.isfile(pkl_input_data_s2):
        pass
        # print('Reading exported data of a secondary star: ' + pkl_input_data)
        # star_data = joblib.load(pkl_input_data_s1)
        #
        # use_flx_key = 'flx'
        # if renorm_orders:
        #     # use renormalized orders (per order and not global normalization)
        #     use_flx_key += '_renorm'
        #
        # # recreate new reference spectrum of a primary
        # ref_flx, _ = create_new_reference(star_data, ref_wvl,
        #                                use_flx_key=use_flx_key, use_rv_key='RV_s1')
    else:
        for i_it in range(n_rv_iterations):
            print(' Secondary star iteration', i_it + 1)

            use_flx_key = 'flx_secon'
            # if renorm_orders and i_it > 0:
            #     # use renormalized orders (per order and not global normalization)
            #     use_flx_key += '_renorm'

            # get per order RV velocities for every exposure
            for exp_id in star_data.keys():
                print('  Exposure:', exp_id)

                if combined_rv_spectrum:
                    # compute RV from a combined spectrum (stack of individual echelle orders)
                    rv_png = results_dir + star_id + '_' + exp_id + '_rv2-combined_{:02d}.png'.format(i_it+1)
                    rv_med, rv_std = get_RV_custom_corr_combined(deepcopy(star_data[exp_id]), ref_flx_sec, ref_wvl,
                                                                 cont_value=0.,
                                                                 rv_ref_val=None, use_flx_key=use_flx_key,
                                                                 plot_rv=True, plot_path=rv_png)
                    print('   Combined secondary RV value:', rv_med, rv_std)
                else:
                    # compute mean RV from all considered orders
                    rv_png = results_dir + star_id + '_' + exp_id + '_rv2-orders_{:02d}.png'.format(i_it+1)
                    rv_med, rv_std = get_RV_custom_corr_perorder(deepcopy(star_data[exp_id]), ref_flx_sec, ref_wvl,
                                                                 cont_value=0.,
                                                                 rv_ref_val=None, use_flx_key=use_flx_key,
                                                                 plot_rv=True, plot_path=rv_png)
                    print('   Median secondary RV value:', rv_med, rv_std)

                # store values to the dictionary
                star_data[exp_id]['RV_s2'] = rv_med
                star_data[exp_id]['e_RV_s2'] = rv_std
            # compute median spectrum of a secondary star and use it as a new and updated RV template

            use_flx_key_median = 'flx_secon'
            # if renorm_orders:
            #     # use renormalized orders (per order and not global normalization)
            #     use_flx_key_median += '_renorm'
            print(' Creating median secondary spectrum', i_it + 1)
            combined_png = results_dir + star_id + '_s2_combined_{:02d}.png'.format(i_it + 1)
            ref_flx_sec, _ = create_new_reference(star_data, ref_wvl,
                                                  use_flx_key=use_flx_key_median, use_rv_key='RV_s2',
                                                  plot_combined=True, plot_shifted=True,
                                                  plot_path=combined_png)

            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------
            # ----------- Plot combined radial velocity as a function of a phase -------------
            # --------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------

            # add RV values of a binary star to the observations metadata table and plot phase RV diagram
            rv_phase_plot_png = results_dir + star_id + '_RV_secondarx_{:02d}.png'.format(i_it + 1)
            obs_metadata = add_rv_to_metadata(star_data, star_id,
                                              deepcopy(obs_metadata), 'RV_s2',
                                              plot=True, plot_path=rv_phase_plot_png)

