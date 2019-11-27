import numpy as np
import joblib

from rv_helper_functions import get_RV_ref_spectrum, get_RV_custom_corr_perorder, get_RV_custom_corr_combined, add_rv_to_metadata
from disentangle_spectra_functions import run_complete_RV_and_template_discovery_procedure, get_spectral_data, create_new_reference, renorm_exposure_perorder, remove_ref_from_exposure
from copy import deepcopy, copy
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
n_rv_star_iterations = 4  # number of iterations per star
n_rv_iterations = 3  # number of iterations per component per star iteration

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


# --------------------------------------------------------------------------------
# ------------------------- Run for every given star -----------------------------
# --------------------------------------------------------------------------------
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
    ref_flx_orig, ref_wvl = get_RV_ref_spectrum(ref_dir + ref_file[i_str])
    # initial primary, secondary and terciary reference flux
    ref_flx_s1 = np.full_like(ref_flx_orig, fill_value=1.)
    ref_flx_s2 = np.full_like(ref_flx_orig, fill_value=0.)
    ref_flx_s3 = np.full_like(ref_flx_orig, fill_value=0.)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ------------ Main part of the binary disentaglement program  -------------------
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
        ref_flx_s1, _ = create_new_reference(star_data, ref_wvl,
                                             #percentile=85.,
                                             w_filt=15,
                                             use_flx_key=use_flx_key, use_rv_key='RV_s1')
    else:
        for i_it_star in range(n_rv_star_iterations):

            # --------------------------------------------------------------------------------
            # ---------- Radial velocity and template spectrum of a primary star -------------
            # --------------------------------------------------------------------------------
            for i_it in range(n_rv_iterations):

                print(' Primary star run {:02d}, iteration {:02d}.'.format(i_it_star + 1, i_it + 1))                
                # determine which reference spectrum will be used - depending on star run and iterration run number
                if i_it_star <= 0:
                    if i_it <= 0:
                        ref_flx_use = copy(ref_flx_orig)
                    else:
                        ref_flx_use = copy(ref_flx_s1)
                else:
                    ref_flx_use = copy(ref_flx_s1 - ref_flx_s2)
                # run procedure for a first, most evident spectral component in a spectrum
                star_data, obs_metadata, ref_flx_s1 = run_complete_RV_and_template_discovery_procedure(star_data, obs_metadata,
                                                                                                       ref_flx_use, ref_wvl,
                                                                                                       star_id=star_id,
                                                                                                       in_flx_key = 'flx', 
                                                                                                       rv_key='RV_s1',
                                                                                                       primary=True,
                                                                                                       renorm_orders=renorm_orders, 
                                                                                                       combined_rv_spectrum=combined_rv_spectrum,
                                                                                                       plot_prefix=results_dir + star_id + '_run{:02d}'.format(i_it_star+1),
                                                                                                       plot_suffix='_it{:02d}'.format(i_it+1),
                                                                                                       save_plots=True,
                                                                                                       verbose=True)

            # --------------------------------------------------------------------------------
            # ------------ Initial guess for a spectrum of a secondary star ------------------
            # --------------------------------------------------------------------------------
            input_flx_key = 'flx'
            if renorm_orders:
                input_flx_key += '_renorm'

            # start recovering flx of a secondary component in a multiple system
            for exp_id in star_data.keys():
                print('  Removing primary component of:', exp_id)

                # remove reference/median flux from all acquired spectra
                ref_removal_png = results_dir + star_id + '_' + exp_id + '_s2_removal.png'
                star_exposure_new = remove_ref_from_exposure(deepcopy(star_data[exp_id]), 
                                                             ref_flx_s1 - ref_flx_s2, ref_wvl,
                                                             use_rv_key='RV_s1',
                                                             input_flx_key=input_flx_key,
                                                             output_flx_key='flx2',
                                                             ref_orig=ref_flx_orig, w_filt=3,
                                                             plot=True, plot_path=ref_removal_png)
                star_data[exp_id] = star_exposure_new

            # --------------------------------------------------------------------------------
            # ---------- Radial velocity and template spectrum of a primary star -------------
            # --------------------------------------------------------------------------------
            for i_it in range(n_rv_iterations):                

                print(' Secondary star run {:02d}, iteration {:02d}.'.format(i_it_star + 1, i_it + 1))                
                # determine which reference spectrum will be used - depending on star run and iterration run number
                if i_it_star <= 0:
                    if i_it <= 0:
                        ref_flx_use = copy(ref_flx_orig) - 1.
                    else:
                        ref_flx_use = copy(ref_flx_s2)
                else:
                    ref_flx_use = copy(ref_flx_s2 - ref_flx_s3)
                # run procedure for a first, most evident spectral component in a spectrum
                star_data, obs_metadata, ref_flx_s2 = run_complete_RV_and_template_discovery_procedure(star_data, obs_metadata,
                                                                                                       ref_flx_use, ref_wvl,
                                                                                                       star_id=star_id,
                                                                                                       in_flx_key = 'flx2', 
                                                                                                       rv_key='RV_s2',
                                                                                                       primary=False,
                                                                                                       renorm_orders=False,  # must always be set to false to be compatible with flx 
                                                                                                       combined_rv_spectrum=combined_rv_spectrum,
                                                                                                       plot_prefix=results_dir + star_id + '_run{:02d}'.format(i_it_star+1),
                                                                                                       plot_suffix='_it{:02d}'.format(i_it+1),
                                                                                                       save_plots=True,
                                                                                                       verbose=True)

'''

        if dump_input_data:
            # export all data and products after the extraction of primary spectrum
            joblib.dump(star_data, pkl_input_data_s1)
            obs_metadata.write(fits_input_data_s1, overwrite=True)
'''