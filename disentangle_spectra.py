import numpy as np
import joblib

from rv_helper_functions import get_RV_ref_spectrum, plot_rv_perorder_scatter, add_rv_to_metadata
from disentangle_spectra_functions import plot_combined_spectrum_using_RV, show_spectra_heliocentric, run_complete_RV_and_template_discovery_procedure, get_spectral_data, create_new_reference, renorm_exposure_perorder, remove_ref_from_exposure, _go_to_dir
from copy import deepcopy, copy
from os import system, path, chdir
from astropy.table import Table
from socket import gethostname


# --------------------------------------------------------------------------------
# --------------------------- Input data setting ---------------------------------
# --------------------------------------------------------------------------------
# define initial reference spectra for the determination of radial velocity
ref_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/rv_ref/'
# read tellurics data and enable discovery of telluric radial velocity
tellurics_dir = '/shared/mari/cotar/Telluric_data/'
tellurics_data = np.loadtxt(tellurics_dir + 'telluric_spectra_conv.dat')
get_tellurics_RV = False


# --------------------------------------------------------------------------------
# ---------------------- Stars and orders data setting ---------------------------
# --------------------------------------------------------------------------------
pc_name = gethostname()
if pc_name == 'gigli':
    # data_dir = '/data4/travegre/Projects/Asiago_binaries/'  # same as mari but mounted to a different folder
    data_dir = '/media/hdd/home2/klemen/Spectroscopic-binary-decomposition/Binaries_spectra/'
else:
    data_dir = '/shared/mari/klemen/Projects/Asiago_binaries/'

# Additional reduced spectra for some stars
# data_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/

out_dir = '/media/hdd/home2/klemen/Spectroscopic-binary-decomposition/Binaries_RV/'
_go_to_dir(out_dir)

stars = ['TV_LMi', 'GZ_Dra', 'V455_Aur', 'GK_Dra', 'DT_Cam', 'V394_Vul', 'CI_CVn', 'V1898_Cyg', 'V417_Aur', 'EQ_Boo', 'V994_Her', 'CN_Lyn', 'DV_Cam'][0:15]

# initial RV reference spectrum for different stars
'''
ref_file = {
            'TV_LMi':   'T05500G40M05V000K2SNWNVR20N.fits',
            'GZ_Dra':   'T07500G40M05V075K2SNWNVR20N.fits',
            'V455_Aur': 'T07000G40M05V050K2SNWNVR20N.fits',
            'GK_Dra':   'T05500G40M05V000K2SNWNVR20N.fits',
            'DT_Cam':   'T09000G40M05V150K2SNWNVR20N.fits',
            'V394_Vul': 'T08750G40M05V150K2SNWNVR20N.fits',
            'CI_CVn':   'T08750G40M05V150K2SNWNVR20N.fits',
            'V1898_Cyg':'T15000G40M05V200K2SODNVR20N.fits',
            'V417_Aur': 'T09000G40M05V150K2SNWNVR20N.fits',
            'EQ_Boo':   'T05500G40M05V000K2SNWNVR20N.fits',
            'V994_Her': 'T10000G40M05V200K2SODNVR20N.fits',
            'CN_Lyn':   'T06500G40M05V000K2SNWNVR20N.fits',
            'DV_Cam':   'T15000G40M05V200K2SODNVR20N.fits'
            }
'''
ref_file = {
            'TV_LMi':   'T05500G40M05V000K2SNWNVR20N.fits',
            'GZ_Dra':   'T05500G40M05V000K2SNWNVR20N.fits',
            'V455_Aur': 'T05500G40M05V000K2SNWNVR20N.fits',
            'GK_Dra':   'T05500G40M05V000K2SNWNVR20N.fits',
            'DT_Cam':   'T05500G40M05V000K2SNWNVR20N.fits',
            'V394_Vul': 'T05500G40M05V000K2SNWNVR20N.fits',
            'CI_CVn':   'T05500G40M05V000K2SNWNVR20N.fits',
            'V1898_Cyg':'T05500G40M05V000K2SNWNVR20N.fits',
            'V417_Aur': 'T05500G40M05V000K2SNWNVR20N.fits',
            'EQ_Boo':   'T05500G40M05V000K2SNWNVR20N.fits',
            'V994_Her': 'T05500G40M05V000K2SNWNVR20N.fits',
            'CN_Lyn':   'T05500G40M05V000K2SNWNVR20N.fits',
            'DV_Cam':   'T05500G40M05V000K2SNWNVR20N.fits'
            }

# all possible order centers in the acquired Echelle spectra
order_centers_all = [3790, 3855, 3910, 3990, 4060, 4140, 4220, 4290, 4380, 4460, 4560, 4640, 4750, 4856, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580, 6770, 6980, 7210]

# select telluric orders only - for estimation of wavelength reduction correctness
if get_tellurics_RV:
    # select orders with a telluric absorption signature
    order_centers = [5880, 6040, 6210, 6390, 6580, 6770, 6980]
else:
    # orders to be loaded from the whole Echelle spectral range
    
    order_centers = {
                     'TV_LMi':   [4980, 5100, 5210, 5340, 5460, 5610, 5880, 6040, 6210, 6390, 6580],
                     'GZ_Dra':   [4290, 4380, 4460, 4560, 4640, 4750, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390],
                     'V455_Aur': [4290, 4380, 4460, 4560, 4640, 4750, 4980, 5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390],
                     'GK_Dra':   [4460, 4560, 4640, 4750, 4856, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390],
                     'DT_Cam':   [4290, 4380, 4460, 4560, 4640, 4750, 4980, 5210, 5340, 5460, 5610, 5730, 6040, 6210, 6390, 6580],
                     'V394_Vul': [4220, 4290, 4380, 4460, 4560, 4640, 4856, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 6210, 6390],
                     'CI_CVn':   [4220, 4290, 4380, 4460, 4560, 4640, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'V1898_Cyg':[4140, 4220, 4290, 4380, 4460, 4560, 4640, 4750, 4856, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 6040, 6210, 6390, 6580],
                     'V417_Aur': [4140, 4220, 4380, 4460, 4560, 4640, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 6040, 6210, 6390],
                     'EQ_Boo':   [4220, 4290, 4380, 4460, 4560, 4640, 4856, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210],
                     'V994_Her': [4460, 4560, 4640, 4750, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'CN_Lyn':   [4140, 4220, 4290, 4380, 4460, 4560, 4640, 4750, 4856, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 6040],
                     'DV_Cam':   [4220, 4460, 4560, 4640, 4980, 5100, 5210, 5340, 5460, 5610, 5730, 5880, 6040, 6390]
                    }
    '''
    order_centers = {
                     'TV_LMi':   [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'GZ_Dra':   [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'V455_Aur': [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'GK_Dra':   [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'DT_Cam':   [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'V394_Vul': [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'CI_CVn':   [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'V1898_Cyg':[5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'V417_Aur': [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'EQ_Boo':   [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'V994_Her': [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'CN_Lyn':   [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580],
                     'DV_Cam':   [5210, 5340, 5460, 5610, 5730, 5880, 6040, 6210, 6390, 6580]
                    }
    '''
obs_metadata = Table.read(data_dir + 'star_data_all.csv')

renorm_orders = True  # should we renormalize orders at the end of an iteration
new_spectra_only = True  # uses only newer spectra with higher SNR and better quality
combined_rv_spectrum = False  # False -> produces one RV measurement per Echelle order
fit_before_removal = False  # do we fit individual components before they are removed from the spectrum
n_rv_star_iterations = 2  # number of iterations per star
n_rv_iterations = 3  # number of iterations per component per star iteration
if get_tellurics_RV:
    n_rv_star_iterations = 1
    n_rv_iterations = 1

dump_input_data = True
additional_verbose = True
save_all_plots = True

# add additional columns to the metadata
for col in ['RV_s1', 'e_RV_s1', 'RV_s2', 'e_RV_s2', 'VHELIO', 'e_VHELIO']:
    obs_metadata[col] = np.nan


# --------------------------------------------------------------------------------
# --------------------------- Output data setting --------------------------------
# --------------------------------------------------------------------------------
results_dir_root = out_dir + 'RV_disentangle_results_RVmasking_allorders'
if new_spectra_only:
    results_dir_root += '_newonly'
else:
    results_dir_root += '_all'
if renorm_orders:
    results_dir_root += '_renorm'
if combined_rv_spectrum:
    results_dir_root += '_combRVspec'
if not fit_before_removal:
    results_dir_root += '_noremovalfit'
if get_tellurics_RV:
    results_dir_root += '_tellurics'
results_dir_root += '/'
_go_to_dir(results_dir_root)

# --------------------------------------------------------------------------------
# ------------------------- Run for every given star -----------------------------
# --------------------------------------------------------------------------------
fits_final = results_dir_root + 'final_results.fits'
for i_str, star_id in enumerate(stars):
    print('Star:', star_id)
    # load all requested spectral data for the selected star
    results_dir = results_dir_root + star_id + '/'
    _go_to_dir(results_dir)
    system('rm -R *')

    pkl_input_data = results_dir + star_id + '_input_data.pkl'
    pkl_input_data_all = results_dir + star_id + '_input_data_all.pkl'
    if path.isfile(pkl_input_data) and path.isfile(pkl_input_data_all):
        print('Reading exported input data: ' + pkl_input_data)
        star_data = joblib.load(pkl_input_data)
        star_data_all = joblib.load(pkl_input_data_all)
    else:
        star_data = get_spectral_data(star_id, order_centers[star_id], data_dir,
                                      new_only=new_spectra_only)
        star_data_all = get_spectral_data(star_id, order_centers_all, data_dir,
                                          new_only=new_spectra_only)
        # export all input data
        if dump_input_data:
            joblib.dump(star_data, pkl_input_data)
            joblib.dump(star_data_all, pkl_input_data_all)

    # add vhelio values to the observations metadata
    obs_metadata = add_rv_to_metadata(star_data, star_id,
                                      deepcopy(obs_metadata), 'VHELIO', plot=False)

    print(star_data.keys())

    # # plot spectra in heliocentric frame - only for visual inspection of reduction
    # print(' Plotting spectra in a observed frame, including telluric reference spectrum')
    # for p_ord in order_centers:
    #     show_spectra_heliocentric(star_data, p_ord,
    #                               tellurics_data=tellurics_data, prefix=str(star_id) + '_')

    # load initial reference file that will be used for initial RV determination
    if not get_tellurics_RV:
        # load synthetic stellar spectrum as a RV reference
        ref_flx_orig, ref_wvl = get_RV_ref_spectrum(ref_dir + ref_file[star_id])
    else:
        # load synthetic telluric spectrum as a RV reference
        ref_flx_orig = tellurics_data[:, 1]
        ref_wvl = tellurics_data[:, 0]
    # initial primary, secondary and tertiary reference flux
    ref_flx_s1 = np.full_like(ref_flx_orig, fill_value=1.)  # primary
    ref_flx_s2 = np.full_like(ref_flx_orig, fill_value=0.)  # secondary
    ref_flx_s3 = np.full_like(ref_flx_orig, fill_value=0.)  # tertiary
    # initial primary, secondary and tertiary reference flux for the whole Echelle range
    ref_flx_s1_all = np.full_like(ref_flx_orig, fill_value=1.)  # primary
    ref_flx_s2_all = np.full_like(ref_flx_orig, fill_value=0.)  # secondary
    ref_flx_s3_all = np.full_like(ref_flx_orig, fill_value=0.)  # tertiary

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ----------- Main part of the binary disentanglement procedure  -----------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    for i_it_star in range(n_rv_star_iterations):

        # --------------------------------------------------------------------------------
        # ---------- Radial velocity and template spectrum of a primary star -------------
        # --------------------------------------------------------------------------------
        plot_prefix_all = results_dir + star_id + '_run{:02d}'.format(i_it_star + 1)
        for i_it in range(n_rv_iterations):

            # determine which reference spectrum will be used - depending on star run and iteration run number
            input_flx_key = 'flx'
            if i_it_star <= 0 and i_it <= 0:
                ref_flx_use1 = copy(ref_flx_orig)
            else:
                ref_flx_use1 = copy(ref_flx_s1)
                if renorm_orders and i_it_star > 0:
                    input_flx_key += '_renorm'

            # --------------------------------------------------------------------------------
            # ------------ Initial guess for a spectrum of a primary star --------------------
            # --------------------------------------------------------------------------------
            print(' Primary star run {:02d}, iteration {:02d}.'.format(i_it_star + 1, i_it + 1))

            # removal should be performed only at the first iteration as the results is the same for all
            # consequent iterations of the same component
            if i_it == 0:
                # start recovering flx of a secondary component in a multiple system
                for exp_id in star_data.keys():
                    print('  Removing secondary component of:', exp_id)
                    # remove reference/median flux from all acquired spectra
                    ref_removal_png = plot_prefix_all + '_' + exp_id + '_s1_removal.png'
                    star_exposure_new = remove_ref_from_exposure(deepcopy(star_data[exp_id]),
                                                                 ref_flx_s2, ref_wvl,
                                                                 primary=True,
                                                                 use_rv_key='RV_s2',
                                                                 input_flx_key=input_flx_key,
                                                                 fit_before_removal=(fit_before_removal and i_it_star > 0),
                                                                 output_flx_key='flx1',
                                                                 ref_orig=None, w_filt=None,
                                                                 plot=True, plot_path=ref_removal_png,
                                                                 verbose=additional_verbose)
                    star_data[exp_id] = star_exposure_new

            # run procedure for a first, most evident spectral component in a spectrum
            star_data, obs_metadata, ref_flx_s1 = run_complete_RV_and_template_discovery_procedure(star_data, obs_metadata,
                                                                                                   ref_flx_use1, ref_wvl,
                                                                                                   star_id=star_id,
                                                                                                   in_flx_key='flx1',
                                                                                                   rv_key='RV_s1',
                                                                                                   primary=True,
                                                                                                   combined_rv_spectrum=combined_rv_spectrum,
                                                                                                   plot_prefix=plot_prefix_all,
                                                                                                   plot_suffix='_it{:02d}'.format(i_it+1),
                                                                                                   save_plots=save_all_plots,
                                                                                                   verbose=additional_verbose)

            if (not combined_rv_spectrum) and save_all_plots:
                plot_rv_perorder_scatter(star_data, rv_key='RV_s1',
                                         plot_path=plot_prefix_all + '_it{:02d}_RV1_scatter.png'.format(i_it+1))

            # remove bad RV orders
            if i_it > 0 or i_it_star > 0:
                pass

        # --------------------------------------------------------------------------------
        # ------------ Initial guess for a spectrum of a secondary star ------------------
        # --------------------------------------------------------------------------------

        # start recovering flx of a secondary component in a multiple system
        for exp_id in star_data.keys():
            print('  Removing primary component of:', exp_id)

            # remove reference/median flux from all acquired spectra
            ref_removal_png = plot_prefix_all + '_' + exp_id + '_s2_removal.png'
            star_exposure_new = remove_ref_from_exposure(deepcopy(star_data[exp_id]),
                                                         ref_flx_s1, ref_wvl,
                                                         primary=False,
                                                         use_rv_key='RV_s1',
                                                         input_flx_key=input_flx_key,
                                                         fit_before_removal=fit_before_removal,
                                                         output_flx_key='flx2',
                                                         ref_orig=None, w_filt=None,
                                                         plot=True, plot_path=ref_removal_png,
                                                         verbose=additional_verbose)
            star_data[exp_id] = star_exposure_new

        # --------------------------------------------------------------------------------
        # --------- Radial velocity and template spectrum of a secondary star ------------
        # --------------------------------------------------------------------------------
        for i_it in range(n_rv_iterations):

            print(' Secondary star run {:02d}, iteration {:02d}.'.format(i_it_star + 1, i_it + 1))
            # determine which reference spectrum will be used - depending on star run and iteration run number
            if i_it_star <= 0 and i_it <= 0:
                ref_flx_use2 = copy(ref_flx_orig) - 1.
            else:
                ref_flx_use2 = copy(ref_flx_s2)
            # run procedure for a first, most evident spectral component in a spectrum
            star_data, obs_metadata, ref_flx_s2 = run_complete_RV_and_template_discovery_procedure(star_data, obs_metadata,
                                                                                                   ref_flx_use2, ref_wvl,
                                                                                                   star_id=star_id,
                                                                                                   in_flx_key='flx2',
                                                                                                   rv_key='RV_s2',
                                                                                                   primary=False,
                                                                                                   combined_rv_spectrum=combined_rv_spectrum,
                                                                                                   plot_prefix=plot_prefix_all,
                                                                                                   plot_suffix='_it{:02d}'.format(i_it+1),
                                                                                                   save_plots=save_all_plots,
                                                                                                   verbose=additional_verbose)

            if (not combined_rv_spectrum) and save_all_plots:
                plot_rv_perorder_scatter(star_data, rv_key='RV_s2',
                                         plot_path=plot_prefix_all + '_it{:02d}_RV2_scatter.png'.format(i_it+1))

        # --------------------------------------------------------------------------------
        # ------------------- Renormalization of original input data ---------------------
        # --------------------------------------------------------------------------------

        # renormalize original Echelle spectrum orders at every iteration if needed
        if renorm_orders:
            ref_flx_renorm = copy(ref_flx_s1)
            # always use original data for the renormalization process
            input_flx_key_renorm = 'flx'
            # begin per order normalization process using the reference spectrum
            for exp_id in star_data.keys():
                if additional_verbose:
                    print('  Renormalizing exposure:', exp_id)

                # renorm using determined RV and make plots
                exp_renorm_png = plot_prefix_all + '_' + exp_id + '_renorm.png'
                star_exposure_new = renorm_exposure_perorder(deepcopy(star_data[exp_id]), ref_flx_renorm, ref_wvl,
                                                             use_rv_key='RV_s1',
                                                             input_flx_key=input_flx_key_renorm,
                                                             output_flx_key=input_flx_key_renorm + '_renorm',
                                                             plot=save_all_plots, plot_path=exp_renorm_png)
                star_data[exp_id] = star_exposure_new

        # --------------------------------------------------------------------------------
        # ----- Plot combined primary and secondary component with original spectrum  ----
        # --------------------------------------------------------------------------------
        if save_all_plots:
            for exp_id in star_data.keys():
                components_png = plot_prefix_all + '_' + exp_id + '_components.png'
                plot_combined_spectrum_using_RV(deepcopy(star_data[exp_id]),
                                                ref_flx_s1, ref_flx_s2, ref_wvl,
                                                prim_rv='RV_s1', sec_rv='RV_s2', input_flx_key=input_flx_key,
                                                plot=True, plot_path=components_png)

        # --------------------------------------------------------------------------------
        # ------------ Create combined spectrum of a primary star and secondary over the whole wavelength range
        # --------------------------------------------------------------------------------
        print(' Preparing and exporting spectra over the whole spectral range.')
        # determine which reference spectrum will be used - depending on star run and iteration run number
        input_flx_key_all = 'flx'
        if i_it_star > 0:
            input_flx_key_all += '_renorm'

        # firstly copy over RV results from analysed orders to complete spectral database
        for exp_id in star_data_all.keys():
            for c_val in ['RV_s1', 'e_RV_s1', 'RV_s2', 'e_RV_s2']:
                star_data_all[exp_id][c_val] = star_data[exp_id][c_val]

        # remove secondary spectrum from observed spectra to obtain individual primary spectra
        for exp_id in star_data_all.keys():
            star_exposure_new = remove_ref_from_exposure(deepcopy(star_data_all[exp_id]),
                                                         ref_flx_s2_all, ref_wvl,
                                                         primary=True,
                                                         use_rv_key='RV_s2',
                                                         input_flx_key=input_flx_key_all,
                                                         output_flx_key='flx1',
                                                         verbose=additional_verbose)
            star_data_all[exp_id] = star_exposure_new

        # create primary spectrum
        ref_flx_s1_all, _ = create_new_reference(star_data_all, ref_wvl,
                                                 use_flx_key='flx1', use_rv_key='RV_s1')

        # remove primary spectrum from observed spectra to obtain individual secondary spectra
        for exp_id in star_data_all.keys():
            star_exposure_new = remove_ref_from_exposure(deepcopy(star_data_all[exp_id]),
                                                         ref_flx_s1_all, ref_wvl,
                                                         primary=False,
                                                         use_rv_key='RV_s1',
                                                         input_flx_key=input_flx_key_all,
                                                         output_flx_key='flx2',
                                                         verbose=additional_verbose)
            star_data_all[exp_id] = star_exposure_new

        # create secondary spectrum
        ref_flx_s2_all, _ = create_new_reference(star_data_all, ref_wvl,
                                                 use_flx_key='flx2', use_rv_key='RV_s2')

        # renorm original spectra using median primary spectrum
        for exp_id in star_data_all.keys():
            star_exposure_new = renorm_exposure_perorder(deepcopy(star_data_all[exp_id]), copy(ref_flx_s1_all), ref_wvl,
                                                         use_rv_key='RV_s1',
                                                         input_flx_key='flx',
                                                         output_flx_key='flx_renorm')
            star_data_all[exp_id] = star_exposure_new

        # export primary and secondary spectrum as csv
        primary_txt = open(f'spec_primary_{star_id}.txt', 'w')
        secondary_txt = open(f'spec_secondary_{star_id}.txt', 'w')
        for i_w, r_w in enumerate(ref_wvl):
            primary_txt.write('{:.4f} {:.3f}\n'.format(ref_flx_s1_all[i_w], r_w))
            secondary_txt.write('{:.5f} {:.3f}\n'.format(ref_flx_s2_all[i_w], r_w))
        primary_txt.close()
        secondary_txt.close()

    # --------------------------------------------------------------------------------
    # ------------ Export results for a given star -----------------------------------
    # --------------------------------------------------------------------------------
    star_obs_metadata = obs_metadata[obs_metadata['star'] == star_id.replace('_', ' ').lower()]
    star_obs_metadata = star_obs_metadata[np.isfinite(star_obs_metadata['RV_s1'])]
    np.savetxt(f'RV_primary_{star_id}.txt', star_obs_metadata['JD', 'RV_s1', 'e_RV_s1'].to_pandas().values, fmt=['%.8f', '%.5f', '%.5f'])
    np.savetxt(f'RV_secondary_{star_id}.txt', star_obs_metadata['JD', 'RV_s2', 'e_RV_s2'].to_pandas().values, fmt=['%.8f', '%.5f', '%.5f'])
    chdir('..')
    obs_metadata.write(fits_final, overwrite=True)

'''
        if dump_input_data:
            # export all data and products after the extraction of primary spectrum
            joblib.dump(star_data, pkl_input_data_s1)
            obs_metadata.write(fits_input_data_s1, overwrite=True)
'''
