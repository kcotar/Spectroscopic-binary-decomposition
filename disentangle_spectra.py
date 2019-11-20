import numpy as np
import matplotlib.pyplot as plt

from rv_helper_function import get_RV_ref_spectrum
from disentangle_spectra_functions import get_spectral_data

ref_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/rv_ref/'
ref_file = 'T05500G40P00V000K2SNWNVR20N.fits'

root_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/'
stars = ['GZ_Dra', 'TV_LMi']
order_centers = [4980, 5100, 5210, 5340, 5460, 5610]

for star in stars[:1]:
    # load all requested spectral data for the selected star
    star_data = get_spectral_data(star, order_centers, root_dir)
    print(star_data.keys())

    # load initial reference file that will be used for initial RV determination
    ref_flx, ref_wvl = get_RV_ref_spectrum(ref_dir + ref_file)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ------------ Radial velocity and spectrum of a primary star --------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    n_rv_iterations = 3
    for i_it in range(n_rv_iterations):
        # get per order RV velocities for every exposure
        for exp_id in star_data.keys():
            star_exposure = star_data[exp_id]

            # compute mean RV from all considered orders

        # compute (median spectrum of a secondary star)

        # update RV template and repeat RV determination with updated template

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # ------------ Radial velocity and spectrum of a secondary star ------------------
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # TODO
