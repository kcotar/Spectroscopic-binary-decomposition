import numpy as np

from glob import glob
from os import scandir, path

norm_suffix = '_normalised.txt'
sigma_norm_suffix = '_sigma_normalised.txt'


def get_reduced_exposures(in_dir):
    return [f.name for f in scandir(in_dir) if f.is_dir()]


def get_normalised_orders(spec_dir, spectrum):
    existing_orders = []
    for i_o in range(35):
        spec_path = spec_dir + spectrum + '/' + spectrum + '_' + str(i_o)
        if path.isfile(spec_path + norm_suffix) and path.isfile(spec_path + norm_suffix):
            existing_orders.append(spectrum + '_' + str(i_o))
    return existing_orders


def get_orderdata_by_wavelength(spec_dir, orders, in_wvl):
    for order in orders:
        flux_data = np.loadtxt(spec_dir + order + norm_suffix)
        sigma_data = np.loadtxt(spec_dir + order + norm_suffix)
        if np.nanmin(flux_data[:, 0]) <= in_wvl <= np.nanmax(flux_data[:, 0]):
            return flux_data[:, 0], flux_data[:, 1], sigma_data[:, 1]
    return None, None, None


def get_spectral_data(star, wvl_orders, in_dir):
    input_dir = in_dir + star + '/spec/'
    list_exposures = get_reduced_exposures(input_dir)
    # create a dictionary of all exposures with their belonging data
    star_data_all = {}
    for exposure in list_exposures:
        # get all possible orders
        print('Exploring exposures of:', exposure)
        all_norm_orders = get_normalised_orders(input_dir, exposure)
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


