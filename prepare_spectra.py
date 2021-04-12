from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from shutil import copyfile
from astropy.table import Table
from pyraf import iraf
from glob import glob
from disentangle_spectra_functions import _spectra_normalize, _go_to_dir


def export_observation_to_txt(fits_path, txt_path):
	print(' Exporting file')
	for order in np.arange(1, 32, 1):
		try:
			iraf.wspectext(input=fits_path+'[*,'+str(order)+',1]', output=txt_path+'_{:.0f}.txt'.format(order), header='no')
		except Exception as e:
			print(e)
			pass


iraf.noao(_doprint=0, Stdout="/dev/null")
iraf.rv(_doprint=0, Stdout="/dev/null")
iraf.imred(_doprint=0, Stdout="/dev/null")
iraf.ccdred(_doprint=0, Stdout="/dev/null")
iraf.images(_doprint=0, Stdout="/dev/null")
iraf.immatch(_doprint=0, Stdout="/dev/null")
iraf.onedspec(_doprint=0, Stdout="/dev/null")
iraf.twodspec(_doprint=0, Stdout="/dev/null")
iraf.apextract(_doprint=0, Stdout="/dev/null")
iraf.imutil(_doprint=0, Stdout="/dev/null")
iraf.echelle(_doprint=0, Stdout="/dev/null")
iraf.astutil(_doprint=0, Stdout="/dev/null")
iraf.apextract.dispaxi = 1
iraf.echelle.dispaxi = 1
iraf.ccdred.instrum = 'blank.txt'
os.environ['PYRAF_BETA_STATUS'] = '1'
os.system('mkdir uparm')
iraf.set(uparm=os.getcwd() + '/uparm')

data_dir = '/data4/travegre/Projects/Asiago_binaries/'
reduc_dir = '/data4/travegre/Projects/Echelle_Asiago_Reduction/delo/observations/'

obs_metadata = Table.read(data_dir + 'star_data_all.csv')
obs_metadata = obs_metadata[obs_metadata['odkdaj'] == 'nova']

_go_to_dir('Binaries_spectra')
copyfile(data_dir + 'star_data_all.csv', 'star_data_all.csv')

stars = ['TV_LMi', 'GZ_Dra', 'V455_Aur', 'GK_Dra', 'DT_Cam', 'V394_Vul', 'CI_CVn', 'V1898_Cyg', 'V417_Aur', 'EQ_Boo', 'V994_Her', 'CN_Lyn', 'DV_Cam'][6:15]

for star in stars:

	print('Working on star ' + star)
	star_obs = obs_metadata[obs_metadata['star'] == star.replace('_', ' ').lower()]

	source_folder = ['binaries_13_' + dd[:4] + dd[5:7] for dd in star_obs['dateobs']]
	star_obs['source_folder'] = source_folder

	print(star_obs[np.argsort(star_obs['JD'])])

	_go_to_dir(star)

	# remove everything
	os.system('rm -R *')

	_go_to_dir('spec')

	n_no_spec = 0
	n_no_wvl = 0
	for i_s in range(len(star_obs)):
		star_spec = star_obs[i_s]
		spec_name = star_spec['filename']
		spec_suff = '.ec.vh'
		print(spec_name)

		# does reduced data exist
		targ_dir = reduc_dir + star_spec['source_folder'] + '/' + spec_name + spec_suff + '.fits'
		if not os.path.isfile(targ_dir):

			# check date foders before and after
			year = int(star_spec['source_folder'][-6:-2])
			month = int(star_spec['source_folder'][-2:])

			if month == 12:
				date_plus = '{:04d}{:02d}'.format(year+1, 1)
			else:
				date_plus = '{:04d}{:02d}'.format(year, month+1)

			if month == 1:
				date_minus = '{:04d}{:02d}'.format(year-1, 12)
			else:
				date_minus = '{:04d}{:02d}'.format(year, month-1)

			targ_dir_m = reduc_dir + star_spec['source_folder'][:-6] + date_minus + '/' + spec_name + spec_suff + '.fits'
			targ_dir_p = reduc_dir + star_spec['source_folder'][:-6] + date_plus + '/' + spec_name + spec_suff + '.fits'

			if os.path.isfile(targ_dir_m):
				print(' Found at previous date - ' + targ_dir_m)
				targ_dir = targ_dir_m
			elif os.path.isfile(targ_dir_p):
				print(' Found at latter date - ' + targ_dir_p)
				targ_dir = targ_dir_p
			else:
				print(' Not found ' + targ_dir)
				n_no_spec += 1
				continue

		# copy reduced data
		copyfile(targ_dir, spec_name + spec_suff + '.fits')
		_go_to_dir(spec_name + spec_suff)
		export_observation_to_txt(targ_dir, spec_name + spec_suff)

		# normalise spectra
		print(' Normalising spectra')
		bad_wvl_order = 0
		no_data_order = 0
		for txt_file in glob(spec_name + '*_*.txt'):
			txt_out = txt_file[:-4] + '_normalised.txt'
			order_data = np.loadtxt(txt_file)

			if len(order_data) == 0:
				print(' No data in order ' + txt_file)
				no_data_order += 1
				continue

			# crop order data to remove noisy part of the echelle order
			n_wvl_rem1 = 50
			order_data = order_data[n_wvl_rem1:-n_wvl_rem1, :]
			n_data = order_data.shape[0]

			ref_flx_norm_curve1 = _spectra_normalize(np.arange(n_data), order_data[:, 1],
													steps=10, sigma_low=2., sigma_high=2.5, n_min_perc=8.,
													order=11, func='cheb', return_fit=True, median_init=False)

			ref_flx_norm_curve2 = _spectra_normalize(np.arange(n_data), order_data[:, 1]/ref_flx_norm_curve1,
													steps=10, sigma_low=2., sigma_high=2.5, n_min_perc=8.,
													order=3, func='cheb', return_fit=True, median_init=True)

			# extra wvl border removal
			n_wvl_rem2 = 100
			order_data = order_data[n_wvl_rem2:-n_wvl_rem2, :]
			ref_flx_norm_curve1 = ref_flx_norm_curve1[n_wvl_rem2:-n_wvl_rem2]
			ref_flx_norm_curve2 = ref_flx_norm_curve2[n_wvl_rem2:-n_wvl_rem2]

			# renorm order
			fig, ax = plt.subplots(3, 1, sharex=True, figsize=(13, 7))
			ax[0].plot(order_data[:, 0], order_data[:, 1])
			ax[0].plot(order_data[:, 0], ref_flx_norm_curve1)
			ax[1].plot(order_data[:, 0], order_data[:, 1] / ref_flx_norm_curve1)
			ax[1].plot(order_data[:, 0], ref_flx_norm_curve2)
			ax[2].plot(order_data[:, 0], order_data[:, 1] / ref_flx_norm_curve1 / ref_flx_norm_curve2)
			ax[1].set(ylim=(0.3, 1.2))
			ax[2].set(ylim=(0.3, 1.2), xlim=(order_data[0, 0], order_data[-1, 0]))
			fig.tight_layout()
			fig.subplots_adjust(hspace=0, wspace=0)
			fig.savefig(txt_file[:-4] + '_normalised.png')
			plt.close(fig)

			if order_data[1, 0] - order_data[0, 0] == 1:
				print(' No wav cal in ' + txt_file)
				bad_wvl_order += 1
				continue

			order_data[:, 1] = order_data[:, 1] / ref_flx_norm_curve1
			np.savetxt(txt_out, order_data, fmt=['%.5f', '%.5f'])

		if bad_wvl_order > 20:
			n_no_wvl += 1

		os.chdir('..')

	print("Stats for {:s} are  ->  exposures: {:.0f}  no-data: {:.0f}  no-wvl: {:.0f}  ok: {:.0f}".format(star, len(star_obs), n_no_spec, n_no_wvl, len(star_obs) - n_no_wvl - n_no_spec))

	os.chdir('..')
	os.chdir('..')
	print('==========================================================================')
	print('==========================================================================')
	print('\n \n \n')
