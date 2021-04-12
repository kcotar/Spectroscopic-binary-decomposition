import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir('Binaries_RV')

hjd0_T = {'V1898_Cyg': [2448501.213, 1.513123, '21 03 54', '46 19 50', 'V 7.82', '3x20min', 'P = 1.51', 'Eclipsing binary of Algol type'], 
          'V394_Vul': [245287.4124, 3.080305, '19 49 03', '20 25 31', 'V 8.73', '3x30min', 'P = 3.08', 'Eclipsing binary'], 
          'V417_Aur': [2448500.5262, 1.86553, '05 13 32', '35 39 11', 'V 7.90', '3x20min', 'P = 1.86', 'Eclipsing binary of Algol type'], 
          'V455_Aur': [2453383.2815, 3.14578, '06 28 55', '52 07 33', 'V 7.90', '3x20min', 'P = 27.0', 'Eclipsing binary of Algol type'], 
          'V994_Her': [2452436.4213, 2.0832179, '18 27 46', '24 41 51', 'V 7.14', '3x20min', 'P = 2.08', 'Eclipsing binary of Algol type'], 
          'CI_CVn': [2452723.4853, 0.8158745, '13 13 33', '47 47 52', 'V 9.37', '3x30min', 'P = 0.81', 'Eclipsing binary of beta Lyr type'], 
          'CN_Lyn': [2453318.6229, 1.95551, '08 01 37', '38 44 57', 'V 9.06', '3x30min', 'P = 1.95', 'Eclipsing binary of Algol type'], 
          'DT_Cam': [2453405.3478, 7.0662637986, '05 13 58', '56 30 29', 'V 8.13', '3x30min', 'P = 7.06', 'Eclipsing binary'],
          'DV_Cam': [2453373.4808, 6.6785, '05 19 27.9', '58 07 02.5', 'V 6.119', '3x20min', 'P = 6.67', 'Eclipsing binary of Algol type'], 
          'EQ_Boo': [2453410.6352, 5.435355, '14 52 26', '17 57 23', 'V 8.80', '3x30min', 'P = 5.43', 'Eclipsing binary of Algol type'], 
          'GK_Dra': [2452000.498, 9.9741499643, '16 45 41', '68 15 30', 'V 8.77', '3x30min', 'P = 9.97', 'Eclipsing binary of Algol type'],
          'GZ_Dra': [2453171.4359, 2.253363, '18 12 41', '54 46 07', 'V 9.49', '3x30min', 'P = 2.25', 'Eclipsing binary'], 
          'TV_LMi': [2452655.3776, 8.47799, '09 55 46', '37 11 42', 'V 8.99', '3x30min', 'P = 8.47', 'Eclipsing binary of Algol type'],
}


def phase_from_MJD(hjd, hjd0, T):    
    #return (hjd-hjd0)/T - np.int((hjd-hjd0)/T)
    return ((hjd-hjd0)%T)/T


dir1 = 'RV_disentangle_results_RVmasking_allorders_newonly_renorm_noremovalfit'
dir2 = 'RV_disentangle_results_RVmasking_newonly_renorm_noremovalfit'

for star in ['TV_LMi', 'GZ_Dra', 'V455_Aur', 'GK_Dra', 'DT_Cam', 'V394_Vul', 'CI_CVn', 'V1898_Cyg', 'V417_Aur', 'EQ_Boo', 'V994_Her', 'CN_Lyn', 'DV_Cam']:
	hjd0_s = hjd0_T[star] 

	s1_1 = np.genfromtxt(dir1 + '/' + star + '/RV_primary_' + star + '.txt')
	s1_2 = np.genfromtxt(dir2 + '/' + star + '/RV_primary_' + star + '.txt')

	s2_1 = np.genfromtxt(dir1 + '/' + star + '/RV_secondary_' + star + '.txt')
	s2_2 = np.genfromtxt(dir2 + '/' + star + '/RV_secondary_' + star + '.txt')

	fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

	ax[0].scatter(phase_from_MJD(s1_1[:, 0], hjd0_s[0], hjd0_s[1]), s1_1[:, 1], label='More orders', lw=0, s=5, c='C2')
	ax[0].scatter(phase_from_MJD(s1_2[:, 0], hjd0_s[0], hjd0_s[1]), s1_2[:, 1], label='Less orders', lw=0, s=5, c='C3')
	ax[0].set(xlabel='RV primary')
	ax[0].grid(ls='--', c='black', alpha=0.2)
	ax[0].legend()

	ax[1].scatter(phase_from_MJD(s2_1[:, 0], hjd0_s[0], hjd0_s[1]), s2_1[:, 1], label='More orders', lw=0, s=5, c='C2')
	ax[1].scatter(phase_from_MJD(s2_2[:, 0], hjd0_s[0], hjd0_s[1]), s2_2[:, 1], label='Less orders', lw=0, s=5, c='C3')
	ax[1].set(xlabel='RV secondary', xlim=(0, 1), ylabel='Phase')
	ax[1].grid(ls='--', c='black', alpha=0.2)
	ax[1].legend()

	fig.tight_layout()
	fig.subplots_adjust(hspace=0, wspace=0)
	fig.savefig(star + '_rv_compare.png', dpi=200)
	plt.close(fig)


