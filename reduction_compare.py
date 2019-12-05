import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile

root = '/shared/data-camelot/cotar/Asiago_binaries_programme/'
dir_g = root + 'GZ_Dra/spec/'
dir_k = root + 'GZ_Dra_obdelava_klemen/'

exposure = 'EC62320'

for i_eo in range(1, 32):
    f1 = dir_g + exposure + '.ec.vh/' + exposure + '.ec.vh_' + str(i_eo) + '_normalised.txt'
    f2 = dir_k + exposure + '_vh_norm_order{:02d}.txt'.format(i_eo)

    print(f1)
    print(f2)

    fig, ax = plt.subplots(1, 1, figsize=(13, 5))
    if isfile(f1):
        d1 = np.loadtxt(f1)
        ax.plot(d1[:, 0], d1[:, 1]/np.median(d1[:, 1]), lw=0.5, label='G')

    if isfile(f2):
        d2 = np.loadtxt(f2)
        ax.plot(d2[:, 0], d2[:, 1]/np.median(d2[:, 1]), lw=0.5, label='K')

    ax.grid(ls='--', alpha=0.2, color='black')
    ax.set(xlabel='Wavelength', ylabel='Normalized flux', ylim=(0.8, 1.05))
    fig.tight_layout()
    ax.legend()
    plt.show(fig)
    plt.close(fig)