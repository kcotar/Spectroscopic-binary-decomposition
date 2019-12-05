import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join

root = '/shared/data-camelot/cotar/Asiago_binaries_programme/'
t = Table.read(root+'RV_disentangle_results_newonly_renorm_noremovalfit_tellurics/final_results.fits')
r = Table.read(root+'RV_disentangle_results_newonly_renorm_noremovalfit/final_results.fits')

star = 'gz dra'
# star = 'tv lmi'

star_id = '_'.join(star.split(' '))
idx = t['star'] == star

fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].scatter(t['VHELIO'][idx], t['RV_s1'][idx], lw=0, s=6)
ax[0].errorbar(t['VHELIO'][idx], t['RV_s1'][idx], fmt='o', ms=1, elinewidth=0.2, lw=0, yerr=t['e_RV_s1'][idx])
ax[0].plot((-35, 35), (-35, 35), c='black', lw=0.5)
ax[0].set(ylabel='RV teluriki', ylim=(-33, 33))
# ax[1].scatter(t['VHELIO'][idx], t['RV_s1'][idx] - t['VHELIO'][idx], lw=0, s=6)
ax[1].errorbar(t['VHELIO'][idx], t['RV_s1'][idx] - t['VHELIO'][idx], fmt='o', ms=1, elinewidth=0.2, lw=0, yerr=t['e_RV_s1'][idx])
ax[1].set(ylabel='Residual', xlabel='VHELIO', xlim=(-33, 33), ylim=(-10, 10))
ax[0].grid(ls='--', color='black', alpha=0.2)
ax[1].grid(ls='--', color='black', alpha=0.2)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(star_id+'_vhelio', dpi=250)
plt.close(fig)


# d_rv = t['VHELIO'] - t['RV_s1']
d_rv = t['RV_s1'] - t['VHELIO']
# d_rv[:] = 0.
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].errorbar(r['phase'][idx], r['RV_s1'][idx] - d_rv[idx], yerr=r['e_RV_s1'][idx],
               fmt='o', ms=1, elinewidth=0.2, lw=0)
ax[0].set(ylabel='RV1 - d_helio')
ax[1].errorbar(t['phase'][idx], r['RV_s2'][idx]-d_rv[idx], yerr=r['e_RV_s2'][idx],
               fmt='o', ms=1, elinewidth=0.2, lw=0)
ax[1].set(ylabel='RV2 - d_helio', xlabel='Phase', xlim=(-0.05, 1.05))
ax[0].grid(ls='--', color='black', alpha=0.2)
ax[1].grid(ls='--', color='black', alpha=0.2)
fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(star_id+'_rv2.png', dpi=250)
plt.close(fig)




