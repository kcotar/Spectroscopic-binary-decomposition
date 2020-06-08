import numpy as np
import joblib
from copy import deepcopy, copy
from os import system, path
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

data_dir = '/shared/data-camelot/cotar/Asiago_binaries_programme/RV_disentangle_results_newonly_renorm_noremovalfit/'
rv_data = Table.read(data_dir + 'final_results.fits')

key_fit = 'RV_s1'
rv_data = rv_data[np.logical_and(np.isfinite(rv_data[key_fit]), rv_data['star'] == 'gz dra')]

period = 2.25335
x = rv_data['JD']%period/period
y = rv_data[key_fit]

fit_f = models.Sine1D(amplitude=40, frequency=1., phase=0.) + models.Const1D(amplitude=np.median(rv_data[key_fit]))
fitter = fitting.LevMarLSQFitter()
fit_res = fitter(fit_f, x, y)

print(fit_res)
x_eval = np.arange(0., 1, 0.01)
plt.scatter(x, y, c='C3')
plt.plot(x_eval, fit_res(x_eval), c='C2')
plt.show()
plt.close()
