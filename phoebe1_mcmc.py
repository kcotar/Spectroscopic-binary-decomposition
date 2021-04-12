# -*- encoding: utf-8 -*-
#!/usr/bin/python

MCMC_VERSION = '0.12'
PHOEBE_VERSION = '1.0-svn'

"""
MCMC wrapper for phoebe 1. The example used is a G-G pair of stars in
the quintuple KIC 4150611 from the Kepler data-set.

What follows is a brief description of how to get started with a fresh
binary in 9 steps, step by step.

(1) Gather all your data and run phoebe 1 in the GUI mode. Get as close
    to the solution as possible by either manual tweaking or by DC.

(2) Note the parameters that you want to pass to MCMC for fitting, and
    their allowed range. For now, all priors on parameters are uniform
    (i.e. non-informative), and the boundaries are imposed as hard
    limits.

(3) Make a list @pars of all adjusted parameter qualifiers. For example:
    
    pars = ['phoebe_incl', 'phoebe_pot1', 'phoebe_pot2', 'phoebe_teff2']

(4) Make a list @priors of all adjusted parameter priors. The priors are
    a list of two-dimensional tuples, one for each parameter in @pars.
    For example:
    
    priors = [(80., 90.), (3.75, 4.25), (6.0, 7.0), (6500., 7500.)]

(5) Build an initial chain file by running @init_parameters:

    >>> import mcmc
    >>> mcmc.init_parameters(chain_file, pars, priors)
    
    Chain file should be the name of the phoebe parameter file with
    '.mcmc' appended to it. For example, if your phoebe parameter file
    is 'castor.phoebe', then chain_file should be 'castor.phoebe.mcmc'.

(6) Take a quick look at the created chain file and make sure that
    everything makes sense. If it does, great. If not, fix it by hand
    or rerun init_parameters().

(7) At the end of this file, in the BASIC SETUP part, change the
    chain_file name to the name of the file you just created. Set the
    number of walkers to the desired number (128 is the default), set
    the number of iterations to the desired number (1000 is the default)
    and, provided that you are not continuing a previous mcmc run, set
    state to None.

(8) The main workhorse is the lnprob function. You need to modify this
    according to what you want to achieve. In this example we are
    fitting one light curve and two radial velocity curves, and we are
    adding their log-likelihoods into the final merit function.

(9) After the first chain is computed, you can continue by setting the
    @state variable to the last iteration's state. The simplest way is
    to run:
    
      tail -n nwalkers chain_file > state
    
    and then set:
    
      state = 'state'
    
    Then you can run the script again and it will continue where it left
    off.

There are also functions for adding parameters to already run chains,
and to resample priors from a subset of posteriors using multivariate
sampling. The latter is particularly important if your chains converge
to local minima with a notably lower log-likelihood and you need to
eliminate them for a more efficient convergence.

Credits for the initial MCMC wrapper go to Gal Matijevic.
"""

import sys

import numpy as np
import phoebeBackend as phb  # so it doesn't get confused with the new phoebe

import emcee
import multiprocessing
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')  
#matplotlib.style.use('classic')
import matplotlib.pyplot as plt 
#import scipy.stats as st
#from phoebe.atmospheres.roche import binary_potential
#import phoebe.algorithms.marching as marching
#from phoebe.utils.coordinates import cart2spher_coord, spher2cart_coord
import os
import gc
#import triangle
import scipy.stats as st

#os.environ['LD_LIBRARY_PATH'] = '/home/travegre/phoebe/stable/phoebe-lib/libphoebe/.libs' 
#os.system('export LD_LIBRARY_PATH=/home/travegre/phoebe/stable/phoebe-lib/libphoebe/.libs') 


hjd0_T = {'V1898 Cyg': [2448501.213, 1.513123, '21 03 54', '46 19 50', 'V 7.82', '3x20min', 'P = 1.51', 'Eclipsing binary of Algol type'], 
          'V394 Vul': [245287.4124, 3.080305, '19 49 03', '20 25 31', 'V 8.73', '3x30min', 'P = 3.08', 'Eclipsing binary'], 
          'V417 Aur': [2448500.5262, 1.86553, '05 13 32', '35 39 11', 'V 7.90', '3x20min', 'P = 1.86', 'Eclipsing binary of Algol type'], 
          'V455 Aur': [2453383.2815, 3.14578, '06 28 55', '52 07 33', 'V 7.90', '3x20min', 'P = 27.0', 'Eclipsing binary of Algol type'], 
          'V994 Her': [2452436.4213, 2.0832179, '18 27 46', '24 41 51', 'V 7.14', '3x20min', 'P = 2.08', 'Eclipsing binary of Algol type'], 
          'CI CVn': [2452723.4853, 0.8158745, '13 13 33', '47 47 52', 'V 9.37', '3x30min', 'P = 0.81', 'Eclipsing binary of beta Lyr type'], 
          'CN Lyn': [2453318.6229, 1.95551, '08 01 37', '38 44 57', 'V 9.06', '3x30min', 'P = 1.95', 'Eclipsing binary of Algol type'], 
          'DT Cam': [2453405.3478, 7.0662637986, '05 13 58', '56 30 29', 'V 8.13', '3x30min', 'P = 7.06', 'Eclipsing binary'],
          'DV Cam': [2453373.4808, 6.6785, '05 19 27.9', '58 07 02.5', 'V 6.119', '3x20min', 'P = 6.67', 'Eclipsing binary of Algol type'], 
          'EQ Boo': [2453410.6352, 5.435355, '14 52 26', '17 57 23', 'V 8.80', '3x30min', 'P = 5.43', 'Eclipsing binary of Algol type'], 
          'GK Dra': [2452000.498, 9.9741499643, '16 45 41', '68 15 30', 'V 8.77', '3x30min', 'P = 9.97', 'Eclipsing binary of Algol type'],
          'GZ Dra': [2453171.4359, 2.253363, '18 12 41', '54 46 07', 'V 9.49', '3x30min', 'P = 2.25', 'Eclipsing binary'], 
          'TV LMi': [2452655.3776, 8.47799, '09 55 46', '37 11 42', 'V 8.99', '3x30min', 'P = 8.47', 'Eclipsing binary of Algol type'],
          #'TYC 5227-1023-1': [2457003.325, 4.306193, '22 00 52.6', '-03 42 12.4', 'V 11.86', '3x30min'],
      #'V1695 Aql': [2457634.7053, 0.4128296, '19 38 22.2982', '-03 32 37.142', 'V 11.0', '1x30min'],
}

d_hjo = 2.  # percent of period for maksimum change of hjdo
d_per = 1.  # percent of maksimum period change

phoebe_pars = {
          'V1898_Cyg': [2448501.213, 1.513123, '21 03 54', '46 19 50', 'V 7.82', '3x20min', 'P = 1.51', 'Eclipsing binary of Algol type'], 
          'V394_Vul': {'t0_supconj': (hjd0_T['V394 Vul'][0]-hjd0_T['V394 Vul'][1]*d_hjo/100., hjd0_T['V394 Vul'][0]+hjd0_T['V394 Vul'][1]*d_hjo/100.), 
                       'period@binary': (hjd0_T['V394 Vul'][1]*(1-d_per/100.), hjd0_T['V394 Vul'][1]*(1+d_per/100.)), 
                       'sma@binary': (10, 15), 
                       'q@binary': (0.6, 0.85),
                       'incl@binary': (80, 86), 
                       'ecc@binary': (0.0, 0.1), 
                       'teff@primary': 8750, 
                       'teff@secondary': (7000, 8500),
                       'pot1': (4, 15),
                       'pot2': (4, 15),
                       'hla1': (0.5, 8),
                       'hla2': (0.5, 8),
                       'el31': (0, 0.5),
                       'el32': (0, 0.5),
                       'ext@primary': (0, 5),
                       'ext@secondary': (0, 5),
                       'met@primary': (-3, 1),
                       'met@secondary': (-3, 1),
                       'vgamma@system': (5, 15), 
                       'perr0@binary': (0, 2*np.pi)
                      }, 
          'V417_Aur': [2448500.5262, 1.86553, '05 13 32', '35 39 11', 'V 7.90', '3x20min', 'P = 1.86', 'Eclipsing binary of Algol type'], 
          'V455_Aur': {'t0_supconj': (hjd0_T['V455 Aur'][0]-hjd0_T['V455 Aur'][1]*d_hjo/100., hjd0_T['V455 Aur'][0]+hjd0_T['V455 Aur'][1]*d_hjo/100.), 
                       'period@binary': (hjd0_T['V455 Aur'][1]*(1-d_per/100.), hjd0_T['V455 Aur'][1]*(1+d_per/100.)), 
                       'sma@binary': (11, 13),
                       'q@binary': (0.9, 1.0),
                       'incl@binary': (84, 86),
                       'ecc@binary': (0.0, 0.02),
                       'teff@primary': 7050, 
                       'teff@secondary': (6000, 7500),
                       'pot1': (8, 12),
                       'pot2': (8, 12),
                       'hla1': (1, 6),
                       'hla2': (4, 8),
                       'el31': (0, 0.5),
                       'el32': (0, 0.5),
                       'ext@primary': (0, 5),
                       'ext@secondary': (0, 5),
                       'met@primary': (-3, 1),
                       'met@secondary': (-3, 1),
                       'vgamma@system': (5, 15),
                       'perr0@binary': (2., 3.)
                      },
          'V994_Her': [2452436.4213, 2.0832179, '18 27 46', '24 41 51', 'V 7.14', '3x20min', 'P = 2.08', 'Eclipsing binary of Algol type'], 
          'CI_CVn': [2452723.4853, 0.8158745, '13 13 33', '47 47 52', 'V 9.37', '3x30min', 'P = 0.81', 'Eclipsing binary of beta Lyr type'], 
          'CN_Lyn': [2453318.6229, 1.95551, '08 01 37', '38 44 57', 'V 9.06', '3x30min', 'P = 1.95', 'Eclipsing binary of Algol type'], 
          'DT_Cam': {'t0_supconj': (hjd0_T['DT Cam'][0]-hjd0_T['DT Cam'][1]*d_hjo/100., hjd0_T['DT Cam'][0]+hjd0_T['DT Cam'][1]*d_hjo/100.), 
                     'period@binary': (hjd0_T['DT Cam'][1]*(1-d_per/100.), hjd0_T['DT Cam'][1]*(1+d_per/100.)), 
                     'sma@binary': (20, 25), 
                     'q@binary': (0.75, 0.9),
                     'incl@binary': (86, 90),
                     'ecc@binary': (0.1, 0.3),
                     'teff@primary': 9040, 
                     'teff@secondary': (5000, 7000),
                     'pot1': (10, 20),
                     'pot2': (10, 20),
                     'hla1': (40, 50),
                     'hla2': (45, 55),
                     'el31': (0, 0.5),
                     'el32': (0, 0.5),
                     'ext@primary': (0, 5),
                     'ext@secondary': (0, 5),
                     'met@primary': (-3, 1),
                     'met@secondary': (-3, 1),
                     'vgamma@system': (11, 22),
                     'perr0@binary': (0, 2*np.pi)
                    },
          'DV_Cam': [2453373.4808, 6.6785, '05 19 27.9', '58 07 02.5', 'V 6.119', '3x20min', 'P = 6.67', 'Eclipsing binary of Algol type'], 
          'EQ_Boo': [2453410.6352, 5.435355, '14 52 26', '17 57 23', 'V 8.80', '3x30min', 'P = 5.43', 'Eclipsing binary of Algol type'], 
          'GK_Dra': {'t0_supconj': (hjd0_T['GK Dra'][0]-hjd0_T['GK Dra'][1]*d_hjo/100., hjd0_T['GK Dra'][0]+hjd0_T['GK Dra'][1]*d_hjo/100.), 
                     'period@binary': (hjd0_T['GK Dra'][1]*(1-d_per/100.), hjd0_T['GK Dra'][1]*(1+d_per/100.)), 
                     'sma@binary': (20, 31),
                     'q@binary': (0.6, 0.8),
                     'incl@binary': (82, 87), 
                     'ecc@binary': (0.15, 0.35),
                     'teff@primary': 6050, 
                     'teff@secondary': (3000, 4500),
                     'pot1': (5, 15),
                     'pot2': (8, 22),
                     'hla1': (10, 20),
                     'hla2': (15, 35),
                     'el31': (0, 0.5),
                     'el32': (0, 0.5),
                     'ext@primary': (0, 5),
                     'ext@secondary': (0, 5),
                     'met@primary': (-3, 1),
                     'met@secondary': (-3, 1),
                     'vgamma@system': (-1, 10),
                     'perr0@binary': (0, 2*np.pi)
                    }, 
          'GZ_Dra': [2453171.4359, 2.253363, '18 12 41', '54 46 07', 'V 9.49', '3x30min', 'P = 2.25', 'Eclipsing binary'],
          'TV_LMi': {'t0_supconj': (hjd0_T['TV LMi'][0]-hjd0_T['TV LMi'][1]*d_hjo/100., hjd0_T['TV LMi'][0]+hjd0_T['TV LMi'][1]*d_hjo/100.),
                     'period@binary': (hjd0_T['TV LMi'][1]*(1-d_per/100.), hjd0_T['TV LMi'][1]*(1+d_per/100.)), 
                     'sma@binary': (19, 24),
                     'q@binary': (0.55, 0.75),
                     'incl@binary': (85, 88),
                     'ecc@binary': (0.2, 0.35),
                     'teff@primary': 5660, 
                     'teff@secondary': (3500, 5000),
                     'pot1': (25, 35),
                     'pot2': (5, 15),
                     'hla1': (10, 20),
                     'hla2': (15, 25),
                     'el31': (0, 0.5),
                     'el32': (0, 0.5),
                     'ext@primary': (0, 5),
                     'ext@secondary': (0, 5),
                     'met@primary': (-3, 1),
                     'met@secondary': (-3, 1),
                     'vgamma@system': (-3, 8),
                     'perr0@binary': (3, 5)
                    },
}

comp_pars = ["phoebe_mass1", "phoebe_mass2", "phoebe_radius1", "phoebe_radius2", "phoebe_mbol1", "phoebe_mbol2", "phoebe_logg1", "phoebe_logg2"]

#https://arxiv.org/pdf/1510.07674.pdf, IAU 2015 Resolution B3 on Recommended Nominal Conversion Constants for Selected Solar and Planetary Properties
Rsun = 6.957*(10**8)
Lsun = 3.828*(10**26)
Tsun = 5772
GMsun = 1.3271244*(10**20)

G = 6.67408*(10**-11) # 2014 CODATA

# http://maia.usno.navy.mil/NSFA/NSFA_cbe.html
c = 299792.458

SIGMA = 5.670367*(10**-8) # 2014 CODATA

# Using the IAU 2012 Resolution B2 definition of the astronomical unit, the parsec corresponds to 3.085 677 581
KPCTOM = 3.085677581*(10**19)


def lnprob(x, adjpars, priors, showlc=False):
    global comp_pars
    
    # Check to see that all values are within the allowed limits:   
    if not np.all([priors[i][0] < x[i] < priors[i][1] for i in range(len(priors))]):
        print 'inf1'
        return -np.inf, []

    for i, j in enumerate(adjpars):
        # print j, x[i]
        phb.setpar(j[0], x[i], j[1])

    # print 'Teff primary:', phb.getpar('phoebe_teff1')
    # Using these new values, update LD coefficients:
    try:
        phb.updateLD()
    except:
        print 'LD error'
        return -np.inf, []
    
    chi2lc1 = phb.cfval('lc', 0, False) #/ 182.
    chi2lc2 = phb.cfval('lc', 1, False) #/ 182.
    #return chi2lc1 + chi2lc2
    chi2rv1 = phb.cfval('rv', 0) #/ 8.
    chi2rv2 = phb.cfval('rv', 1) #/ 8.

    if False:#multiprocessing.current_process()._identity[0] == 10:
        # print chi2lc1, chi2lc2, chi2rv1, chi2rv2
        phs = tuple(np.linspace(-0.25, 8., 1.5/0.001 + 1))
        rv1 = phb.rv1(phs, 0)
        rv2 = phb.rv2(phs, 1)
        phs = np.array(phs)
        plt.plot(phs, rv1, color='red')
        plt.plot(phs, rv2, color='blue')    
        #plt.show()
        plt.savefig('rv_mcmc.png')
        plt.close()

    if np.isnan(chi2lc1) or np.isnan(chi2lc2) or np.isnan(chi2rv1) or np.isnan(chi2rv2):
        print 'inf2'
        return -np.inf, []

    lnp = - 0.5 * (chi2lc1 + chi2lc2 + chi2rv1 + chi2rv2)
    
    # ========
    # DISTANCE    
    # ========
    
    dparsec = 0

    print lnp
    return lnp, [phb.getpar(par) for par in comp_pars] + [dparsec]


def run_sampler(star, sampler, niter, ntot_iter, p0, phoebe_file, prefix):
    day = 86400.0

    f = open(phoebe_file + prefix + '.mcmc', "a")

    for result in sampler.sample(p0, iterations=niter, storechain=False):        
        ntot_iter += 1
        position = result[0]
        computed = result[3]
       
        # for each walker      
        for k in range(position.shape[0]):
            
            try:
                R1 = computed[k][2]
                R2 = computed[k][3]                
                P = position[k][1]
                sma = position[k][2]
                q = position[k][3]
                inc = position[k][4]
                ecc = position[k][5]
                sma2 = sma / (1.+q)
                sma1 = sma - sma2
            
                r1 = R1 / sma
                r2 = R2 / sma
                K1 = 2*np.pi*sma1*Rsun*np.sin(inc*np.pi/180.)/P/day/np.sqrt(1.-ecc*ecc) # K1 = 2pi sma1 sini / P / sqrt(1-e^2)
                K2 = 2*np.pi*sma2*Rsun*np.sin(inc*np.pi/180.)/P/day/np.sqrt(1.-ecc*ecc) # K2 = 2pi sma2 sini / P / sqrt(1-e^2)
                dparsec = computed[k][-1] 
            except:            
                r1 = r2 = K1 = K2 = dparsec = 0
                computed[k] = [0,0,0,0,0,0,0,0,0]

            #print position[k]

            f.write("%d %s %s %s %s %s %s %s %f\n" % ( k, 
                                    " ".join(['%.12f' % i for i in position[k]]),   # phoebe adjusted parameters
                                    " ".join(['%.12f' % i for i in computed[k]]),   # phoebe computed parameters
                                    '%.12f' % r1,                           # r1 = R1 / sma
                                    '%.12f' % r2,                           # r2 = R2 / sma
                                    '%.12f' % K1,    
                                    '%.12f' % K2,
                                    '%.12f' % dparsec,
                                    result[1][k]                                    # lnprob value
                                  )
            )

        
        #print "updated file"
    f.close()

    return ntot_iter

def run(star, phoebe_file, adjpars, priors, state, nwalkers, niter, prefix):    
    phb.init()
    phb.configure()
    phb.open('/home/klemen/phoebe1/'+star+'/'+phoebe_file)

    ndim = len(adjpars)  

    mcmc_files = phoebe_file + prefix + '.mcmc'

    if state:
      p0 = np.genfromtxt(mcmc_files)[-nwalkers:, 1:ndim+1]
      print "state file loaded p0"
    else:
      try:
        os.remove(mcmc_files)
      except:
        pass
      p0 = np.array([[p[0] + (p[1]-p[0])*np.random.rand() for p in priors] for i in xrange(nwalkers)])

    pool = Pool(160)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[adjpars, priors], pool=pool)    
    
    ntot_iter = 0
    ntot_iter = run_sampler(star, sampler, 800, ntot_iter, p0, phoebe_file, prefix)
    #
    p0 = force_reshufle_walkers(mcmc_files, nwalkers, ndim, 90, use_epochs=10)
    # #force_walkers(star, mcmc_files, nwalkers, ndim, 25, nwalkers_out=nwalkers)
    # p0 = np.genfromtxt(mcmc_files)[-nwalkers:, 1:ndim+1]
    #
    ntot_iter = run_sampler(star, sampler, niter, ntot_iter, p0, phoebe_file, prefix)
    #
    # p0 = force_reshufle_walkers(mcmc_files, nwalkers, ndim, 90, use_epochs=10)
    # force_walkers(star, mcmc_files, nwalkers, ndim, 50, nwalkers_out=nwalkers)
    # p0 = np.genfromtxt(mcmc_files)[-nwalkers:, 1:ndim+1]
    #
    # ntot_iter = run_sampler(star, sampler, niter, ntot_iter, p0, phoebe_file, prefix)

    '''
    n_per_c_step = 1000
    # while not convergence_achieved(mcmc_files, nwalkers, 150):
    for i_step in range(0, niter, n_per_c_step):
        print 'Starting with step', i_step

        p0 = force_reshufle_walkers(mcmc_files, nwalkers, ndim, 80, use_epochs=20)
        # print 'New p0'
        # print p0
        #
        # force_walkers(star, mcmc_files, nwalkers, ndim, 50, nwalkers_out=nwalkers)
        # p0_ = np.genfromtxt(mcmc_files)[-nwalkers:, 1:ndim+1]
        # print 'Old p0'
        # print p0_
        #
        # raise SystemExit

        ntot_iter = run_sampler(star, sampler, n_per_c_step, ntot_iter, p0, phoebe_file, prefix)
    '''

    pool.close()

    phb.quit()

def convergence_achieved(state, nwalkers, span):
  samples_all = np.genfromtxt(state)
  samples = samples_all[:, 1:]

  span = span-50
  first_chunk = np.array([samples_all[i::nwalkers][-span:-span+15, -1] for i in range(nwalkers)]).flatten()
  last_chunk = np.array([samples_all[i::nwalkers][-15:, -1] for i in range(nwalkers)]).flatten()
  last_step = np.array([samples_all[i::nwalkers][-1, -1] for i in range(nwalkers)]).flatten()

  upp = last_step[last_step >= np.percentile(last_step, 90)]
  dow = last_step[last_step <= np.percentile(last_step, 10)]
  
  print "conv crit: ", abs(np.nanmedian(first_chunk) - np.nanmedian(last_chunk)), np.nanstd(last_chunk)
  # prevent split (although converged) branches
  print "20 * upper std, split diff: ", 20*np.nanstd(upp), abs(np.min(dow) - np.max(upp))
  
  if (abs(np.nanmedian(first_chunk) - np.nanmedian(last_chunk)) < (0.5 * np.nanstd(last_chunk))) & (20*np.nanstd(upp) > abs(np.min(dow) - np.max(upp))):
    return 1
  else:
    return 0


def force_reshufle_walkers(mcmc_files, nwalkers, ndim, percentile, use_epochs=20):
    print 'Forcing and resampling walkers initial positions'
    # eliminate the worst branches and reshufle/resample other branches
    samples_all = np.genfromtxt(mcmc_files)
    samples_last = samples_all[-(use_epochs*nwalkers):, :]

    lnprob_thr = np.nanpercentile(samples_last[:, -1], percentile)
    idx_use = samples_last[:, -1] >= lnprob_thr
    samples_use = samples_last[idx_use, :]

    # determine statistics of the best samples
    stats = list([])
    for i_c in range(1, ndim+1):
        med = np.nanmedian(samples_use[:, i_c])
        std = np.nanstd(samples_use[:, i_c])
        idx_rows = np.abs(samples_use[:, i_c] - med) < 3 * std
        stats.append([np.nanmedian(samples_use[idx_rows, i_c]), np.nanstd(samples_use[idx_rows, i_c])])

    # generate new initial p0 walkers
    p0 = list([])
    for i_w in range(walkers):
        p0_w = list([])
        for i_s in range(len(stats)):
            p0_w.append(np.random.normal(stats[i_s][0], stats[i_s][1]/10, 1)[0])
        p0.append(p0_w)

    return np.array(p0)


def force_walkers(star, mcmc_files, nwalkers, ndim, percentile, nwalkers_out=128):
  # ========= FORCE WALKERS TO LOWEST BRANCH IN THE MIDDLE OF ITERATIONS
  print "forcing to ", percentile, ' ', star

  samples_all = np.genfromtxt(mcmc_files)
  samples = samples_all[:, 1:]
      
  lnprob_lim_high = np.inf

  all_w = np.array([samples_all[i::nwalkers][-2:-1:, -1] for i in range(nwalkers)]).flatten()
  perc_up10 = np.nanpercentile(all_w, percentile)
  lnprob_lim_low = perc_up10
  
  samples_by_walkers = []

  for i in range(nwalkers):
    walkers_burnin_cut = samples_all[i::nwalkers][-2:-1, -1]       
    
    if walkers_burnin_cut > lnprob_lim_low:
      samples_by_walkers.append(samples[i::nwalkers][-2:-1])  
  
  #print samples_by_walkers
  samples = np.concatenate(np.transpose(np.array(samples_by_walkers), (1,0,2)))

  samples = sorted(samples, key=lambda x: x[-1], reverse=True)

  smpls = []
  if nwalkers_out > len(samples):
    samples = samples * int(np.ceil(nwalkers_out/len(samples))+2)
  
  for i in range(nwalkers_out):    
    smpls.append(samples.pop(0))

  f = open(mcmc_files, 'a')
  for i, sm in enumerate(smpls):
    f.write("%d " % (i))
    for j in sm:
      f.write('%.5f ' % j)
    f.write('\n')
  f.close()
  # ============================================


if __name__ == '__main__':

    what_to_fit = [0,  # origin of HJD time
                   0,  # orbital period
                   1,  # semi-major axis
                   1,  # mass ratio
                   1,  # inclination
                   1,  # eccentricity
                   1,  # secondary temperature
                   1,  # radius proxy
                   1,  # radius proxy
                   1,  # first band luminosities
                   1,  # secondary band luminosities
                   0,  # third light
                   0,  # third light
                   0,  # primary extinction
                   0,  # secondary extinction
                   0,  # primary metallicity
                   0,  # secondary metallicity
                   1,  # centre of mass velocity
                   1,  # argument of periastron
                  ]

    # set, nwalkers, niters

    in_args = sys.argv
    if len(in_args) > 1:
        set = str(in_args[1])
    else:
        set = 'd'
    if len(in_args) > 2:
        walkers = int(in_args[2])
    else:
        walkers = 128
    if len(in_args) > 3:
        epochs = int(in_args[3])
    else:
        epochs = 1000

    state = False

    init_sets = {    
    'a': [walkers, epochs, 'TV_LMi_partial_results', what_to_fit, 'TV_LMi'],
    'b': [walkers, epochs, 'V394_Vul_partial_results', what_to_fit, 'V394_Vul'],
    'c': [walkers, epochs, 'V455_Aur_partial_results', what_to_fit, 'V455_Aur'],
    'd': [walkers, epochs, 'GK_Dra_partial_results', what_to_fit, 'GK_Dra'],
    'e': [walkers, epochs, 'DT_Cam_partial_results', what_to_fit, 'DT_Cam'],
    }

    # print init_sets[set]
    # raise SystemExit

    star = init_sets[set][4]
    pp = phoebe_pars[star]
    adjpars = [['phoebe_hjd0', 0], 
               ['phoebe_period', 0], 
               ['phoebe_sma', 0], 
               ['phoebe_rm', 0], 
               ['phoebe_incl', 0], 
               ['phoebe_ecc', 0], 
               ['phoebe_teff2', 0], 
               ['phoebe_pot1', 0], 
               ['phoebe_pot2', 0], 
               ['phoebe_hla', 0], 
               ['phoebe_hla', 1], 
               ['phoebe_el3', 0], 
               ['phoebe_el3', 1], 
               ['phoebe_extinction', 0], 
               ['phoebe_extinction', 1], 
               ['phoebe_met1', 0], 
               ['phoebe_met2', 0], 
               ['phoebe_vga', 0], 
               ['phoebe_perr0', 0]
              ]
    priors = [  
                pp['t0_supconj'], 
                pp['period@binary'], 
                pp['sma@binary'],
                pp['q@binary'], 
                pp['incl@binary'], 
                pp['ecc@binary'],
                pp['teff@secondary'],
                pp['pot1'],
                pp['pot2'],
                pp['hla1'],
                pp['hla2'],
                pp['el31'],
                pp['el32'],
                pp['ext@primary'],
                pp['ext@secondary'], 
                pp['met@primary'], 
                pp['met@secondary'],
                pp['vgamma@system'],
                pp['perr0@binary']
            ]

    mask = np.array(init_sets[set][3]) == 1
    adjpars = np.array(adjpars)[mask]
    priors = np.array(priors)[mask]
    adjpars = [[x[0], int(x[1])] for x in adjpars]
    priors = [tuple(x) for x in priors]

    #force_walkers(init_sets[set][2]+set, init_sets[set][0], len(adjpars), 80, nwalkers_out=init_sets[set][0])
    #sys.exit()

    print 'Period range:', pp['period@binary']

    output_dir = 'Binaries_Phoebe1_MCMC_allparams'
    try:
        os.makedirs(output_dir)
    except:
        pass
    os.chdir(output_dir)

    try:
        os.makedirs(star)
    except:
        pass
    
    os.chdir(star)
    run(star, init_sets[set][2], adjpars, priors, state, init_sets[set][0], init_sets[set][1], set)
    os.chdir('..')    
