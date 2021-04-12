# -*- encoding: utf-8 -*-
#!/usr/bin/python


import numpy as np
import phoebeBackend as phb  # so it doesn't get confused with the new phoebe
import pickle
import sys
import scipy.stats as st
#from phoebe.atmospheres.roche import binary_potential
#import phoebe.algorithms.marching as marching
#from phoebe.utils.coordinates import cart2spher_coord, spher2cart_coord

import matplotlib
matplotlib.use('Agg')
#matplotlib.style.use('classic')
import matplotlib.pyplot as plt 
import os
import gc
import operator
#import triangle
import corner

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import math
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.coordinates import SkyCoord as FK5
from astropy import units as unit
from astropy.time import Time
from astropy import coordinates as coord
from astropy import stats
import commands

hjd0_T = {'V1898 Cyg': [2448501.213, 1.513123, '21 03 54', '46 19 50', 'V 7.82', '3x20min', 'P = 1.51', 'Eclipsing binary of Algol type'], 
          'V394 Vul': [245287.4124, 3.080305, '19 49 03', '20 25 31', 'V 8.73', '3x30min', 'P = 3.08', 'Eclipsing binary'], 
          'V417 Aur': [2448500.5262, 1.86553, '05 13 32', '35 39 11', 'V 7.90', '3x20min', 'P = 1.86', 'Eclipsing binary of Algol type'], 
          'V455 Aur': [2453383.2815, 3.14578, '06 28 55', '52 07 33', 'V 7.90', '3x20min', 'P = 27.0', 'Eclipsing binary of Algol type'], 
          'V994 Her': [2452436.4213, 2.0832179, '18 27 46', '24 41 51', 'V 7.14', '3x20min', 'P = 2.08', 'Eclipsing binary of Algol type'], 
          'CI CVn': [2452723.4853, 0.8158745, '13 13 33', '47 47 52', 'V 9.37', '3x30min', 'P = 0.81', 'Eclipsing binary of beta Lyr type'], 
          'CN Lyn': [2453318.6229, 1.95551, '08 01 37', '38 44 57', 'V 9.06', '3x30min', 'P = 1.95', 'Eclipsing binary of Algol type'], 
          'DT Cam': [2453405.3478, 7.0662638, '05 13 58', '56 30 29', 'V 8.13', '3x30min', 'P = 7.06', 'Eclipsing binary'], 
          'DV Cam': [2453373.4808, 6.6785, '05 19 27.9', '58 07 02.5', 'V 6.119', '3x20min', 'P = 6.67', 'Eclipsing binary of Algol type'], 
          'EQ Boo': [2453410.6352, 5.435355, '14 52 26', '17 57 23', 'V 8.80', '3x30min', 'P = 5.43', 'Eclipsing binary of Algol type'], 
          'GK Dra': [2452000.498, 9.97415, '16 45 41', '68 15 30', 'V 8.77', '3x30min', 'P = 9.97', 'Eclipsing binary of Algol type'], 
          'GZ Dra': [2453171.4359, 2.253363, '18 12 41', '54 46 07', 'V 9.49', '3x30min', 'P = 2.25', 'Eclipsing binary'], 
          'TV LMi': [2452655.3776, 8.47799, '09 55 46', '37 11 42', 'V 8.99', '3x30min', 'P = 8.47', 'Eclipsing binary of Algol type'],
          #'TYC 5227-1023-1': [2457003.325, 4.306193, '22 00 52.6', '-03 42 12.4', 'V 11.86', '3x30min'],
      #'V1695 Aql': [2457634.7053, 0.4128296, '19 38 22.2982', '-03 32 37.142', 'V 11.0', '1x30min'],
}



comp_pars = ["phoebe_mass1", "phoebe_mass2", "phoebe_radius1", "phoebe_radius2", "phoebe_mbol1", "phoebe_mbol2", "phoebe_logg1", "phoebe_logg2", 'distance']
SHOW_PLOT = False

#https://arxiv.org/pdf/1510.07674.pdf, IAU 2015 Resolution B3 on Recommended Nominal Conversion Constants for Selected Solar and Planetary Properties
RSun = 6.957*(10**8)
Lsun = 3.828*(10**26)
Tsun = 5772
GMsun = 1.3271244*(10**20)

G = 6.67408*(10**-11) # 2014 CODATA

# http://maia.usno.navy.mil/NSFA/NSFA_cbe.html
c = 299792.458

SIGMA = 5.670367*(10**-8) # 2014 CODATA

# Using the IAU 2012 Resolution B2 definition of the astronomical unit, the parsec corresponds to 3.085 677 581
KPCTOM = 3.085677581*(10**19)

day = 86400.0


def show_results(star, phoebe_file, prefix, adjpars, labels, burn_cut, lnprob_lim_low, nwalkers=64, niter=1000,
                 teff1=np.nan, dteff1=np.nan, dparsec=np.nan, ddparsec=np.nan):
    phb.init()
    phb.configure()
    phb.open('/home/klemen/phoebe1/'+star+'/'+phoebe_file)

    
    # REVIEW MCMC RESULTS
    # ===================
    if True:

        # burn_cut = -700
        burn_cut_up = -1
        lnprob_lim_high = 0
        # lnprob_lim_low = -2e9

        #samples = np.loadtxt(phoebe_file+prefix+'.chain.mcmc')
        samples_all = np.genfromtxt(phoebe_file + prefix + '.mcmc')
        samples = samples_all[:, 1:]
        samples_by_walkers = []
        #print samples_all

        if True:
            # show walkers convergence            
            for i in range(nwalkers):   
                #print len(samples_all[i::nwalkers][:, -1]), np.sum(np.isfinite(samples_all[i::nwalkers][:, -1])), samples_all[i::nwalkers][:, -1]     
                plt.plot(samples_all[i::nwalkers][:, -1], lw=0.5)
            if SHOW_PLOT:           
                plt.show()

            plt.ylim((np.nanpercentile(samples_all[:, -1], 35), np.nanmax(samples_all[:, -1])))
            plt.tight_layout()
            plt.savefig("%s_walkers_mcmc.png" % (star), dpi=250)
            plt.close()

        
        labels.extend([i.replace('phoebe_', '') for i in comp_pars])
        labels.extend(['r1', 'r2', 'K1', 'K2'])
        print labels

        for i in range(nwalkers):

            walkers_burnin_cut = samples_all[i::nwalkers][burn_cut:burn_cut_up, -1]

            #print walkers_burnin_cut
            print i, len(walkers_burnin_cut), len(walkers_burnin_cut[(walkers_burnin_cut > lnprob_lim_low) & (walkers_burnin_cut < lnprob_lim_high)])
            print np.nanmin(walkers_burnin_cut), np.nanmax(walkers_burnin_cut), np.sum(walkers_burnin_cut > lnprob_lim_low), np.sum(walkers_burnin_cut < lnprob_lim_high)
            
            if len(walkers_burnin_cut) == len(walkers_burnin_cut[(walkers_burnin_cut > lnprob_lim_low) & (walkers_burnin_cut < lnprob_lim_high)]):
                samples_by_walkers.append(samples[i::nwalkers][burn_cut:burn_cut_up])
                #plt.plot(samples_all[i::64][:, -1])
        #if SHOW_PLOT:
        #    plt.show()

        best_result_by_lnprob = sorted(samples_all, key = operator.itemgetter(-1), reverse = True)[0]
        
        print np.array(samples_by_walkers).shape
        samples = np.concatenate(np.transpose(np.array(samples_by_walkers), (1,0,2)))  

        #print samples[:5]  
        #samples = np.array(sorted(samples, key = operator.itemgetter(-1), reverse = True))[:, 0:len(labels)]
        samples = samples[:, 0:len(labels)]
        #print samples[:5] 
           
        
        averages = zip(np.average(samples, axis=0), np.std(samples, axis=0))
        averages = np.array(averages)
        percentiles_posterior = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
        #percentiles = averages

    
        # print results
        print "\nMCMC RESULTS:\n"
        mcmc_results = {}

        for x in zip(labels, percentiles_posterior):
            mcmc_results[x[0]] = x[1]
            print x[0], x[1]
        

        # save last positions of all walkers for LC plotting
        last_walkers_values = []
        
        for j in range(nwalkers):
            percentiles = []
            #print j, samples[-j][10]
            for i in samples[-j]:
                percentiles.append((i, 0, 0))
            last_walkers_values.append(percentiles)
 
    
        # show values from phoebe parameter file
        print "\nVALUES FROM PHOEBE PAR FILE:\n"
        for par in [["phoebe_hla", 0], ["phoebe_hla", 1], ["phoebe_cla", 0], ["phoebe_cla", 1], ["phoebe_plum1", 0], ["phoebe_plum2", 0], "phoebe_mass1", "phoebe_mass2", "phoebe_radius1", "phoebe_radius2", "phoebe_mbol1", "phoebe_mbol2", "phoebe_logg1", "phoebe_logg2", "phoebe_sbr1", "phoebe_sbr2", "phoebe_vol1", "phoebe_vol2"]:
            if len(par) == 2:
                #print par[0], par[1]
                print "%s %s: %s" %(par[0], par[1], phb.getpar(par[0], par[1]))
            else:
                print "%s: %s" %(par, phb.getpar(par))
    
    
    # COMPUTE NON-FITTED PARAMETERS
    # =============================
    if True:

        for i, j in enumerate(adjpars):
            phb.setpar(j[0], float(percentiles_posterior[i][0]), j[1])    
        
        # Using these new values, update LD coefficients:
        phb.updateLD()
        chi2lc1 = -0.5*phb.cfval('lc', 0, False)
        chi2lc2 = -0.5*phb.cfval('lc', 1, False)
        chi2rv1 = -0.5*phb.cfval('rv', 0)
        chi2rv2 = -0.5*phb.cfval('rv', 1)
        lnp = (chi2lc1 + chi2lc2 + chi2rv1 + chi2rv2)
        print lnp, chi2lc1, chi2lc2, chi2rv1, chi2rv2
        
        phb.save('phoebe_solution6')

        # show computed values after update
        print "\nCOMPUTED VALUES:\n"
        for par in ["phoebe_mass1", "phoebe_mass2", "phoebe_radius1", "phoebe_radius2", "phoebe_mbol1", "phoebe_mbol2", "phoebe_logg1", "phoebe_logg2", "phoebe_sbr1", "phoebe_sbr2", "phoebe_vol1", "phoebe_vol2"]:
            if len(par) == 2:
                print "%s %s: %s" %(par[0], par[1], phb.getpar(par[0], par[1]))
            else:
                print "%s: %s" %(par, phb.getpar(par))
    

    
    # PRINT MCMC POSTERIORS
    # =====================
    if True:
        # http://corner.readthedocs.io/en/latest/api.html
        nicer_labels = {'P':'$P$', 'hjd0':'$T_0$', 'K1':'$K_1$', 'K2':'$K_2$', 'a':'$a$', 'q':'$q$', 'i':'$i$', 'e':'$e$', 'teff2':'$T_2$', 'omega1':'$\Omega_1$', 'omega2':'$\Omega_2$', 'mass1':'$M_1$', 'mass2':'$M_2$', 'radius1':'$R_1$', 'radius2':'$R_2$', 'mbol1':'M$_{bol, 1}$', 'mbol2':'M$_{bol, 2}$', 'logg1':'log $g_1$', 'logg2':'log $g_2$', 'distance':'$d$', 'hla1':'hla1', 'hla2':'hla2', 'vga':'$V_{sys}$', 'per0':'per0'}
        printed_labels = ['hjd0', 'P', 'a', 'q', 'i', 'e', 'teff2', 'omega1', 'omega2', 'hla1', 'hla2', 'vga', 'mass1', 'mass2', 'radius1', 'radius2', 'per0']     
        
        # print only available labels
        printed_labels = [pl for pl in printed_labels if pl in labels]        

        print samples[0], labels, averages
        print len(samples[0]), len(labels), len(averages)
        new_samples = []
        new_averages = []
        for lab in printed_labels:
            new_samples.append(samples[:, labels.index(lab)])
            new_averages.append(averages[labels.index(lab)])
            print lab, samples[:, labels.index(lab)]
        new_samples = np.array(new_samples).T
        new_averages = np.array(new_averages)

        rang = []
        for sam in new_samples.T:
          if list(sam)[1:] == list(sam)[:-1]:
            rang.append((sam[0]*0.8, sam[0]*1.2))
          else:
            rang.append((min(sam), max(sam)))
        
        fig = corner.corner(new_samples, range=rang, quantiles=[0.16, 0.5, 0.84], labels=[nicer_labels[lab] for lab in printed_labels], label_kwargs={"fontsize": 18}, show_titles=True, title_fmt='.2f', title_kwargs={"fontsize": 13}, truths=new_averages[:, 0])#, labels=[r"$x$", r"$y$", r"$\log \alpha$", r"$\Gamma \, [\mathrm{parsec}]$"], truths=[0.0, 0.0, 0.0], quantiles=[0.16, 0.5, 0.84], show_titles=True, title_args={"fontsize": 12} labels_args={"fontsize": 40})
        #plt.tight_layout()
        
        #plt.tick_params(axis='y', which='both', direction='in')
        #fig.subplots_adjust(hspace=0, wspace=0)
        #fig = plt.figure(figsize=(10,15))
        fig.set_size_inches(9*1.8, 11*1.8)
        fig.subplots_adjust(left=0.09, top=0.95, bottom=0.1)
        # fig.tight_layout()

        if SHOW_PLOT:
            plt.show()
        
        fig.savefig("%s_triangle_mcmc.png" % (star), dpi=150)
        #fig.savefig("triangle%s.pdf" % (prefix))
        #fig.savefig("triangle"+prefix+".eps")
        plt.close(fig)

    # COMPUTE RV AND LC CURVE
    # =======================
    if True:

        # LC from median of posterior distr.
        '''
        for i, j in enumerate(adjpars):
            print j[0], float(percentiles_posterior[i][0]), j[1]
            phb.setpar(j[0], float(percentiles_posterior[i][0]), j[1])             
        phb.updateLD() 
        chi2lc1 = phb.cfval('lc', 0, False) #/ 182.
        chi2lc2 = phb.cfval('lc', 1, False) #/ 182.
        #return chi2lc1 + chi2lc2
        chi2rv1 = phb.cfval('rv', 0) #/ 8.
        chi2rv2 = phb.cfval('rv', 1) #/ 8.

        print chi2lc1, chi2lc2, chi2rv1, chi2rv2

        phb.save('phoebe_solution6')
        '''
        # Compute and display a RV curve:
        phs = tuple(np.linspace(-0.25, 1.25, 1.5/0.001 + 1))
        rv1 = phb.rv1(phs, 0)
        rv2 = phb.rv2(phs, 1)
        phs = np.array(phs)

        

        #times, rvs, sigmas = np.loadtxt('DT_Cam/spec/RV_primary.txt', unpack=True, dtype='float')        
        #times, rvs, sigmas = np.loadtxt('DT_Cam/spec/RV_secondary.txt', unpack=True, dtype='float')   

        # ADJUST LC phased by params of the solution
        hjd0 = hjd0_T[star.replace('_', ' ')][0]#float(mcmc_results['hjd0'][0])
        T = hjd0_T[star.replace('_', ' ')][1]#float(mcmc_results['P'][0])
        print "hjd0, P: ", hjd0, T
        
        RV1 = np.loadtxt('/home/klemen/phoebe1/%s/RV_primary_%s.txt'%(star, star), dtype='float')
        RV2 = np.loadtxt('/home/klemen/phoebe1/%s/RV_secondary_%s.txt'%(star, star), dtype='float')

        phot_phases1 = np.array([phase_from_MJD(ph, hjd0, T) for ph in RV1[:, 0]])
        phot_phases2 = np.array([phase_from_MJD(ph, hjd0, T) for ph in RV2[:, 0]])

        RV1[:, 0] = phot_phases1
        RV2[:, 0] = phot_phases2
        
        RV1 = np.concatenate( (np.array([RV1[:, 0], RV1[:, 1], RV1[:, 2]]).T, np.array([RV1[:, 0]-1, RV1[:, 1], RV1[:, 2]]).T, np.array([RV1[:, 0]+1, RV1[:, 1], RV1[:, 2]]).T) )
        RV2 = np.concatenate( (np.array([RV2[:, 0], RV2[:, 1], RV2[:, 2]]).T, np.array([RV2[:, 0]-1, RV2[:, 1], RV2[:, 2]]).T, np.array([RV2[:, 0]+1, RV2[:, 1], RV2[:, 2]]).T) ) 


        if True:
            fig = plt.figure()
            fig.set_size_inches(18,12)

            plt.plot(phs, rv1, color='red')
            plt.plot(phs, rv2, color='blue')    
            plt.errorbar(RV1[:, 0].astype(float), RV1[:, 1].astype(float), yerr=RV1[:, 2].astype(float), fmt='o', c='red', label='primary', markersize=8)
            plt.errorbar(RV2[:, 0].astype(float), RV2[:, 1].astype(float), yerr=RV2[:, 2].astype(float), fmt='o', c='blue', label='secondary', markersize=8)    
            
            plt.legend(loc='lower left', fontsize=20)
            plt.xlim([-0.1, 1.1])
            plt.xlabel('phase', fontsize=20)
            plt.ylabel('RV', fontsize=20)
            ax = fig.gca()
            ax.set_xticks(np.arange(0.0, 1.0, 0.25))
            #ax.set_yticks(np.arange(-100., 100., 10))
            plt.grid(ls='--', lw=0.3, alpha=0.2, c='black')
            if SHOW_PLOT:
                plt.show()
            
            plt.tight_layout()
            fig.savefig("%s_RV_mcmc.png" % (star), dpi=250)
            #fig.savefig("RV%s.pdf" % (prefix))
            plt.close(fig)

        # Compute and display a LC curve:
        #phs = tuple(np.linspace(-0.6, 0.6, 1.2/0.001 + 1))
        phs = tuple(np.linspace(-0.25, 1.25, 1.5/0.001 + 1))

        m0 = float(phb.getpar('phoebe_mnorm'))
        f0 = 1.0

        # assuming that the LC data were correctlly added to Phoebe software
        lc1_post = phb.lc(phs, 0)  # B band
        lc2_post = phb.lc(phs, 1)  # V band
        
        phs = np.array(phs)

        LC1 = np.loadtxt('/home/klemen/phoebe1/%s/Data_%s_B_new.txt'%(star, star), dtype='float')
        LC2 = np.loadtxt('/home/klemen/phoebe1/%s/Data_%s_V_new.txt'%(star, star), dtype='float')

        phot_phases1 = np.array([phase_from_MJD(ph, hjd0, T) for ph in LC1[:, 0]])
        phot_phases2 = np.array([phase_from_MJD(ph, hjd0, T) for ph in LC2[:, 0]])

        LC1[:, 0] = phot_phases1
        LC2[:, 0] = phot_phases2

        LC1_diff2 = phb.lc(tuple(LC1[:, 0]), 0)  
        LC2_diff2 = phb.lc(tuple(LC2[:, 0]), 1) 

        LC1[:, 0] = np.array([x < 0 and x+1 or x for x in LC1[:, 0]])
        LC2[:, 0] = np.array([x < 0 and x+1 or x for x in LC2[:, 0]])
        LC1 = np.concatenate( (np.array([LC1[:, 0], LC1[:, 1]]).T,
                               np.array([LC1[:, 0]-1, LC1[:, 1]]).T,
                               np.array([LC1[:, 0]+1, LC1[:, 1]]).T) )
        LC2 = np.concatenate( (np.array([LC2[:, 0], LC2[:, 1]]).T, np.array([LC2[:, 0]-1, LC2[:, 1]]).T, np.array([LC2[:, 0]+1, LC2[:, 1]]).T) ) 


        LC1_diff = phb.lc(tuple(LC1[:, 0]), 0)  
        LC2_diff = phb.lc(tuple(LC2[:, 0]), 1) 

        if lc1_post[0] < 5:                
            print "flux, not magnitudes"
            lc1_post = tuple(-2.5 * np.log10(np.array(lc1_post)/f0) + m0)
            lc2_post = tuple(-2.5 * np.log10(np.array(lc2_post)/f0) + m0)

            LC1_diff = -2.5 * np.log10(np.array(LC1_diff)/f0) + m0
            LC2_diff = -2.5 * np.log10(np.array(LC2_diff)/f0) + m0 

            LC1_diff2 = -2.5 * np.log10(np.array(LC1_diff2)/f0) + m0
            LC2_diff2 = -2.5 * np.log10(np.array(LC2_diff2)/f0) + m0 

        LC1_diff = LC1[:, 1] - LC1_diff
        LC2_diff = LC2[:, 1] - LC2_diff

        if True:
            fig = plt.figure()
            fig.set_size_inches(18,12)

            plt.plot(phs, lc1_post, color='blue')
            plt.plot(phs, lc2_post, color='green')    
            
            plt.scatter(LC1[:, 0].astype(float), LC1[:, 1].astype(float), s=1, c='blue', label='Johnson B')    
            plt.scatter(LC2[:, 0].astype(float), LC2[:, 1].astype(float), s=1, c='green', label='Johnson V')
            
            plt.legend(loc='lower center', fontsize=20)
            plt.xlim([-0.1, 1.1])
            plt.xlabel('phase', fontsize=20)
            plt.ylabel('mag', fontsize=20)
            ax = fig.gca()
            ax.set_xticks(np.arange(0.0, 1.0, 0.25))
            plt.gca().invert_yaxis()
            #ax.set_yticks(np.arange(-100., 100., 10))
            plt.grid(ls='--', lw=0.3, alpha=0.2, c='black')
            if SHOW_PLOT:
                plt.show()
            
            plt.tight_layout()
            fig.savefig("%s_LC_mcmc.png" % (star), dpi=250)
            #fig.savefig("LC%s.pdf" % (prefix))
            plt.close(fig)
   
     
    # PRINT PARAMETERS FOR PDF
    # ========================
    if True:

        mcmc_results['teff1'] = (teff1,) #(teff1, dteff1)
        #mcmc_results['dist'] = (dparsec, ddparsec)

        pars = ['P', 'hjd0', 'K1', 'K2', 'a', 'vga', 'q', 'i', 'e', 'teff2', 'omega1', 'omega2', 'r1', 'r2', 'radius1', 'radius2', 'mass1', 'mass2', 'mbol1', 'mbol2', 'logg1', 'logg2', 'distance']
        
        # use only fitted labels
        i_use = [ip for ip, pl in enumerate(pars) if pl in labels]

        pars = np.array(pars)[i_use]
        name = np.array(['\sl P', '\sl T$_{\\rm 0}$', '\sl $K_1$', '\sl $K_2$', '\sl a', '\sl V$_{\gamma }$', '\sl q = $\\frac{m_2}{m_1}$', '\sl i', '\sl e', '\sl T$_{2}$', '\sl $\Omega_{1}$', '\sl $\Omega_{2}$', '$r_1$', '$r_2$', '$R_1$', '$R_2$', '$M_1$', '$M_2$', 'M$_{bol, 1}$', 'M$_{bol, 2}$', 'log $g_1$', 'log $g_2$', 'distance'])[i_use]
        desc = np.array(['(d)', '(HJD)', '(km sec$^{-1}$)', '(km sec$^{-1}$)', '(R$_{\odot }$)', '(km sec$^{-1}$)', '(deg)', '', '', '(K)', '', '', '($R_1$/{\sl a})', '($R_2$/{\sl a})', '(R$_\odot$)', '(R$_\odot$)', '(M$_\odot$)', '(M$_\odot$)', '', '', '(cgs)', '(cgs)', '(pc)'])[i_use]
        formatt = np.array(['%.6f', '%.4f', '%.1f', '%.1f', '%.4f', '%.2f', '%.4f', '%.2f', '%.5f', '%.0f', '%.2f', '%.2f', '%.4f', '%.4f', '%.3f', '%.3f', '%.4f', '%.4f', '%.2f', '%.2f', '%.2f', '%.2f', '%.0f'])[i_use]
        
        print pars
        print i_use

        if False:
            #### SHOULDN'T HAVE TO DO THAT, SOMETHING WRONG IN .mcmc FILE FOR intermediate COMPUTED PARAMETERS ####
            #### FIX FIX FIX ####
            
            
            mcmc_results['mass1'] = (phb.getpar('phoebe_mass1'), 0)
            mcmc_results['mass2'] = (phb.getpar('phoebe_mass2'), 0)
            mcmc_results['mbol1'] = (phb.getpar('phoebe_mbol1'), 0)
            mcmc_results['mbol2'] = (phb.getpar('phoebe_mbol2'), 0)
            mcmc_results['radius1'] = (phb.getpar('phoebe_radius1'), 0)
            mcmc_results['radius2'] = (phb.getpar('phoebe_radius2'), 0)
            mcmc_results['logg1'] = (phb.getpar('phoebe_logg1'), 0)
            mcmc_results['logg2'] = (phb.getpar('phoebe_logg2'), 0)

            sma = mcmc_results['a'][0]
            q = mcmc_results['q'][0]
            P = mcmc_results['P'][0]
            inc = mcmc_results['i'][0]
            ecc = mcmc_results['e'][0]
            sma2 = sma / (1.+q)
            sma1 = sma - sma2
            mcmc_results['K1'] = ((2*np.pi*sma1*RSun*np.sin(inc*np.pi/180.)/P/day/np.sqrt(1.-ecc*ecc)), 0)
            mcmc_results['K2'] = ((2*np.pi*sma2*RSun*np.sin(inc*np.pi/180.)/P/day/np.sqrt(1.-ecc*ecc)), 0)
            mcmc_results['r1'] = mcmc_results['radius1'] / sma
            mcmc_results['r2'] = mcmc_results['radius2'] / sma
            
        
        with open('parameters_' + phoebe_file + prefix + '.tex', 'w') as f:
            f.write("\documentclass[10pt]{article} \n")
            f.write("\pagestyle{empty} \n")
            f.write("\\begin{document} \n")
            f.write("\\begin{table} \n")
            f.write("\\begin{tabular}{l l r l } \n")
            f.write("\hline  \n")                           
            f.write("&&&\\\\ \n")

            for i, par in enumerate(pars):
                if len(mcmc_results[par]) == 3:
                    
                    if par == 'teff2':
                        f.write(('{%s} & %s & '+formatt[i]+' & $^{+'+formatt[i]+'}_{-'+formatt[i]+'}$ \\\\[1.3ex] \n') % ('\sl T$_{1}$ - T$_{2}$', '(K)', mcmc_results['teff1'][0] - mcmc_results[par][0], mcmc_results[par][1], mcmc_results[par][2]))
                    else:
                        f.write(('{%s} & %s & '+formatt[i]+' & $^{+'+formatt[i]+'}_{-'+formatt[i]+'}$ \\\\[1.3ex] \n') % (name[i], desc[i], mcmc_results[par][0], mcmc_results[par][1], mcmc_results[par][2]))
                else:
                    f.write(('{%s} & %s & '+formatt[i]+' & $\pm$ '+formatt[i]+'\\\\[1.3ex] \n') % (name[i], desc[i], mcmc_results[par][0], mcmc_results[par][1]))

            f.write("&&&\\\\ \n")
            f.write("\hline \n")
            f.write("\end{tabular} \n")   
            f.write("\end{table} \n")
            f.write("\end{document} \n")

        os.system('pdflatex ' + 'parameters_' + phoebe_file + prefix + '.tex')
        print "PDF created"

        print "Accuracies: M1: %s, M2: %s, R1: %s, R2: %s" % (max(abs(mcmc_results['mass1'][1]), abs(mcmc_results['mass1'][2])) / mcmc_results['mass1'][0],                 max(abs(mcmc_results['mass2'][1]), abs(mcmc_results['mass2'][2])) / mcmc_results['mass2'][0],                      max(abs(mcmc_results['radius1'][1]), abs(mcmc_results['radius1'][2])) / mcmc_results['radius1'][0],                       max(abs(mcmc_results['radius2'][1]), abs(mcmc_results['radius2'][2])) / mcmc_results['radius2'][0])
        if 'P' in mcmc_results:
            print "rotational velocities: 1 %s, 2 %s" % (RSun*2*math.pi*mcmc_results['radius1'][0] / (mcmc_results['P'][0]*24*3600), RSun*2*math.pi*mcmc_results['radius2'][0] / (mcmc_results['P'][0]*24*3600))

 
    return

def add_subplot_axes(fig, ax,rect,axisbg='w'):
    #fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax



def phase_from_MJD(hjd, hjd0, T, short=False):    
  if short:
    return float("%.4f" % ((hjd-hjd0)/T - int((hjd-hjd0)/T)))
  else:
    return (hjd-hjd0)/T - int((hjd-hjd0)/T)


if __name__ == '__main__':
    # BASIC SETUP:
        
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
                   0,  # seconda00ry extinction
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
        n_last_use = int(in_args[3])
    else:
        n_last_use = 300
    
    init_sets = {    
    'a': [walkers, 0, 'TV_LMi_partial_results', what_to_fit, 'TV_LMi', -n_last_use, -4e8, 5660],
    'b': [walkers, 0, 'V394_Vul_partial_results', what_to_fit, 'V394_Vul', -n_last_use, -4e8, 8750],
    'c': [walkers, 0, 'V455_Aur_partial_results', what_to_fit, 'V455_Aur', -n_last_use, -4e8, 7050],
    'd': [walkers, 0, 'GK_Dra_partial_results', what_to_fit, 'GK_Dra', -n_last_use, -4e8, 6050],
    'e': [walkers, 0, 'DT_Cam_partial_results', what_to_fit, 'DT_Cam', -n_last_use, -4e8, 9040],
    }

    adjpars = [['phoebe_hjd0', 0], ['phoebe_period', 0], ['phoebe_sma', 0], ['phoebe_rm', 0], ['phoebe_incl', 0], ['phoebe_ecc', 0], ['phoebe_teff2', 0], ['phoebe_pot1', 0], ['phoebe_pot2', 0], ['phoebe_hla', 0], ['phoebe_hla', 1], ['phoebe_el3', 0], ['phoebe_el3', 1], ['phoebe_extinction', 0], ['phoebe_extinction', 1], ['phoebe_met1', 0], ['phoebe_met2', 0], ['phoebe_vga', 0], ['phoebe_perr0', 0]]
    labels = [r"hjd0", r"P", r"a", r"q", r"i", r"e", r"teff2", r"omega1", r"omega2", r"hla1", r"hla2", r"el31", r"el32", r"extinc1", r"extinc2", r"met1", r"met2", r"vga", r"per0"]
    
    mask = np.array(init_sets[set][3]) == 1
    adjpars = np.array(adjpars)[mask]
    adjpars = [[x[0], int(x[1])] for x in adjpars]
    labels = list(np.array(labels)[mask])

    star = init_sets[set][4]
    output_dir = 'Binaries_Phoebe1_MCMC_allparams'
    os.chdir(output_dir)
    os.chdir(star)

    show_results(init_sets[set][4], init_sets[set][2], set, adjpars, labels, init_sets[set][5], init_sets[set][6], init_sets[set][0], init_sets[set][1],
                 teff1=init_sets[set][7])
