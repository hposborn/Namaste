#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
:py:mod:`Namaste.py` - Single transit fitting code
-------------------------------------
'''
import autograd.numpy as np2
import matplotlib
#matplotlib.use('Agg')
import pylab as plt
plt.ioff()
import scipy.optimize as optimize
from os import sys, path
import datetime
import logging
import pandas as pd

import emcee
import celerite
from .planetlib import *
from .Crossfield_transit import *
from scipy import stats
import scipy.optimize as opt
import pickle
import autograd.numpy.numpy_boxes

import os
Namwd = path.dirname(path.realpath(__file__))

logger = logging.getLogger('namaste')
logger.setLevel(level=logging.DEBUG)
fh = logging.FileHandler(filename=Namwd+'/namaste_runlog.log')
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s'))
logger.addHandler(fh)

#logger.basicConfig(filename=Namwd+'/namaste_runlog.log', level=logger.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logger = logger.getLogger(__name__)

#How best to do this...
#
# make lightcurve a class
# make candidate a class
# make star a class
# make planet a class

# Give candidate a star and planet and lightcurve
# Use the planet to generate the lightcurve model
# Use the star to generate priors
# Give candidate data in the form of a lightcurve - can switch out for CoRoT, Kepler, etc.


class Settings():
    '''
    The class that contains the model settings
    '''
    def __init__(self, **kwargs):
        self.GP = True                    # GP on/off
        self.nopt = 25                    # Number of optimisation steps for each GP and mono before mcmc
        self.nsteps = 12500               # Number of MCMC steps
        self.npdf = 6000                  # Number of samples in the distributions from which to doing calcs.
        self.nwalkers = 24                # Number of emcee walkers
        self.nthreads = 8                 # Number of emcee threads
        self.ndurswindow = 6             # Number of Tdurs either side of the transit to be fitted.
        self.anomcut = 3.75               # Number of sigma above/below to count as an outlier.
        #self.binning = -1                # Days to bin to. -1 dictates no bin
        #self.error = False               # No error detected
        #self.error_comm = ''             # Error description
        #self.use_previous_samples = False# Using samaples from past run
        self.fitsloc = './InputFiles/'     # Storage location to load stuff
        self.outfilesloc = './Outputs/'   # Storage location to save stuff
        self.cadence = 0.0204318          # Cadence. Defaults to K2
        self.kernel = 'quasi'             # Kernel for use in GPs
        self.verbose = True               # Print statements or not...
        self.mission = 'K2'               # Mission
        self.fit_tdur = False             # Use Tdur, rather than velocity, to perform fits
        self.fit_vel = True               # Use velocity, rather than Tdur, to perform fits
        self.update_logger()

    def update_logger(self):
        if self.verbose and logging.StreamHandler not in [type(hand) for hand in logger.handlers]:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(lineno)d:%(filename)s(%(process)d) - %(message)s'))
            logger.addHandler(sh)
        elif not self.verbose:
            logger.setLevel(level=logging.WARNING)


    def update(self, **kwargs):
        '''
        Adding new settings...
        '''
        self.GP = kwargs.pop('GP', self.GP)
        self.nopt = kwargs.pop('nopt', self.nopt)
        self.nsteps = kwargs.pop('nsteps', self.nsteps)
        self.npdf = kwargs.pop('npdf', self.npdf)
        self.nwalkers = kwargs.pop('nwalkers', self.nwalkers)
        self.nthreads = kwargs.pop('nthreads', self.nthreads)
        self.ndurswindow = kwargs.pop('ndurswindow', self.ndurswindow)
        self.anomcut = kwargs.pop('anomcut', self.anomcut)
        #self.binning = kwargs.pop('binning', self.binning)
        #self.error = kwargs.pop('error', self.binning)
        #self.error_comm = kwargs.pop('error_comm', self.binning)
        #self.use_previous_samples = kwargs.pop('use_previous_samples', self.binning)
        self.fitsloc = kwargs.pop('fitsloc', self.fitsloc)
        self.outfilesloc = kwargs.pop('outfilesloc', self.outfilesloc)
        self.cadence = kwargs.pop('cadence', self.cadence)
        self.kernel = kwargs.pop('kernel', self.kernel)
        self.verbose = kwargs.pop('verbose', self.verbose)
        if 'verbose' in kwargs:
            self.update_logger()
        self.mission = kwargs.pop('mission', self.mission)
        self.fit_tdur = kwargs.pop('fit_tdur', self.fit_tdur)
        self.fit_vel = kwargs.pop('fit_vel', self.fit_vel)
    def printall(self):
        print(vars(self))




class Star():
    def __init__(self, name, settings):
        logger.info(name+' - initialising Star object')
        self.objname = str(name)
        self.settings = settings
        #Initialising lists of monotransits and multi-pl
        self.meanmodels=[] #list of mean models to fit.
        self.fitdict={} #dictionary of model parameter PDFs to fit

    def exofop_dat(self):
        #Getting information from exofop...
        logger.info(self.objname+' - Getting information from exofop')
        sdic=ExoFop(int(self.objname))
        if 'radius' in sdic.columns:
            self.addRad(sdic['radius'],sdic['radius_err'],np.max([sdic['radius']*0.8,sdic['radius_err']]))
        else:
            raise ValueError("No radius")
        if 'teff' in sdic.columns:
            self.addTeff(sdic['teff'],sdic['teff_err'],sdic['teff_err'])
        else:
            raise ValueError("No teff")
        if 'mass' in sdic.columns:
            self.addMass(sdic['mass'],sdic['mass_err'],sdic['mass_err'])
        else:
            raise ValueError("No mass")
        if 'logg' in sdic.columns:
            self.addlogg(sdic['logg'],sdic['logg_err'],sdic['logg_err'])
        if 'feh' in sdic.columns:
            self.addfeh(sdic['feh'],sdic['feh_err'],sdic['feh_err'])
        if 'density' in sdic.columns:
            self.addDens(sdic['density'],sdic['density_err'],sdic['density_err'])
        else:
            self.addDens()

    def csvfile_dat(self,file,errorstyle=['ep','em']):
        #Collecting from CSV file, eg Best_Stellar_Params_nogriz
        logger.info(self.objname+' - Getting information from csv')
        if type(file) is not pd.DataFrame and type(file) is not pd.Series:
            df=pd.DataFrame.from_csv(file)
        if df.shape[1]==1:
            #1D file:
            row=df.T
        else:
            row=df.loc[df.epic==int(self.objname)]
        csvname=row.index.values[0]
        row=row.T.to_dict()[csvname]

        self.addRad(row['rad'],row['rad'+errorstyle[0]],row['rad'+errorstyle[1]])
        self.addTeff(row['teff'],row['teff'+errorstyle[0]],row['teff'+errorstyle[1]])
        self.addMass(row['mass'],row['mass'+errorstyle[0]],row['mass'+errorstyle[1]])
        self.addlogg(row['logg'],row['logg'+errorstyle[0]],row['logg'+errorstyle[1]])
        self.addfeh(row['feh'],row['feh'+errorstyle[0]],row['feh'+errorstyle[1]])
        if not pd.isnull(row['rho']):
            self.addDens(row['rho'],row['rho'+errorstyle[0]],row['rho'+errorstyle[1]])
        else:
            self.addDens()
	#avs	avsem	avsep	dis	disem	disep	epic	feh	fehem	fehep	input 2MASS	input BV
    #input SDSS	input_spec	logg	loggem	loggep	lum	lumem	lumep	mass	massem	massep
    #n_mod	prob	rad	radem	radep	rank	realspec	rho	rho_err	rhoem	rhoep	teff	teffem	teffep

    def addTeff(self,val,uerr,derr=None):
        self.steff = float(val)
        self.steffuerr = float(uerr)
        self.steffderr = self.steffuerr if type(derr)==type(None) else float(derr)
    def addRad(self,val,uerr,derr=None):
        self.srad = float(val)
        self.sraduerr = float(uerr)
        self.sradderr = self.sraduerr if type(derr)==type(None) else float(derr)
    def addMass(self,val,uerr,derr=None):
        self.smass = float(val)
        self.smassuerr = float(uerr)
        self.smassderr = self.smassuerr if type(derr)==type(None) else float(derr)
    def addlogg(self,val,uerr,derr=None):
        self.slogg = float(val)
        self.slogguerr = float(uerr)
        self.sloggderr = self.slogguerr if type(derr)==type(None) else float(derr)
    def addfeh(self,val,uerr,derr=None):
        self.sfeh = float(val)
        self.sfehuerr = float(uerr)
        self.sfehderr = self.sfehuerr if type(derr)==type(None) else float(derr)
    def addDens(self,val=None,uerr=None,derr=None):
        if val==None:
            sdenss=CalcDens(self,returnpost=False)#[self.teff,self.steffuerr,self.steffderr,self.srad,self.sraduerr,self.sradderr,
            pctns=np.percentile(sdenss,[15.865525393145707, 50.0, 84.13447460685429])
            self.sdens=pctns[1]
            self.sdensuerr=pctns[2]-pctns[1]
            self.sdensderr=pctns[1]-pctns[0]                       #self.smass,self.smassuerr,self.smassderr,self.slogg,self.slogguerr,self.sloggderr])
        else:
            #Density defined by others
            if float(val)>200:
                #Normalising to Solar density:
                val=float(val)/1410.0;uerr=float(uerr)/1410.0;derr=float(derr)/1410.0
            else:
                val=float(val);uerr=float(uerr);derr=float(derr)
            self.sdens = float(val)
            self.sdensuerr = float(uerr)
            self.sdensderr = self.sdensuerr if type(derr)==type(None) else float(derr)

    def addLightcurve(self,file):
        self.Lcurve=Lightcurve(file,self.objname,self.settings)
        self.mag=self.Lcurve.mag
        if self.settings.mission=="kepler" or self.settings.mission=='k2':
            self.wn=2.96e-4/np.sqrt(10**((14-self.mag)/2.514)) #2.42e-4 is the White Noise at 14th magnitude for Kepler.
        else:
            self.wn=np.percentile(abs(np.diff(self.Lcurve.lc[:,1])),40) #Using 40th percentile of the absolute differences.
        return self.Lcurve

    def EstLimbDark(self):
        if not hasattr(self,'sloggs'):
            self.PDFs()
        if self.settings.mission=="kepler" or self.settings.mission=='k2':
            LDs=getKeplerLDs(self.steffs,logg=self.sloggs,FeH=self.sfeh)
        elif self.settings.mission=="TESS":
            #logger.error("TESS not yet supported")
            raise ValueError("TESS not yet supported")
        else:
            #assume mission==Kepler
            LDs=getKeplerLDs(self.steffs,logg=self.sloggs,FeH=self.sfeh)

        pctns=np.percentile(LDs[0],[15.865525393145707, 50.0, 84.13447460685429])
        self.LD1s=LDs[:,0]
        self.LD1=pctns[1]
        self.LD1uerr=pctns[2]-pctns[1]
        self.LD1derr=pctns[1]-pctns[0]

        pctns=np.percentile(LDs[1],[15.865525393145707, 50.0, 84.13447460685429])
        self.LD2s=LDs[:,1]
        self.LD2=pctns[1]
        self.LD2uerr=pctns[2]-pctns[1]
        self.LD2derr=pctns[1]-pctns[0]

        self.initLD()

    def PDFs(self):
        if hasattr(self,'steff')*hasattr(self,'steffuerr')*hasattr(self,'steffderr'):
            self.steffs=noneg_GetAssymDist(self.steff,self.steffuerr,self.steffderr,nd=self.settings.npdf)
        if hasattr(self,'srad')*hasattr(self,'sraduerr')*hasattr(self,'sradderr'):
            self.srads =noneg_GetAssymDist(self.srad,self.sraduerr,self.sradderr,nd=self.settings.npdf)
        if hasattr(self,'smass')*hasattr(self,'smassuerr')*hasattr(self,'smassderr'):
            self.smasss=noneg_GetAssymDist(self.smass,self.smassuerr,self.smassderr,nd=self.settings.npdf)
        if hasattr(self,'slogg')*hasattr(self,'slogguerr')*hasattr(self,'sloggderr'):
            self.sloggs=GetAssymDist(self.slogg,self.slogguerr,self.sloggderr,nd=self.settings.npdf,returndist=True)
        elif hasattr(self,'srads')*hasattr(self,'smasss'):
            self.sloggs=np.log10(self.smasss/self.srads)+4.44
            #GetAssymDist(self.slogg,self.slogguerr,self.sloggderr,nd=self.settings.npdf,returndist=True)
        if hasattr(self,'sdens')*hasattr(self,'sdensuerr')*hasattr(self,'sdensderr'):
            self.sdenss=noneg_GetAssymDist(self.sdens,self.sdensuerr,self.sdensderr,nd=self.settings.npdf)
        #Adding sfeh if necessary
        if not hasattr(self,'sfeh'):
            self.sfeh=0.0

    def pdflist(self):
        if not hasattr(self,'steffs'):
            self.PDFs()
        if not hasattr(self,'LD1s'):
            self.EstLimbDark()
        return {'steffs':self.steffs,'srads':self.srads,'smasss':self.smasss,'sloggs':self.sloggs,'sdenss':self.sdenss,\
                'LD1s':self.LD1s,'LD2s':self.LD2s}

    def addGP(self,vector=None,overwrite=True):
        logger.info(self.objname+' - Adding GP')
        if self.settings.GP and (not hasattr(self, 'kern') or overwrite):
            if self.settings.kernel=='Real':
                self.kern=celerite.terms.RealTerm(log_a=np.log(np.var(self.Lcurve.lc[:,1])), log_c=-np.log(3.0),
                                                  bounds={'log_a':(-20,0.0),'log_c':(-3,4.5)})+\
                          celerite.terms.JitterTerm(np.log(self.wn),bounds=dict(log_sigma=(np.log(self.wn)-0.25,np.log(self.wn)+0.25)))
                if vector is not None:
                    self.kern.set_parameter_vector(vector)
                    self.kern.freeze_parameter('terms[1]:log_sigma')    # and freezing white noise
            elif self.settings.kernel=='quasi':
                self.kern= RotationTerm(np.log(np.var(self.Lcurve.lc[:,1])), np.log(0.25*self.Lcurve.range), np.log(2.5), 0.0,
                                   bounds=dict(
                                            log_amp=(-20.0, -2.0),
                                            log_timescale=(np.log(1.5), np.log(5*self.Lcurve.range)),
                                            log_period=(np.log(1.2), np.log(2*self.Lcurve.range)),
                                            log_factor=(-5.0, 5.0),
                                              )
                                    )+\
                           celerite.terms.JitterTerm(np.log(self.wn),
                                    bounds=dict(
                                            log_sigma=(np.log(self.wn)-0.25,np.log(self.wn)+0.25)
                                            )
                                    )
                if vector is not None:
                    self.kern.set_parameter_vector(vector)
                logger.info(self.objname+' - GP is '+' , '.join(list(self.kern.get_parameter_vector().astype(str))))
                #self.initgp={'log_amp':,'log_timescale':,'log_period':,'log_factor':,'log_sigma':}
                self.kern.freeze_parameter('terms[0]:log_factor')   # freezing log factor
                self.kern.freeze_parameter('terms[1]:log_sigma')    # and freezing white noise
            self.initgp={itrm:self.kern.get_parameter(itrm) for itrm in self.kern.get_parameter_names()}
        elif self.settings.GP:
            self.initgp={itrm:self.kern.get_parameter(itrm) for itrm in self.kern.get_parameter_names()}
        else:
            self.kern = None
            self.initgp = None

    def Optimize_GP(self):
        #Optimizing initial GP parameters, depending on GP supplied...
        '''
        #This optimizes the gaussian process on out-of-transit data. This is then held with a gaussian prior during modelling
        '''
        if not self.settings.GP:
            self.fitdict={}
            raise ValueError("GP is set as False.")
        logger.info(self.objname+' - optimizing GP')
        import scipy.optimize as op
        #Cutting transits from lc
        self.Lcurve.calc_mask(self.meanmodels)
        lc_trn=self.Lcurve.lc[self.Lcurve.lcmask]
        #logger.debug(self.objname+' -  mask: '+','.join(self.Lcurve.lcmask.astype(str)))

        logger.debug(self.objname+' -  nans in y: '+str(np.sum(np.isnan(lc_trn[:,1]))))

        lc_trn[:,1]/=np.nanmedian(lc_trn[:,1])#This half may be different in median from the full lc, so adjusting for this...

        if not hasattr(self,'gp'):
            self.addGP()

        self.kern.thaw_all_parameters()
        gp_notr=celerite.GP(kernel=self.kern,mean=1.0,fit_mean=True)
        gp_notr.compute(lc_trn[:,0],lc_trn[:,2])

        #logger.debug(self.objname+' -  y: '+','.join(lc_trn[:,1].astype(str)))
        logger.debug(self.objname+' - vect: '+','.join(gp_notr.get_parameter_vector().astype(str)))
        logger.debug(self.objname+' - bounds: '+' , '.join(['('+str(b[0])+' | '+str(b[1])+')' for b in gp_notr.get_parameter_bounds()]))
        logger.debug(self.objname+' - nll: '+str(neg_log_like(gp_notr.get_parameter_vector(),lc_trn[:,1],gp_notr)))

        #Initial results:
        init_res=op.minimize(neg_log_like, list(gp_notr.get_parameter_vector()), args=(lc_trn[:,1],gp_notr),jac=grad_neg_log_like)#, method="L-BFGS-B")#,jac=grad_neg_log_like)
        logger.debug(self.objname+' - init GP lpp is '+str(init_res.fun)+' using '+' , '.join(list(init_res.x.astype(str))))

        fails=0
        if self.settings.kernel=='quasi':
            # Run the optimization routine for a grid of size self.settings.nopt
            #log_amp, log_timescale, log_period, log_factor, log_sigma, mean = params
            iterparams= np.column_stack((np.random.normal(gp_notr.kernel.get_parameter_vector()[0],3.0,self.settings.nopt),
                                         np.random.uniform(1.2,np.log(1.5*self.Lcurve.range),self.settings.nopt),
                                         np.random.uniform(np.log(6*self.settings.cadence),np.log(1.5*self.Lcurve.range),self.settings.nopt),
                                         np.tile(0.0,self.settings.nopt),
                                         np.tile(np.log(self.wn),self.settings.nopt),
                                         np.tile(1.0,self.settings.nopt)))
        elif self.settings.kernel=='Real':
            #log_a, log_c = params
            iterparams=np.column_stack((np.random.normal(gp_notr.kernel.get_parameter_vector()[0],np.sqrt(abs(gp_notr.kernel.get_parameter_vector()[0])),self.settings.nopt),
                                        np.random.normal(gp_notr.kernel.get_parameter_vector()[1],np.sqrt(abs(gp_notr.kernel.get_parameter_vector()[1])),self.settings.nopt),
                                        np.tile(np.log(self.wn),self.settings.nopt),
                                        np.tile(1.0,self.settings.nopt)))

        suc_res=np.zeros(len(iterparams[0])+1)

        for n_p in np.arange(self.settings.nopt):
            vect=iterparams[n_p]
            print(vect)
            #gp_notr.kernel.set_parameter_vector(vect)
            try:
                #logger.debug("vect"+','.join(list(vect.astype(str))))
                result = op.minimize(neg_log_like, vect, args=(lc_trn[:,1], gp_notr), method="L-BFGS-B")#,jac=grad_neg_log_like)## jac=grad_nll,")
                #logger.debug(','.join(list(gp_notr.get_parameter_vector().astype(str))))
                if result.success:
                    #print("success,",result.fun)
                    suc_res=np.vstack((suc_res,np.hstack((result.x,result.fun))))
                else:
                    fails+=1
                #logger.debug(result.success+" - "+str(result.fun))
            except:
                #print("fail,",vect)
                fails+=1
        print(suc_res) if self.settings.verbose else logger.info(self.objname+" - succesful GP optimization with vector: "+','.join(list(suc_res.astype(str))))
        print(str(fails)+" failed attempts out of "+str(self.settings.nopt)) if self.settings.verbose else logger.info(self.objname+str(fails)+" failed attempts out of "+str(self.settings.nopt))

        if len(np.shape(suc_res))==1:
            logger.warning("No successful GP minimizations")
            raise ValueError("No successful GP minimizations")
        else:
            suc_res=suc_res[1:,:]
            bestres=suc_res[np.argmin(suc_res[:,-1])]

            gp_notr.set_parameter_vector(bestres[:-1])
            self.optimised_gp=gp_notr

            wn_factor = bestres[-3]-np.log(self.wn)
            self.optgp={itrm:gp_notr.get_parameter(itrm) for itrm in gp_notr.get_parameter_names()}

            # Update the kernel and print the final log-likelihood.
            for itrm in gp_notr.kernel.get_parameter_names()[:-1]:
                self.kern.set_parameter(itrm,gp_notr.kernel.get_parameter(itrm))
            if self.settings.kernel=='quasi':
                self.kern.freeze_parameter('terms[0]:log_factor')   # re-freezing log factor
            else:
                self.kern.freeze_parameter('terms[1]:log_sigma')    # and re-freezing white noise

            #Adding
            self.fitdict.update({'kernel:'+nm:np.random.normal(gp_notr.get_parameter('kernel:'+nm),1.5,self.settings.nwalkers) for nm in self.kern.get_parameter_names()})

            print("white noise changed by a factor of "+str(np.exp(wn_factor))[:4]) if self.settings.verbose else logger.info(self.objname+"white noise changed by a factor of "+str(np.exp(wn_factor))[:4])

        print("GP improved from ",init_res.fun," to ",bestres[-1]) if self.settings.verbose else logger.info(self.objname+"GP improved from "+init_res.fun+" to ",bestres[-1])
        '''return bestres[:-2] #mean_shift... is this indicative of the whole lightcurve or just this half of it?'''

    def AddMonotransit(self, tcen, tdur, depth, b=0.41,replace=True,name='mono1'):
        '''
        #Add's monotransit signal to a star
        Requires:
         * tcen (time of central transit, in the same units as the lightcurve)
         * tdur (duration of transit)
         * depth (depth of transit - in flux, not percent)
         * b (optional, impact parameter, defaults to most likely value; 0.41)
         * replace (optional, defauls to True. If False, should create a second monotransit)
         * name (optional. Defaults to mono1)
        '''

        if not hasattr(self,'steffs'):
            self.PDFs()

        #Adding monotransit classes to the star class... Up to four possible.

        if not hasattr(self,name) or replace:
            #self.LD1s, self.LD2s, self.sdenss,self.Lcurve.lc,
            setattr(self,name, Monotransit(tcen, tdur, depth, self.settings, name=str(self.objname)+'.1'))
            if not hasattr(self,'sdenss'):
                self.addDens()
            getattr(self,name).calcmaxvel(self.Lcurve.lc,self.sdenss)
            getattr(self,name).Optimize_mono(self.Lcurve.flatten(),self.LDprior.copy())
            getattr(self,name).SaveInput(self.pdflist())
            self.meanmodels+=[getattr(self,name)]
        '''
        while a<=5 and len(self.meanmodels)==initlenmonos:
            if not hasattr(self,'mono'+str(a)) or replace:
                setattr(self,'mono'+str(a)) = Monotransit(tcen, tdur, depth, self.settings, self.LD1s, self.LD2s, self.denss,self.lc, name=self.objname+'.'+str(a), b=b)
                exec("self.mono"+str(a)+".Optimize_mono(Lcurve.flaten())")
                exec("self.mono"+str(a)+".calcmaxvel(self.Lcurve.lc,self.sdenss)")
                exec("self.meanmodels+=[self.mono"+str(a)+"]")
            a+=1
        '''

    def AddNormalPlanet(self, tcen, tdur, depth, Period, b=0.41,replace=False):
        '''
        # This function should allow short-period planet signals (ie, non-monotransits) to be added to the model
        '''
        if not hasattr(self,'steffs'):
            self.PDFs()
        #Adding transiting planet classes to the star class using dfm's "transit"... Up to four possible.
        if not hasattr(self,'norm1') or replace:
            self.norm1 = Multtransit(tcen, tdur, depth, self.settings, name=self.objname+'.5', b=b)
            self.meanmodels+=[self.norm1]
        '''
        while a<=5 and len(self.meanmodels)==initlenmonos:
            if not hasattr(self,'mono'+str(a)) or replace:
                setattr(self,'mono'+str(a)) = Monotransit(tcen, tdur, depth, self.settings, self.LD1s, self.LD2s, self.denss,self.lc, name=self.objname+'.'+str(a), b=b)
                exec("self.mono"+str(a)+".Optimize_mono(Lcurve.flaten())")
                exec("self.mono"+str(a)+".calcmaxvel(self.Lcurve.lc,self.sdenss)")
                exec("self.meanmodels+=[self.mono"+str(a)+"]")
            a+=1
        '''

    def initLD(self):
        if not hasattr(self,'LD1s'):
            self.EstLimbDark()
        #Getting LD parameters for transit modelling:
        self.LDprior={'LD1':[0,1.0,'gaussian',np.median(self.LD1s),np.std(self.LD1s)],
                      'LD2':[0,1.0,'gaussian',np.median(self.LD2s),np.std(self.LD2s)]}

    def BuildMeanModel(self):
        if not hasattr(self,'LD1s'):
            self.EstLimbDark()
        #for model in self.meanmodels: #<<<TBD
        self.meanmodel_comb=MonotransitModel(tcen=self.mono1.tcen,
                             b=self.mono1.b,
                             vel=self.mono1.vel,
                             RpRs=self.mono1.RpRs,
                             LD1=np.median(self.LD1s),
                             LD2=np.median(self.LD2s))

    def BuildMeanPriors(self):
        #Building mean model priors
        if not hasattr(self, 'LDprior'):
            self.initLD()
        if not hasattr(self,'meanmodel_comb'):
            self.BuildMeanModel()
        self.meanmodel_priors=self.mono1.priors.copy()
        self.meanmodel_priors.update({'mean:'+ldp:self.LDprior[ldp] for ldp in self.LDprior})

    def BuildAllPriors(self,keylist=None):
        #Building priors from both GP and mean model and ordering by
        if not hasattr(self, 'meanmodel_priors'):
            self.BuildMeanPriors()

        self.priors=self.meanmodel_priors.copy()#{key:self.meanmodel_priors[key] for key in self.meanmodel_priors.keys()}
        if self.settings.GP:
            self.priors.update({'kernel:'+self.kern.get_parameter_names()[keyn]:[self.kern.get_parameter_bounds()[keyn][0],
                                                                                 self.kern.get_parameter_bounds()[keyn][1]]
                                for keyn in range(len(self.kern.get_parameter_names()))
                                })

            if 'kernel:terms[0]:log_amp' in self.kern.get_parameter_names():
                self.priors['kernel:terms[0]:log_amp']=self.priors['kernel:terms[0]:log_amp']+['evans',0.25*len(self.Lcurve.lc[:,0])]
            if 'kernel:terms[0]:log_a' in self.kern.get_parameter_names():
                self.priors['kernel:terms[0]:log_a']=self.priors['kernel:terms[0]:log_a']+['evans',0.25*len(self.Lcurve.lc[:,0])]

        if keylist is not None:
            #Sorting to match parameter vector:
            newprior={key:self.priors[key] for key in keylist}
            #print(str(len(keylist))+" keys in vector leading to "+str(len(newprior))+" new keys in priors, from "+str(len(self.priors))+" initially") if self.settings.verbose else 0
            self.priors=newprior
            print(self.priors) if self.settings.verbose else logger.info(self.objname+' - Priors set as '+' , '.join([key+' - '+str(self.priors[key]) for key in self.priors.keys()]))

    def RunMCMC(self):
        self.BuildMeanPriors()
        self.BuildMeanModel()
        if self.settings.GP:
            self.gp=celerite.GP(kernel=self.kern,mean=self.meanmodel_comb,fit_mean=True)
            self.BuildAllPriors(self.gp.get_parameter_names())
        else:
            self.BuildAllPriors(list(self.fitdict.keys()))
        #Returning monotransit model from information.
        chx=np.random.choice(self.settings.npdf,self.settings.nwalkers,replace=False)

        self.fitdict.update({'mean:'+nm:getattr(self.mono1,nm+'s')[chx] for nm in ['tcen','b','vel','RpRs']})
        self.fitdict.update({'mean:'+nm:getattr(self,nm+'s')[chx] for nm in ['LD1','LD2']})

        #Removing medians:
        for row in self.fitdict:
            self.fitdict[row][np.isnan(self.fitdict[row])]=np.nanmedian(np.isnan(self.fitdict[row]))
        if self.settings.GP:
            dists=[self.fitdict[cname] for cname in self.gp.get_parameter_names()]

        else:
            dists=[self.fitdict[cname] for cname in list(self.fitdict.keys())]


        mask=abs(self.Lcurve.lc[:,0]-self.meanmodel_comb.get_parameter_dict()['tcen']<
                 (self.settings.ndurswindow*CalcTdur(self.meanmodel_comb.get_parameter_dict()['vel'],
                                                     self.meanmodel_comb.get_parameter_dict()['b'],
                                                     self.meanmodel_comb.get_parameter_dict()['RpRs']
                                                     )
                 )
                )
        self.init_mcmc_params=np.column_stack(dists)

        # Cutting region around transit:

        #Plotting:
        if self.settings.GP:
            PlotModel(self.Lcurve.lc[mask,:], self.gp, np.median(self.init_mcmc_params,axis=0), fname=self.settings.outfilesloc+self.objname+'_initfit.png',GP=True)
        else:
            PlotModel(self.Lcurve.lc[mask,:], self.meanmodel_comb, np.median(self.init_mcmc_params,axis=0), fname=self.settings.outfilesloc+self.objname+'_initfit.png',GP=False)

        #dists=[np.random.normal(self.gp.get_parameter(nm),abs(self.gp.get_parameter(nm))**0.25,len(chx)) for nm in ['kernel:terms[0]:log_amp', 'kernel:terms[0]:log_timescale', 'kernel:terms[0]:log_period']]+\
        #      [self.tcens[chx],self.bs[chx],self.vels[~np.isnan(self.vels)][chx],self.RpRss[chx],self.LD1s[chx],self.LD2s[chx]]
        #'kernel:terms[0]:log_factor', 'kernel:terms[1]:log_sigma' <- frozen and not used
        #print(len(pos[0,:]))
        #[np.array(list(initparams.values())) *(1+ 1.5e-4*np.random.normal()) for i in range(nwalkers)]
        print("EMCEE HAPPENING. INIT DISTS:") if self.settings.verbose else logger.info(self.objname+" - EMCEE HAPPENING. INIT DISTS:")
        print(self.init_mcmc_params[0,:]) if self.settings.verbose else logger.info(self.objname+' , '.join(list(self.init_mcmc_params[0,:].astype(str))))
        if self.settings.GP:
            print(self.gp.get_parameter_names()) if self.settings.verbose else logger.info(self.objname+' , '.join(list(self.gp.get_parameter_names())))
            print(self.priors.keys()) if self.settings.verbose else logger.info(self.objname+' , '.join(list(self.priors.keys())))
            #print(' , '.join( [str(list(self.priors.keys())[nk]) [-8:]+' - '+str(abs(self.priors[list(self.priors.keys())[nk]]-self.init_mcmc_params[nk]))[:5] for nk in range(len(self.priors.keys()))]) )
            prilist=' \n '.join([str(list(self.priors.keys())[nk])+' - '+str(self.priors[list(self.priors.keys())[nk]][0])+" > "+str(np.median(self.init_mcmc_params[nk]))+\
                                    " < "+str(self.priors[list(self.priors.keys())[nk]][1]) for nk in range(len(self.priors.keys()))]\
                                  )
            print(prilist) if self.settings.verbose else logger.info(self.objname+prilist)

            sampler = emcee.EnsembleSampler(self.settings.nwalkers, len(dists), MonoLogProb, args=(self.Lcurve.lc[mask,:],self.priors,self.gp), threads=self.settings.nthreads)
        else:
            print(list(self.fitdict.keys())) if self.settings.verbose else logger.info(self.objname+' , '.join(list(self.fitdict.keys())))
            print(self.priors.keys()) if self.settings.verbose else logger.info(self.objname+' , '.join(list(self.priors.keys())))
            prilist=' \n '.join([str(list(self.priors.keys())[nk])+' - '+str(self.priors[list(self.priors.keys())[nk]][0])+" > "+str(np.median(self.init_mcmc_params[nk]))+\
                                    " < "+str(self.priors[list(self.priors.keys())[nk]][1]) for nk in range(len(self.priors.keys()))]\
                                  )
            sampler = emcee.EnsembleSampler(self.settings.nwalkers, len(dists), MonoOnlyLogProb, args=(self.Lcurve.lc[mask,:],self.priors,self.meanmodel_comb), threads=self.settings.nthreads)

        sampler.run_mcmc(self.init_mcmc_params, 1, rstate0=np.random.get_state())

        sampler.run_mcmc(self.init_mcmc_params, self.settings.nsteps, rstate0=np.random.get_state())

        #Trimming samples:
        ncut=np.min([int(self.settings.nsteps*0.25),3000])
        self.alllnprobs=sampler.lnprobability#[:,ncut:]#.reshape(-1)
        prcnt=np.percentile(self.alllnprobs[:,ncut:],[50,95],axis=1)
        #"Failed" walkers are where the 97th percentile is below the median of the rest
        good_wlkrs=(prcnt[1]>np.median(prcnt[0]))
        self.allsamples=sampler.chain
        if self.settings.GP:
            self.sampleheaders=list(self.gp.get_parameter_names())+['logprob']
        else:
            self.sampleheaders=list(self.fitdict.keys())+['logprob']
        self.samples = self.allsamples[good_wlkrs, ncut:, :].reshape((-1, len(dists)))
        self.samples = np.column_stack((self.samples,self.alllnprobs[good_wlkrs,ncut:].reshape(-1)))
        #Making impact parameter always positive:
        #self.samples[:,1]=abs(self.amples[:,1])

        #self.SaveMCMC()

    def SaveMCMC(self, samples=None, overwrite=False,suffix=''):
        if path.exists(self.settings.outfilesloc+self.objname+'_MCMCsamples'+suffix) and not overwrite:
            raise ValueError("Cannot save as file exists. Set, for example, overwrite=True, or suffix='_new'")
        if not samples:
            samples=self.samples
        np.save(self.settings.outfilesloc+self.objname+'_MCMCsamples'+suffix,samples)

    def SaveAll(self,overwrite=False,suffix=''):
        if path.exists(self.settings.outfilesloc+self.objname+'_obj'+suffix+'.pickle') and not overwrite:
            raise ValueError("Cannot save as file exists. Set, for example, overwrite=True, or suffix='_new'")
        #Saves object
        import pickle
        with open(self.settings.outfilesloc+self.objname+'_obj'+suffix+'.pickle','wb') as picklefile:
            pickle.dump(self, picklefile)

    def MonoFinalPars(self,model=None):
        if model is None and hasattr(self,'mono1'):
            model=self.mono1
        #Taking random Nsamples from samples to put through calculations
        #Need to form assymetric gaussians of Star Dat parameters if not equal
        #Rstardist2=np.hstack((np.sort(Rstardist[:, 0])[0:int(nsamp/2)], np.sort(Rstardist[:, 1])[int(nsamp/2):] ))
        modelmeanvals=[col.find('mean:')!=-1 for col in self.sampleheaders]
        model.gen_PDFs({self.sampleheaders[nmmv].split(":")[-1]+'s':self.samples[:,nmmv] for nmmv in range(len(self.sampleheaders)) if modelmeanvals[nmmv]})
        rn=np.random.choice(len(self.samples[:,0]),self.settings.npdf,replace=False)
        #for model in meanmodels:
        setattr(model,'Rps',(self.samples[rn,self.sampleheaders.index('mean:RpRs')]*695500000*self.srads)/6.371e6)#inearths
        setattr(model,'Prob_pl',len(model.Rps[model.Rps<(1.5*11.2)])/len(model.Rps))
        aest,Pest=VelToOrbit(self.samples[rn,self.sampleheaders.index('mean:vel')], self.srads, self.smasss, self.sdenss)#Rs, Ms, dens
        setattr(model,'smas',aest)
        setattr(model,'Ps',Pest)
        setattr(model,'Mps',PlanetRtoM(model.Rps))
        setattr(model,'Krvs',((2.*np.pi*6.67e-11)/(model.Ps*86400))**(1./3.)*(model.Mps*5.96e24/((1.96e30*self.smasss)**(2./3.))))
        #sigs=np.array([2.2750131948178987, , 97.7249868051821])
        sigs=[15.865525393145707, 50.0, 84.13447460685429]
        for val in ['Rps','smas','Ps','Mps','Krvs']:
            percnts=np.percentile(np.array(getattr(model,val)), sigs)
            setattr(model,val[:-1],percnts[1])
            setattr(model,val[:-1]+'uerr',(percnts[2]-percnts[1]))
            setattr(model,val[:-1]+'derr',(percnts[1]-percnts[0]))

    def PlotMCMC(self,suffix='',usecols=None):
        import corner
        print(self.objname,"Plotting MCMC corner") if self.settings.verbose else logger.info(self.objname+" - Plotting MCMC corner")
        if not self.settings.GP:
            newnames={'mean:tcen':r'$t_{cen}$',
                     'mean:b':r'$b$',
                     'mean:vel':r'$v/R_s$',
                     'mean:RpRs':r'$R_p/R_s$',
                     'mean:LD1':r'LD$_1$',
                     'mean:LD2':r'LD$_2$'}
            model2plot=self.meanmodel_comb
        elif self.settings.kernel=='quasi':
            newnames={'kernel:terms[0]:log_amp':r'$\log{a}$',
                     'kernel:terms[0]:log_timescale':r'$\log{\tau}$',
                     'kernel:terms[0]:log_period':r'$\log{P}$',
                     'mean:tcen':r'$t_{cen}$',
                     'mean:b':r'$b$',
                     'mean:vel':r'$v/R_s$',
                     'mean:RpRs':r'$R_p/R_s$',
                     'mean:LD1':r'LD$_1$',
                     'mean:LD2':r'LD$_2$'}
            model2plot=self.gp
        elif self.settings.kernel=='Real':
            newnames={'kernel:terms[0]:log_a':r'$\log{a}$',
                     'kernel:terms[0]:log_c':r'$\log{c}$',
                     'mean:tcen':r'$t_{cen}$',
                     'mean:b':r'$b$',
                     'mean:vel':r'$v/R_s$',
                     'mean:RpRs':r'$R_p/R_s$',
                     'mean:LD1':r'LD$_1$',
                     'mean:LD2':r'LD$_2$'}
            model2plot=self.gp
        if usecols is None and self.settings.GP:
            #Plotting corner with all parameter names
            usecols=self.gp.get_parameter_names()
        elif usecols is None and not self.settings.GP:
            usecols=list(self.fitdict.keys())

        plt.figure(1)

        Npars=len(self.samples[0])-1
        tobeplotted=np.tile(True,len(usecols))#np.in1d(self.gp.get_parameter_names(),usecols)
        #Clipping extreme values (top.bottom 0.1 percentiles)
        toclip=np.array([(np.percentile(self.samples[:,t],99.9)>self.samples[:,t])+(self.samples[:,t]>np.percentile(self.samples[:,t],0.1)) for t in np.arange(0,Npars,1)[tobeplotted]]).all(axis=0)
        clipsamples=self.samples[toclip,:-1]

        #Earmarking the difference between GP and non
        labs = [newnames[key] for key in usecols]
        #This plots the corner:
        fig = corner.corner(clipsamples[:,tobeplotted], labels=labs, quantiles=[0.16, 0.5, 0.84], plot_datapoints=False,range=np.tile(0.985,Npars))

        #Making sure the lightcurve plot doesnt overstep the corner
        ndim=np.sum(tobeplotted)
        rows=int((ndim)/2)+1
        cols=int((ndim)/2)-1

        #Printing Kepler name on plot
        plt.subplot(ndim,ndim,3).axis('off')
        plt.title(str(self.objname), fontsize=22)

        #This plots the full lc at the top:
        ax = plt.subplot2grid((ndim,ndim), (0, 2), rowspan=1, colspan=ndim-2)
        modelfits=PlotModel(self.Lcurve.lc, model2plot, np.nanmedian(clipsamples,axis=0),GP=self.settings.GP) #(lc, samples, scale=1.0, GP=GP)

        cliplc=self.Lcurve.lc[abs(self.Lcurve.lc[:,0]-self.mono1.tcen)<2.0,:]

        if self.settings.GP:
            #This plots the full lc at the top:
            ax_zoom=plt.subplot2grid((ndim,ndim), (1, ndim-cols), rowspan=1, colspan=cols)
            modelfits=PlotModel(cliplc, model2plot, np.nanmedian(clipsamples,axis=0),GP=self.settings.GP,title=False) #(lc, samples, scale=1.0, GP=GP)

            #If we do a Gaussian Process fit, plotting both the transit-subtractedGP model and the residuals
            ax=plt.subplot2grid((ndim,ndim), (2, ndim-cols), rowspan=1, colspan=cols)
            _=PlotModel(cliplc, model2plot, np.nanmedian(clipsamples,axis=0), prevmodels=modelfits, subGP=True,GP=self.settings.GP,title=False)

            #plotting residuals beneath:
            ax = plt.subplot2grid((ndim*2,ndim*2), (6, 2*(ndim-cols)), rowspan=1, colspan=2*cols)
            _=PlotModel(cliplc, model2plot, np.nanmedian(clipsamples,axis=0), prevmodels=modelfits, residuals=True,GP=self.settings.GP,title=False)
            #Adding text values to MCMC pdf
            #Plotting text wrt to residuals plot...
            xlims=ax_zoom.get_xlim()
            x0=xlims[0]-(0.5*(xlims[1]-xlims[0])) #Left of box in x
            xwid=0.5*(xlims[1]-xlims[0]) #Total width of box in x
            ylims=ax_zoom.get_ylim()
            y1=ylims[1]-0.08*(ylims[1]-ylims[0]) #Top of y box
            yheight=-5*(ylims[1]-ylims[0])/(rows-2) #Total height of box in y

        else:
            #This plots the full lc at the top:
            ax_zoom=plt.subplot2grid((ndim,ndim), (1, ndim-cols), rowspan=1, colspan=cols)
            modelfits=PlotModel(cliplc, model2plot, np.nanmedian(clipsamples,axis=0),GP=self.settings.GP,title=False) #(lc, samples, scale=1.0, GP=GP)

            #Not GP:
            ax=plt.subplot2grid((ndim*2,ndim*2), (4, 2*(ndim-cols)), rowspan=1, colspan=2*(cols))
            _=PlotModel(cliplc, model2plot, np.nanmedian(clipsamples,axis=0), prevmodels=modelfits, residuals=True,GP=self.settings.GP,title=False)

            #ax = plt.subplot2grid((ndim,ndim), (rows, ndim-cols-1), rowspan=1, colspan=cols+1)
            #Adding text values to MCMC pdf
            #Plotting text wrt to residuals plot...
            xlims=ax_zoom.get_xlim()
            xwid=0.5*(xlims[1]-xlims[0]) #Total width of box in x
            x0=xlims[0]+xwid#-(0.5*(xlims[1]-xlims[0])) #Left of box in x
            ylims=ax_zoom.get_ylim()
            yheight=-5*(ylims[1]-ylims[0])/(rows-2) #Total height of box in y
            y1=ylims[0]-0.8*(ylims[1]-ylims[0]) #Top of y box




        from matplotlib import rc;rc('text', usetex=True);plt.rc('text', usetex=True)

        #matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
        matplotlib.rc('font', family='serif', serif='cm10')
        matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
        matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']#Needed for latex commands here:
        #matplotlib.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}']

        ax_zoom.text(x0+0.025*xwid,y1,"EPIC"+str(self.objname),fontsize=20)

        n_textpos=0
        txt=["EPIC"+str(self.objname)]
        sigs=[2.2750131948178987, 15.865525393145707, 50.0, 84.13447460685429, 97.7249868051821]

        for lab in labs:
            xloc=x0+0.005*xwid
            yloc=y1+0.02*yheight+(0.05*yheight*(n_textpos+1))
            if lab==r'$t_{cen}$':#Need larger float size for Tcen...
                txt+=[r'\textbf{'+lab+":} "+('%s' % float('%.8g' % (np.median(self.samples[:,n_textpos]))))+" +"+('%s' % float('%.2g' % (np.percentile(self.samples[:,n_textpos],sigs[3])-np.median(self.samples[:,n_textpos]))))+" -"+\
                       ('%s' % float('%.2g' % (np.median(self.samples[:,n_textpos])-np.percentile(self.samples[:,n_textpos],sigs[1]))))]
                ax_zoom.text(xloc,yloc,txt[-1])
            else:
                txt+=[r'\textbf{'+lab+":} "+('%s' % float('%.3g' % (np.median(self.samples[:,n_textpos]))))+" +"+('%s' % float('%.2g' % (np.percentile(self.samples[:,n_textpos],sigs[3])-np.median(self.samples[:,n_textpos]))))+" -"+\
                       ('%s' % float('%.2g' % (np.median(self.samples[:,n_textpos])-np.percentile(self.samples[:,n_textpos],sigs[1]))))]
                ax_zoom.text(xloc,yloc,txt[-1])
            n_textpos+=1

        info={'Rps':r'$R_p (R_{\oplus})$','Ps':'Per (d)','smas':'A (au)','Mps':r'$M_p (M_{\oplus})$','Krvs':r'K$_{rv}$(ms$^{-1}$)','Prob_pl':r'ProbPl ($\%$)',\
              'steffs':'Teff (K)','srads':r'Rs ($R_{\odot}$)','smasss':r'Ms ($M_{\odot}$)','sloggs':'logg','sdenss':r'$\rho_s (\rho_{\odot})$'}

        pdfs=self.pdflist()
        for ival in pdfs:
            if ival[:2]!='LD':
                xloc=x0+0.005*xwid
                yloc=y1+0.02*yheight+(0.05*yheight*(n_textpos+1.2))
                vals=np.percentile(pdfs[ival],sigs)
                txt+=[(r"\textbf{%s:} " % info[ival])+('%s' % float('%.3g' % vals[2]))+" +"+\
                      ('%s' % float('%.2g' % (vals[3]-vals[2])))+" -"+('%s' % float('%.2g' % (vals[2]-vals[1])))]
                ax_zoom.text(xloc,yloc,txt[-1])
                n_textpos+=1


        self.MonoFinalPars()
        for ival in ['Rps','smas','Ps','Mps','Krvs']:
            xloc=x0+0.05*xwid
            yloc=y1+0.02*yheight+(0.05*yheight*(n_textpos+1.2))
            vals=np.percentile(getattr(self.mono1,ival),sigs)
            txt+=[(r"\textbf{%s:} " % info[ival])+('%s' % float('%.3g' % vals[2]))+" +"+\
                  ('%s' % float('%.2g' % (vals[3]-vals[2])))+" -"+('%s' % float('%.2g' % (vals[2]-vals[1])))]
            ax_zoom.text(xloc,yloc,txt[-1])
            n_textpos+=1

        xloc=x0+0.05*xwid
        yloc=y1+0.02*yheight+(0.05*yheight*(n_textpos+1.2))
        txt+=[(r"\textbf{%s:} " % info['Prob_pl'])+('%s' % float('%.3g' % getattr(self.mono1,'Prob_pl')))]
        ax_zoom.text(xloc,yloc,txt[-1])
        with open(self.settings.outfilesloc+'latextable.tex','a') as latextable:
            latextable.write(' & '.join(txt)+'\n')


        #Saving as pdf. Will save up to 3 unique files.
        fname='';n=0
        while fname=='':
            if os.path.exists(self.settings.outfilesloc+'Corner_'+str(self.objname)+'_'+str(int(n))+'_'+suffix+'.pdf'):
                n+=1
            else:
                fname=self.settings.outfilesloc+'/Corner_'+self.objname+'_'+str(int(n))+'_'+suffix+'.pdf'

        plt.savefig(fname.replace('.pdf','.pdf'),Transparent=True,dpi=300)
        plt.savefig(fname.replace('pdf','png'),Transparent=True,dpi=300)


class Monotransit():
    #Monotransit detection to analyse
    def __init__(self, tcen, tdur, depth, settings, name, b=0.4, RpRs=None, vel=None,\
                 tcenuerr=None,tduruerr=None,depthuerr=None,buerr=None,RpRsuerr=None, veluerr=None,\
                 tcenderr=None,tdurderr=None,depthderr=None,bderr=None,RpRsderr=None,velderr=None):
        self.settings=settings
        self.mononame = str(name)
        self.starname = name.split('.')[0]
        self.pdfs = {}

        self.update_pars(tcen, tdur, depth, b=b, RpRs=RpRs, vel=vel,
                         tcenuerr=tcenuerr,tduruerr=tduruerr,depthuerr=depthuerr,buerr=buerr, RpRsuerr=RpRsuerr,veluerr=veluerr,
                         tcenderr=tcenderr,tdurderr=tdurderr,depthderr=depthderr,bderr=bderr,RpRsderr=RpRsderr, velderr=velderr)
        self.gen_PDFs()


    '''
   def addStarDat(self,pdflist):
        self.LD1s=pdflist['LD1s']
        self.LD2s=pdflist['LD2s']
        self.srads=pdflist['srads']
        self.sdenss=pdflist['sdenss']
        self.smasss=pdflist['smasss']
        self.steffs=pdflist['steffs']
        self.pdfs.update({'LD1s':self.LD1s,'LD2s':self.LD2s,'srads':self.srads,'sdenss':self.sdenss,'smasss':self.smasss,'steffs':self.steffs})
    '''

    def update_pars(self, tcen=None, tdur=None, depth=None, b=0.4, RpRs=None, vel=None, tcenuerr=None,tduruerr=None,depthuerr=None,buerr=None,RpRsuerr=None, tcenderr=None,tdurderr=None,depthderr=None,bderr=None,RpRsderr=None,veluerr=None,velderr=None):
        if tcen is not None:
            self.tcen = float(tcen)                                                          # detected transit centre
            self.tcenuerr = 0.05*tdur if type(tcenuerr)==type(None) else tcenuerr             # estimated transit centre errors (default = 0.1*dur)
            self.tcenderr = 0.05*tdur if type(tcenderr)==type(None) else tcenderr             # estimated transit centre errors (default = 0.1*dur)
        if tdur is not None:
            self.tdur = float(tdur)                                                          # detected transit duration
            self.tduruerr = 0.33*tdur if type(tcenuerr)==type(None) else tcenuerr             # estimated transit duration errors (default = 0.2*dur)
            self.tdurderr = 0.33*tdur if type(tcenuerr)==type(None) else tcenuerr             # estimated transit duration errors (default = 0.2*dur)
        if depth is not None:
            self.depth = float(depth)                                                        # detected transit depth
            self.depthuerr = 0.33*depth if type(tcenuerr)==type(None) else tcenuerr           # estimated transit depth errors (default = 0.1*depth)
            self.depthderr = 0.33*depth if type(tcenuerr)==type(None) else tcenuerr           # estimated transit depth errors (default = 0.1*depth)

        self.b = 0.5 if type(b)==type(None) else b                                       # estimated impact parameter (default = 0.5)
        self.buerr = 0.5 if type(buerr)==type(None) else buerr                           # estimated impact parameter errors (default = 0.5)
        self.bderr = 0.5 if type(bderr)==type(None) else bderr                           # estimated impact parameter errors (default = 0.5)

        self.RpRs = self.depth**0.5 if type(RpRs)==type(None) else RpRs                  # Ratio of planet to star radius
        self.RpRsuerr = 0.5*self.RpRs if type(RpRsuerr)==type(None) else RpRsuerr       # Ratio of planet to star radius errors (default = 25%)
        self.RpRsderr = 0.5*self.RpRs if type(RpRsderr)==type(None) else RpRsderr       # Ratio of planet to star radius errors (default = 25%)
        #self.vel = CalcVel()        # Velocity of planet relative to stellar radius

        if vel is not None:
            self.vel = vel                                                              # Velocity scaled to stellar radius
        elif not hasattr(self,'vel'):
            self.vel = None                                                             # Velocity scaled to stellar radius

    def gen_PDFs(self,paramdict=None):
        #Turns params into PDFs
        if paramdict is None:
            self.tcens=GetAssymDist(self.tcen,self.tcenuerr,self.tcenderr,nd=self.settings.npdf,returndist=True)
            #self.depths=GetAssymDist(self.depth,self.depthuerr,self.depthderr,nd=self.settings.npdf,returndist=True)
            self.bs=abs(GetAssymDist(self.b,self.buerr,self.bderr,nd=self.settings.npdf,returndist=True))
            self.RpRss=GetAssymDist(self.RpRs,self.RpRsuerr,self.RpRsderr,nd=self.settings.npdf,returndist=True)

            #Velocity tends to get "Nan"-y, so looping to avoid that:
            nanvels=np.tile(True,self.settings.npdf)
            v=np.zeros(self.settings.npdf)
            while np.sum(nanvels)>self.settings.npdf*0.002:
                v[nanvels]=CalcVel(np.random.normal(self.tdur,self.tdur*0.15,nanvels.sum()), self.bs[np.random.choice(self.settings.npdf,np.sum(nanvels))], self.RpRss[np.random.choice(self.settings.npdf,np.sum(nanvels))])
                nanvels=(np.isnan(v))*(v<0.0)*(v>100.0)
            self.vels=v
            prcnts=np.diff(np.percentile(self.vels[~np.isnan(self.vels)],[15.865525393145707, 50.0, 84.13447460685429]))
            self.veluerr=prcnts[1]
            self.velderr=prcnts[0]
            if self.vel is not None and ~np.isnan(self.vel):
                #Velocity pre-defined. Distribution is not, however, so we'll use the scaled distribution of the "derived" velocity dist to give the vel errors
                velrat=self.vel/np.nanmedian(self.vels)
                self.vels*=velrat
                self.veluerr*=velrat
                self.velderr*=velrat
            else:
                self.vel=np.nanmedian(self.vels)
        else:
            #included dictionary of new "samples"
            sigs=[15.865525393145707, 50.0, 84.13447460685429]
            for colname in ['tcens','bs','vels','RpRss']:
                setattr(self,colname,paramdict[colname])
                percnts=np.percentile(np.array(getattr(self,colname)), sigs)
                setattr(self,colname[:-1],percnts[1])
                setattr(self,colname[:-1]+'uerr',percnts[2]-percnts[1])
                setattr(self,colname[:-1]+'derr',percnts[1]-percnts[0])


        self.pdfs.update({'tcens':self.tcens,'bs':self.bs,'RpRss':self.RpRss,'vels':self.vels})
        #if StarPDFs is not None:
        #    self.pdflist.update(StarPDFs)
    def loadmodel(self,LDprior):
        self.model=MonotransitModel(tcen=self.tcen,
                             b=self.b,
                             vel=self.vel,
                             RpRs=self.RpRs,
                             LD1=LDprior['LD1'][3],
                             LD2=LDprior['LD2'][3])

    def calcmaxvel(self,lc,sdenss):
        #Estimate maximum velocity given lightcurve duration without transit.
        self.calcminp(lc)
        maxvels=np.array([((18226*rho)/self.minp)**(1/3.0) for rho in abs(sdenss)])
        prcnts=np.percentile(maxvels,[15.865525393145707, 50.0, 84.13447460685429])
        self.maxvelderr=prcnts[1]-prcnts[0]
        self.maxvel=prcnts[1]
        self.maxveluerr=prcnts[2]-prcnts[1]

    def calcminp(self,lc):
        #finding tdur-wide jumps in folded LC
        dur_jumps=np.where(np.diff(abs(lc[:,0]-self.tcen))>self.tdur)[0]
        if len(dur_jumps)==0:
            #No tdur-wide jumps until end of lc - using the maximum difference to a point in the lc
            self.minp=np.max(abs(lc[:,0]-self.tcen))+self.tdur*0.33
        else:
            #Taking the first Tdur-wide jump in the folded lightcurve where a transit could be hiding.
            self.minp=abs(lc[:,0]-self.tcen)[dur_jumps[0]]+self.tdur*0.33
    '''
    def CalcOrbit(self,denss):
        #Calculating orbital information
        #VelToOrbit(Vel, Rs, Ms, ecc=0, omega=0):
        SMA,P=Vel2Per(denss,self.vels)
        self.SMAs=SMA
        self.PS=P

    def update(self, **kwargs):

        #Modify detection parameters...

        self.tcen  = kwargs.pop('tcen', self.tcen)          # detected transit centre
        self.tdur  = kwargs.pop('tdur', self.tdur)          # detected transit duration
        self.depth = kwargs.pop('dep', self.depth)          # detected transit depth
        self.b     = kwargs.pop('b', self.b)                # estimated impact parameter (default = 0.4)
        self.RpRs  = kwargs.pop('RpRs', self.RpRs)          # Ratio of planet to star radius
        self.vel   = kwargs.pop('vel', self.vel)            # Velocity of planet relative to stellar radius

    def FitParams(self,star,info):
        #Returns params array needed for fitting
        if info.GP:
            return np.array([self.tcen,self.b,self.vel,self.RpRs])
        else:
            return np.array([self.tcen,self.b,self.vel,self.RpRs])

    def InitialiseGP(self,settings,star,Lcurve):
        import george
        self.gp, res, self.lnlikfit = TrainGP(Lcurve.lc,self.tcen,star.wn)
        self.newmean,self.newwn,self.a,self.tau=res

    def FitPriors(self,star,settings):
        #Returns priros array needed for fitting
        if settings.GP:
            if not self.hasattr('tau'):
                self.InitialiseGP()
            return np.array([[self.tcen,self.tcen,self.tcen,self.tcen],
                             [-1.2,1.2,0,0],
                             [0.0,100.0,self.vmax,self.vmaxerr],
                             [0.0,0.3,0,0],
                             [0.0,1.0,star.LD1,np.average(star.LD1uerr,star.LD1derr)],
                             [0.0,1.0,star.LD2,np.average(star.LD2uerr,star.LD2derr)],
                             [np.log(star.wn)-1.5,np.log(star.wn)+1.5,np.log(star.wn),0.3],
                             [self.tau-10,self.tau+10,self.tau,np.sqrt(np.abs(self.tau))],
                             [self.a-10,self.a+10,self.a,np.sqrt(np.abs(self.a))]])
        else:
            return np.array([[self.tcen,self.tcen,self.tcen,self.tcen],
                             [-1.2,1.2,0,0],
                             [0.0,100.0,self.vmax,self.vmaxerr],
                             [0.0,0.3,0,0],
                             [0.0,1.0,star.LD1,np.average(star.LD1uerr,star.LD1derr)],
                             [0.0,1.0,star.LD2,np.average(star.LD2uerr,star.LD2derr)]]
    '''

    def Optimize_mono(self,flatlc,LDprior,nopt=20):
        if not hasattr(self,'priors'):
            self.monoPriors()
        #Cutting to short area around transit:
        flatlc=flatlc[abs(flatlc[:,0]-self.tcen)<5*self.tdur]
        #Optimizing initial transit parameters
        opt_monomodel=MonotransitModel(tcen=self.tcen,
                             b=self.b,
                             vel=self.vel,
                             RpRs=self.RpRs,
                             LD1=LDprior['LD1'][3],
                             LD2=LDprior['LD2'][3])
        temp_priors=self.priors.copy()
        temp_priors['mean:LD1'] = LDprior['LD1']
        temp_priors['mean:LD2'] = LDprior['LD2']
        print("monopriors:",temp_priors) if self.settings.verbose else logger.info(self.mononame+" - monopriors: "+' , '.join([key+' - '+str(temp_priors[key]) for key in temp_priors.keys()]))
        init_neglogprob=MonoOnlyNegLogProb(opt_monomodel.get_parameter_vector(),flatlc,temp_priors,opt_monomodel)
        print("nll init",init_neglogprob) if self.settings.verbose else logger.info(self.mononame+" - nll init: "+str(init_neglogprob))
        suc_res=np.zeros(7)

        LD1s=np.random.normal(LDprior['LD1'][3],LDprior['LD1'][4],self.settings.npdf)
        LD2s=np.random.normal(LDprior['LD2'][3],LDprior['LD2'][4],self.settings.npdf)

        #Running multiple optimizations using rough grid of important model paramsself.
        for n_par in np.random.choice(self.settings.npdf,nopt,replace=False):

            initpars=np.array([self.tcens[n_par],self.bs[n_par],self.vels[n_par],self.RpRss[n_par],LD1s[n_par],LD2s[n_par]])
            result = opt.minimize(MonoOnlyNegLogProb, initpars, args=(flatlc, temp_priors, opt_monomodel), method="L-BFGS-B")
            if result.success:
                suc_res=np.vstack((suc_res,np.hstack((result.x,result.fun))))

        if len(np.shape(suc_res))==1:
            raise ValueError("No successful Monotransit minimizations")
        else:
            #Ordering successful optimizations by neglogprob...
            suc_res=suc_res[1:,:]
            #print("All_Results:",suc_res) if self.settings.verbose else logger.info(self.objname,"mono opt, All_Results:",suc_res)
            suc_res=suc_res[~np.isnan(suc_res[:,-1]),:]
            bestres=suc_res[np.argmin(suc_res[:,-1])]
            self.bestres=bestres
            print("Best_Result:",bestres) if self.settings.verbose else logger.info(self.mononame+" - mono opt, Best Results:"+' , '.join(list(bestres.astype(str))))
            #tcen, tdur, depth, b=0.4, RpRs=None, vel=None
            self.update_pars(bestres[0], CalcTdur(bestres[2], bestres[1], bestres[3]), bestres[3]**2, b=bestres[1], RpRs=bestres[3],vel=bestres[2])
            print("initial fit nll: ",init_neglogprob," to new fit nll: ",bestres[-1]) if self.settings.verbose else logger.info(self.mononame+"initial fit nll: "+str(init_neglogprob)+" to new fit nll: "+' , '.join(list(bestres[-1].astype(str))))
            #for nn,name in enumerate(['mean:tcen', 'mean:b', 'mean:vel', 'mean:RpRs', 'mean:LD1', 'mean:LD2']):
            #        self.gp.set_parameter(name,bestres[nn])

        PlotBestMono(flatlc, opt_monomodel, bestres[:-1], fname=self.settings.outfilesloc+self.mononame+'_init_monoonly_fit.png')


    def monoPriors(self,name='mean'):
        self.priors={}
        self.priors.update({name+':tcen':[self.tcen-self.tdur*0.3,self.tcen+self.tdur*0.3],
                    name+':b':[0.0,1.25],
                    name+':vel':[0,self.maxvel+5*self.maxveluerr,'normlim',self.maxvel,self.maxveluerr],
                    name+':RpRs':[0.02,0.25]
                    })
        return self.priors
    '''
    def RunModel(self,lc,gp):
        self.modelPriors()
        #Returning monotransit model from information.
        sampler = emcee.EnsembleSampler(self.settings.nwalkers, len(gp.get_parameter_vector()), MonoLogProb, args=(lc,self.priors,gp), threads=self.settings.nthreads)
        chx=np.random.choice(np.sum(~np.isnan(self.vels)),self.settings.nwalkers,replace=False)
        dists=[np.random.normal(gp.get_parameter(nm),abs(gp.get_parameter(nm))**0.25,len(chx)) for nm in ['kernel:terms[0]:log_amp', 'kernel:terms[0]:log_timescale', 'kernel:terms[0]:log_period']]+\
              [self.tcens[chx],self.bs[chx],self.vels[~np.isnan(self.vels)][chx],self.RpRss[chx],self.LD1s[chx],self.LD2s[chx]]
               #'kernel:terms[0]:log_factor', 'kernel:terms[1]:log_sigma' <- frozen and not used
        col=['kernel:terms[0]:log_amp', 'kernel:terms[0]:log_timescale', 'kernel:terms[0]:log_period',\
             'mean:tcen','mean:b','mean:vel','mean:RpRs','mean:LD1','mean:LD2']
        pos=np.column_stack(dists)
        self.init_mcmc_params=pos
        #print(len(pos[0,:]))
        #[np.array(list(initparams.values())) *(1+ 1.5e-4*np.random.normal()) for i in range(nwalkers)]

        Nsteps = 30000
        sampler.run_mcmc(pos, 1, rstate0=np.random.get_state())

        sampler.run_mcmc(pos, self.settings.nsteps, rstate0=np.random.get_state())
        self.samples = sampler.chain[:, 3000:, :].reshape((-1, ndim))
        return self.samples
        #.light_curve(np.arange(0,40,0.024),texp=0.024))
    '''
    def SaveInput(self,stellarpdfs):
        np.save(self.settings.fitsloc+self.mononame+'_inputsamples',np.column_stack(([self.pdfs[ipdf] for ipdf in self.pdfs.keys()]+[stellarpdfs[ipdf] for ipdf in stellarpdfs.keys()])))


class Lightcurve():
    # Lightcurve class - contains all lightcurve information
    def __init__(self, file, epic, settings):
        self.settings=settings
        self.fileloc=file
        self.lc,self.mag=OpenLC(self.fileloc)
        try:
            self.mag=k2_quickdat(epic)['k2_kepmag']
        except:
            self.mag=self.mag
        self.lc=self.lc[~np.isnan(np.sum(self.lc,axis=1))]
        self.fluxmed=np.nanmedian(self.lc[:,1])
        self.lc[:,1:]/=self.fluxmed
        self.lc=self.lc[AnomCutDiff(self.lc[:,1],self.settings.anomcut)]
        self.range=self.lc[-1,0]-self.lc[self.lc[:,0]!=0.0,0][0]
        self.cadence=np.nanmedian(np.diff(self.lc[:,0]))

        self.lcmask=np.tile(True,len(self.lc[:,0]))

    def BinLC(self, binsize,gap=0.4):
        #Bins lightcurve to some time interval. Finds gaps in the lightcurve using the threshold "gap"
        spl_ts=np.array_split(self.lc[:,0],np.where(np.diff(self.lc[:,0])>gap)[0]+1)
        bins=np.hstack([np.arange(s[0],s[-1],binsize) for s in spl_ts])
        digitized = np.digitize(self.lc[:,0], bins)
        ws=(self.lc[:,2])**-2.0
        ws=np.where(ws==0.0,np.median(ws[ws!=0.0]),ws)
        bin_means = np.array([np.ma.average(self.lc[digitized==i,2],weights=ws[digitized==i]) for i in range(np.max(digitized))])
        bin_stds = np.array([np.ma.average((self.lc[digitized==i,2]-bin_means[i])**2, weights=ws[digitized==i]) for i in range(np.max(digitized))])
        whok=(~np.isnan(bin_means))&(bin_means!=0.0)
        self.binlc=np.column_stack((bins,bin_means,bin_stds))[whok,:]
        self.binsize=binsize
        return self.binlc
    '''
    def keys(self):
        return ['NPTS','SKY_TILE','RA_OBJ','DEC_OBJ','BMAG','VMAG','JMAG','KMAG','HMAG','PMRA','PMDEC','PMRAERR','PMDECERR','NFIELDS']
    def keyvals(self,*args):
        #Returns values for the given key list
        arr=[]
        for ke in args[0]:
            exec('arr+=[self.%s]' % ke)
        return arr
    '''
    def savelc(self):
        np.save(self.settings.fitsloc+self.OBJNAME.replace(' ','')+'_bin.npy',self.get_binlc())
        np.save(self.settings.fitsloc+self.OBJNAME.replace(' ','')+'.npy',self.get_lc())
    def flatten(self,winsize=4.5,stepsize=0.125):
        from . import k2flatten
        return k2flatten.ReduceNoise(self.lc,winsize=winsize,stepsize=stepsize)
    def calc_mask(self,meanmodels):
        for model in meanmodels:
            if hasattr(model,'P'):
                self.lcmask[((abs(self.lc[:,0]-model.tcen)%model.P)<(self.cadence+model.tdur*0.5))+((abs(self.lc[:,0]-model.tcen)%model.P)>(model.P-(model.tdur*0.5+self.cadence)))]=False
            else:
                #Mono
                self.lcmask[abs(self.lc[:,0]-model.tcen)<(self.cadence+model.tdur*0.5)]=False

class MonotransitModel(celerite.modeling.Model):
    parameter_names = ("tcen", "b","vel","RpRs","LD1","LD2")

    def get_value(self,t):
        #Getting fine cadence (cad/100). Integrating later:
        cad=np.median(np.diff(t))
        oversamp=10#Oversampling
        finetime=np.hstack(([t+overts for overts in cad*np.arange(0,1.0,1/oversamp)]))
        finetime=np.sort(finetime)
        z = np.sqrt(self.b**2+(self.vel*(finetime - self.tcen))**2)
        #Removed the flux component below. Can be done by GP
        model = occultquad(z, self.RpRs, np.array((self.LD1,self.LD2)))
        return np.average(np.resize(model, (len(t), oversamp)), axis=1)
'''
#TBD:
class MonotransitModel_x2(celerite.modeling.Model):
    nmodels=2
    parameter_names = tuple([item for sublist in [["tcen"+str(n), "b"+str(n),"vel"+str(n),"RpRs"+str(n)] for n in range(1,nmodels+1)]+[["LD1","LD2"]] for item in sublist ])
    def get_value(self,t):
        cad=np.median(np.diff(t))
        oversamp=10#Oversampling
        finetime=np.hstack(([t+overts for overts in cad*np.arange(0,1.0,1/oversamp)]))
        finetime=np.sort(finetime)
        model=np.zeros((len(finetime)))
        for nmod in range(nmodels):
            z = np.sqrt(getattr(self,'b'+str(nmod))**2+(getattr(self,'vel'+str(nmod))*(finetime - getattr(self,'tcen'+str(nmod))))**2)
            #Removed the flux component below. Can be done by GP
            model *= occultquad(z, getattr(self,'RpRs'+str(nmod)), np.array((self.LD1,self.LD2)))
        return np.average(np.resize(model, (len(t), oversamp)), axis=1)
class MonotransitModel_x3(celerite.modeling.Model):
    nmodels  =3
    parameter_names = tuple([item for sublist in [["tcen"+str(n), "b"+str(n),"vel"+str(n),"RpRs"+str(n)] for n in range(1,nmodels+1)]+[["LD1","LD2"]] for item in sublist ])
    def get_value(self,t):
        cad=np.median(np.diff(t))
        oversamp=10#Oversampling
        finetime=np.hstack(([t+overts for overts in cad*np.arange(0,1.0,1/oversamp)]))
        finetime=np.sort(finetime)
        model=np.zeros((len(finetime)))
        for nmod in range(nmodels):
            z = np.sqrt(getattr(self,'b'+str(nmod))**2+(getattr(self,'vel'+str(nmod))*(finetime - getattr(self,'tcen'+str(nmod))))**2)
            #Removed the flux component below. Can be done by GP
            model *= occultquad(z, getattr(self,'RpRs'+str(nmod)), np.array((self.LD1,self.LD2)))
        return np.average(np.resize(model, (len(t), oversamp)), axis=1)

import batman
class MonotransitModel_plus_pl(celerite.modeling.Model):
    parameter_names = ("monotcen", "monob","monovel","monoRpRs","multitcen", "multib","multiP","multiRpRs","multia_Rs","LD1","LD2")

    def get_value(self,t):
        oversamp=10#Oversampling

        params = batman.TransitParams()
        params.t0 = self.multitcen                       #time of inferior conjunction
        params.per = self.multiP                         #orbital period
        params.rp = self.multiRpRs                       #planet radius (in units of stellar radii)
        params.a = self.multia_Rs                        #semi-major axis (in units of stellar radii)
        params.inc = np.acos(self.multib/self.multia_Rs)      #orbital inclination (in degrees)
        params.u = [self.LD1, self.LD2]                       #limb darkening coefficients
        params.ecc = 0.;params.w = 90.;params.limb_dark = "quadratic"
        multilc=batman.TransitModel(params,t,supersample_factor=oversamp).light_curve(params)

        cad=np.median(np.diff(t))
        finetime=np.hstack(([t+overts for overts in cad*np.arange(0,1.0,1/oversamp)]))
        finetime=np.sort(finetime)
        model=np.zeros((len(finetime)))
        for nmod in range(nmodels):
            z = np.sqrt(getattr(self,'b'+str(nmod))**2+(getattr(self,'vel'+str(nmod))*(finetime - getattr(self,'tcen'+str(nmod))))**2)
            #Removed the flux component below. Can be done by GP
            model *= occultquad(z, getattr(self,'RpRs'+str(nmod)), np.array((self.LD1,self.LD2)))

        return np.average(np.resize(model, (len(t), oversamp)), axis=1)-(1.0-multilc)

class MonotransitModel_plus_plx2(celerite.modeling.Model):
    parameter_names = tuple(["monotcen", "monob","monovel","monoRpRs"]+
                            ['multi'+item for sublist in [["tcen"+str(n), "b"+str(n),"P"+str(n),"RpRs"+str(n),"a_Rs"+str(n)] for n in range(1,3)]+
                            [["LD1","LD2"]] for item in sublist ])
    def get_value(self,t):
        cad=np.median(np.diff(t))
        oversamp=10#Oversampling

        p1 = batman.TransitParams()
        p1.t0 = self.multitcen1                       #time of inferior conjunction
        p1.per = self.multiP1                         #orbital period
        p1.rp = self.multiRpRs1                       #planet radius (in units of stellar radii)
        p1.a = self.multia_Rs1                        #semi-major axis (in units of stellar radii)
        p1.inc = np.acos(self.multib1/self.multia_Rs1)      #orbital inclination (in degrees)
        p1.u = [self.LD1, self.LD2]                       #limb darkening coefficients
        p1.ecc = 0.;p1.w = 90.;p1.limb_dark = "quadratic"

        p2 = batman.TransitParams()
        p2.t0 = self.multitcen2                       #time of inferior conjunction
        p2.per = self.multiP2                         #orbital period
        p2.rp = self.multiRpRs2                       #planet radius (in units of stellar radii)
        p2.a = self.multia_Rs2                        #semi-major axis (in units of stellar radii)
        p2.inc = np.acos(self.multib2/self.multia_Rs2)      #orbital inclination (in degrees)
        p2.u = [self.LD1, self.LD2]                       #limb darkening coefficients
        p2.ecc = 0.;p2.w = 90.;p2.limb_dark = "quadratic"

        multilc=batman.TransitModel(p1,t,supersample_factor=oversamp).light_curve(p1)-(1.0-batman.TransitModel(p1,t,supersample_factor=oversamp).light_curve(p1))

        finetime=np.hstack(([t+overts for overts in cad*np.arange(0,1.0,1/oversamp)]))
        finetime=np.sort(finetime)
        model=np.zeros((len(finetime)))
        for nmod in range(1,3):
            z = np.sqrt(getattr(self,'b'+str(nmod))**2+(getattr(self,'vel'+str(nmod))*(finetime - getattr(self,'tcen'+str(nmod))))**2)
            #Removed the flux component below. Can be done by GP
            model *= occultquad(z, getattr(self,'RpRs'+str(nmod)), np.array((self.LD1,self.LD2)))
        return np.average(np.resize(model, (len(t), oversamp)), axis=1)-(1.0-multilc)

class MonotransitModelx3_plus_plx2(celerite.modeling.Model):
    parameter_names = tuple(['mono'+item for sublist in [["tcen"+str(n), "b"+str(n),"vel"+str(n),"RpRs"+str(n)] for n in range(1,4)]]+\
                            ['multi'+item for sublist in [["tcen"+str(n), "b"+str(n),"P"+str(n),"RpRs"+str(n),"a_Rs"+str(n)] for n in range(4,6)]]+\
                            ["LD1","LD2"]])
    def get_value(self,t):
        cad=np.median(np.diff(t))
        oversamp=10#Oversampling

        multilc=np.zeros_like(t)
        for nmodel in range(1,2):
            params = batman.TransitParams()
            params.t0 = getattr(self,'multitcen'+str(nmodel))       #time of inferior conjunction
            params.per = getattr(self,'multiP'+str(nmodel))                         #orbital period
            params.rp = getattr(self,'multiRpRs'+str(nmodel))                       #planet radius (in units of stellar radii)
            params.a = getattr(self,'multia_Rs+str(nmodel))                        #semi-major axis (in units of stellar radii)
            params.inc = np.acos(getattr(self,'multib'+str(nmodel))/params.a)      #orbital inclination (in degrees)
            params.u = [self.LD1, self.LD2]                       #limb darkening coefficients
            params.ecc = 0.;params.w = 90.;params.limb_dark = "quadratic"
            multilc+=(1-batman.TransitModel(params,t,supersample_factor=oversamp).light_curve(params))

        finetime=np.hstack(([t+overts for overts in cad*np.arange(0,1.0,1/oversamp)]))
        finetime=np.sort(finetime)
        model=np.zeros((len(finetime)))
        for nmod in range(1,4):
            z = np.sqrt(getattr(self,'b'+str(nmod))**2+(getattr(self,'vel'+str(nmod))*(finetime - getattr(self,'tcen'+str(nmod))))**2)
            #Removed the flux component below. Can be done by GP
            model *= occultquad(z, getattr(self,'RpRs'+str(nmod)), np.array((self.LD1,self.LD2)))
        return np.average(np.resize(model, (len(t), oversamp)), axis=1)+multilc
'''
def k2_quickdat(kic):
    kicdat=pd.DataFrame.from_csv("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=k2targets&where=epic_number=%27"+str(int(kic))+"%27")
    if len(kicdat.shape)>1:
        kicdat=kicdat.iloc[0]
    return kicdat

'''
def MonoLnPriorDict(params,priors):
    lp=0
    for key in priors.keys():
        #print(params[n],key,priors[key])
        if params[key]<priors[key][0] or params[key]>priors[key][1]:
            #hidesously low number that still has gradient towards "mean" of the uniform priors.
            lp-=1e15 * (params[key]-(0.5*(priors[key][0]+priors[key][1])))**2
            #print(key," over prior limit")
        if len(priors[key])>2:
            if priors[key][2]=='gaussian':
                lp+=stats.norm(priors[key][3],priors[key][4]).pdf(params[key])
            elif priors[key][2]=='normlim':
                #Special velocity prior from min period & density:
                lp+=((1.0-stats.norm.cdf(params[key],priors[key][3],priors[key][4]))*params[key])**2
    return lp
'''

def MonoLnPrior(params,priors):
    lp=0
    for n,key in enumerate(priors.keys()):
        #print(params[n],key,priors[key])
        if params[n]<priors[key][0] or params[n]>priors[key][1]:
            #hideously low number that still has gradient towards "mean" of the uniform priors. Scaled by prior width
            lp-=1e20 * (((params[n]-0.5*(priors[key][0]+priors[key][1])))/(priors[key][1]-priors[key][0]))**2
            #print(key," over prior limit")
        if len(priors[key])>2:
            if priors[key][2]=='gaussian':
                lp+=stats.norm(priors[key][3],priors[key][4]).pdf(params[n])
            elif priors[key][2]=='normlim':
                #Special velocity prior from min period & density:
                lp+=((1.0-stats.norm.cdf(params[n],priors[key][3],priors[key][4]))*params[n])**2
            elif priors[key][2]=='evans':
                #Evans 2015 limit on amplitude - same as p(Ai) = Gam(1, 100) (apparently, although I cant see it.)
                #Basically prior is ln
                lp-=priors[key][3]*np.exp(params[n])#
    return lp

def MonoLogProb(params,lc,priors,gp):
    gp.set_parameter_vector(params)
    gp.compute(lc[:,0],lc[:,2])
    ll = gp.log_likelihood(lc[:,1])
    lp = MonoLnPrior(params,priors)
    return (ll+lp)


def MonoNegLogProb(params,lc,priors,gp):
    return -1*MonoLogProb(params,lc,priors,gp)

def MonoOnlyLogProb(params,lc,priors,monomodel):
    monomodel.set_parameter_vector(params)
    ll = np.sum(-2*lc[:,2]**-2*(lc[:,1]-monomodel.get_value(lc[:,0]))**2)#-(0.5*len(lc[:,0]))*np.log(2*np.pi*lc[:,2]**2) #Constant not neceaary for gradient descent
    lp = MonoLnPrior(params,priors)
    return (ll+lp)

def MonoOnlyNegLogProb(params,lc,priors,monomodel):
    return -1*MonoOnlyLogProb(params,lc,priors,monomodel)


class RotationTerm(celerite.terms.Term):
    parameter_names = ("log_amp", "log_timescale", "log_period", "log_factor")

    def get_real_coefficients(self, params):
        log_amp, log_timescale, log_period, log_factor = params
        if type(log_factor)==np2.numpy_boxes.ArrayBox:
            log_factor=log_factor._value#flatten#.astype(float)
            #log_factor=log_factor.astype(float)
            #print(log_factor)
        #    f = log_factor.__rpow__(np.exp(1))
        #    print(f)
        #    return (log_amp.__rpow__(np.exp(1)).__mul__(f.__add__(1.0)).__div__(f.__add__(2.0)),log_timescale.__mul__(-1).__rpow__(np.exp(1)))
        #else:
        f = np2.exp(log_factor)
        return (np2.exp(log_amp) * (1.0 + f) / (2.0 + f),np2.exp(-log_timescale))

    def get_complex_coefficients(self, params):
        log_amp, log_timescale, log_period, log_factor = params
        f = np2.exp(log_factor)
        return (
            np2.exp(log_amp) / (2.0 + f),
            0.0,
            np2.exp(-log_timescale),
            2*np.pi*np2.exp(-log_period),
        )

def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

def grad_neg_log_like(params, y, gp):
    gp.set_parameter_vector(np.array(params))
    return -gp.grad_log_likelihood(y)[1]

'''
def grad_nll(p,gp,y,prior=[]):
    #Gradient of the objective function for TrainGP
    gp.set_parameter_vector(p)
    return -gp.grad_log_likelihood(y, quiet=True)

def nll_gp(p,gp,y,prior=[]):
    #inverted LnLikelihood function for GP training
    gp.set_parameter_vector(p)
    if prior!=[]:
        prob=MonoLogProb(p,y,prior,gp)
        return -prob if np.isfinite(prob) else 1e25
    else:
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25'''

def VelToOrbit(vel, Rs, Ms, dens=None, ecc=0, omega=0,timebin=86400.0):
    '''Takes in velocity (in units of stellar radius), Stellar radius estimate and (later) eccentricity & angle of periastron.
    Returns Semi major axis (AU) and period (days)'''
    if dens is None:
        dens=Ms/Rs**3
    Per=18226*dens/(vel**3)
    # 13.7375 = G*M_sol*au^-1*R_sun^-2*(secs/d)^2 = ((1.3271244e20)/(695700000/86400)**2)/1.49e11
    SMA= 13.7375*(Ms*(Rs/vel)**2)
    return SMA, Per

def getKeplerLDs(Ts,logg=4.43812,FeH=0.0,how='2'):
    #Get Kepler Limb darkening coefficients.
    from scipy.interpolate import CloughTocher2DInterpolator as ct2d
    #print(label)
    types={'1':[3],'2':[4, 5],'3':[6, 7, 8],'4':[9, 10, 11, 12]}
    if how in types:
        checkint = types[how]
        #print(checkint)
    else:
        print("no key...")

    #arr = np.genfromtxt("KeplerLDlaws.txt",skip_header=2)
    try:
        arr = np.genfromtxt("KeplerLDlaws.txt",skip_header=2)
    except:
        arr = np.genfromtxt("KeplerLDlaws.txt",skip_header=2)
    FeHarr=np.unique(arr[:, 2])
    #Just using a single value of FeH
    if not (type(FeH)==float) and not (type(FeH)==int):
        FeH=np.nanmedian(FeH)
    FeH=FeHarr[find_nearest(FeHarr,FeH)]

    feh_ix=arr[:,2]==FeH
    feh_ix=arr[:,2]==FeH
    Tlen=1 if type(Ts)==float or type(Ts)==int else len(Ts)
    outarr=np.zeros((Tlen,len(checkint)))
    for n,i in enumerate(checkint):
        u_interp=ct2d(np.column_stack((arr[feh_ix,0],arr[feh_ix,1])),arr[feh_ix,i])
        if (type(Ts)==float)+(type(Ts)==int) and ((Ts<50000.)*(Ts>=2000)):
            outarr[0,n]=u_interp(Ts,logg)
        elif ((Ts<50000.)*(Ts>=2000)).all():
            if type(logg)==float:
                outarr[:,n]=np.array([u_interp(T,logg) for T in Ts])
            else:
                outarr[:,n]=np.array([u_interp(Ts[t],logg[t]) for t in range(len(Ts))])
        else:
            print('Temperature outside limits')
            outarr=None
            break
    return outarr

def nonan(lc):
    return np.logical_not(np.isnan(np.sum(lc,axis=1)))

def AnomCutDiff(flux,thresh=4.2):
    #Uses differences between points to establish anomalies.
    #Only removes single points with differences to both neighbouring points greater than threshold above median difference (ie ~rms)
    #Fast: 0.05s for 1 million-point array.
    #Must be nan-cut first
    diffarr=np.vstack((np.diff(flux[1:]),np.diff(flux[:-1])))
    diffarr/=np.median(abs(diffarr[0,:]))
    anoms=np.hstack((True,((diffarr[0,:]*diffarr[1,:])>0)+(abs(diffarr[0,:])<thresh)+(abs(diffarr[1,:])<thresh),True))
    return anoms

def CalcTdur(vel, b, p):
    '''Caculates a velocity (in v/Rs) from the input transit duration Tdur, impact parameter b and planet-to-star ratio p'''
    # In Rs per day
    return (2*(1+p)*np.sqrt(1-(b/(1+p))**2))/vel

def CalcVel(Tdur, b, p):
    '''Caculates a velocity (in v/Rs) from the input transit duration Tdur, impact parameter b and planet-to-star ratio p'''
    # In Rs per day
    return (2*(1+p)*np.sqrt(1-(b/(1+p))**2))/Tdur

def PlotBestMono(lc, monomodel, vector, fname=None):
    cad=np.median(np.diff(lc[:,0]))
    t=np.arange(lc[0,0],lc[-1,0]+0.01,cad)
    monomodel.set_parameter_vector(vector)
    ypred=monomodel.get_value(t)
    plt.errorbar(lc[:, 0], lc[:,1], yerr=lc[:, 2], fmt='.',color='#999999')
    plt.plot(lc[:, 0], lc[:,1], '.',color='#333399')
    plt.plot(t,ypred,'--',color='#003333',linewidth=2.0,label='Median transit model fit')
    if fname is not None:
        plt.savefig(fname)

def PlotModel(lc, model, vector, prevmodels=[], plot=True, residuals=False, scale=1, nx=10000, GP=False, subGP=False, monomodel=0, verbose=True,fname=None,title=True):
    # - lc : lightcurve
    # - model : celerite-style model (eg gp) to apply the vector to
    # - vector : best-fit parameters to use
    cad=np.median(np.diff(lc[:,0]))
    t=np.arange(lc[0,0],lc[-1,0]+cad*0.5,cad)
    if len(prevmodels)==0:
        model.set_parameter_vector(vector)
        if GP:
            model.compute(lc[:,0],lc[:,2])
            ypreds,varpreds=model.predict(lc[:,1], t)
            stds=np.sqrt(np.diag(varpreds))
            modelfits=np.column_stack((ypreds-stds*2,ypreds-stds,ypreds,ypreds+stds,ypreds+stds*2,model.mean.get_value(t)))

        else:
            ypreds = model.get_value(t)
            stds=np.nanstd(lc[:,1]),len(t)*np.sqrt(len(lc[:,0]))
            modelfits=np.column_stack((ypreds-stds*2,ypreds-stds,ypreds,ypreds+stds,ypreds+stds*2,ypreds))
        #model.mean.get_value(t)
    else:
        modelfits=prevmodels

    if residuals:
        #Subtracting bestfit model from both flux and model to give residuals
        newmodelfits=np.column_stack((modelfits[:,:5]-np.tile(modelfits[:,2], (5, 1)).swapaxes(0, 1),modelfits[:,5])) #subtracting median fit
        Ploty=lc[:,1]-modelfits[:, 2][(np.round((lc[:,0]-lc[0,0])/0.020431700249901041)).astype(int)]
        #p.xlim([t[np.where(redfits[:,2]==np.min(redfits[:,2]))]-1.6, t[np.where(redfits[:,2]==np.min(redfits[:,2]))]+1.6])
    elif subGP:
        newmodelfits=np.column_stack((modelfits[:,:5]-np.tile(modelfits[:,5], (5, 1)).swapaxes(0, 1),modelfits[:,5])) #subtracting median fit
        Ploty=lc[:,1]-modelfits[:,5][(np.round((lc[:,0]-lc[0,0])/0.020431700249901041)).astype(int)]
        #p.xlim([t[np.where(redfits[:,2]==np.min(redfits[:,2]))]-1.6, t[np.where(redfits[:,2]==np.min(redfits[:,2]))]+1.6])
    else:
        Ploty=lc[:,1]
        newmodelfits=np.copy(modelfits)

    if plot:
        plt.errorbar(lc[:, 0], Ploty, yerr=lc[:, 2], fmt=',',color='#999999',alpha=0.8,zorder=-100)
        plt.plot(lc[:, 0], Ploty, '.',color='#333399')

        #Plotting 1-sigma error region and models
        plt.fill(np.hstack((t,t[::-1])),
                 np.hstack((newmodelfits[:,2]-(newmodelfits[:,2]-newmodelfits[:,1]),(newmodelfits[:,2]+(newmodelfits[:,3]-newmodelfits[:,2]))[::-1])),
                 '#3399CC', linewidth=0,label=r'$1-\sigma$ region ('+str(scale*100)+'% scaled)')
        #if not residuals:
        #    #Plotting fill from transit-less model:
        #    plt.fill(np.hstack((t,t[::-1])),
        #             np.hstack(((1.0+newmodelfits[:,2]-newmodelfits[:,5])-(newmodelfits[:,2]-newmodelfits[:,1]),((1.0+newmodelfits[:,2]-newmodelfits[:,5])+(newmodelfits[:,3]-newmodelfits[:,2]))[::-1])),
        #             '#66BBCC', linewidth=0,label='$1-\sigma$ region without transit')

        plt.plot(t,newmodelfits[:,2],'-',color='#003333',linewidth=2.0,label='Median model fit')
        if not residuals and not subGP and GP:
            plt.plot(t,model.mean.get_value(t),'--',color='#003333',linewidth=2.0,label='Median transit model fit')
        elif not residuals and not subGP and not GP:
            plt.plot(t,model.get_value(t),'--',color='#003333',linewidth=2.0,label='Median transit model fit')

    if not residuals and not subGP:
        #Putting title on upper (non-residuals) graph
        if title:
            plt.title('Best fit model')
        plt.legend(loc=2,fontsize=9)

    if fname is not None:
        plt.savefig(fname)
    plt.xlim(np.min(lc[:,0]),np.max(lc[:,0]))
    return modelfits

#def logclick(string,level='info'):
#    #Function to both print string to terminal (with click) and log string to file.
#    click.echo(string) if verbose else 0
#    exec('logger.'+level+'(string)')

def RunShort(epic):
    st=namaste.Star(str(epic),namaste.Settings(nsteps=30000,nthreads=4))
    st.csvfile_dat('Best_Stellar_params_NoGRIZ.csv')
    st.initLD()
    st.addLightcurve('Lightcurves/EPIC'+epic+'_bestlc_ev.npy')
    st.addGP()
    st.Optimize_GP()
    params=pd.DataFrame.from_csv('orb_params_sorted.csv')
    #2121.02,0.5,0.0032
    st.AddMonotransit(params.loc[int(epic),'cen'],params.loc[int(epic),'dur'],params.loc[int(epic),'dep'])
    st.SaveAll()
    st.RunMCMC()
    st.PlotMCMC()
    st.SaveMCMC()

if __name__=='__main__':
    RunShort(sys.argv[1])
