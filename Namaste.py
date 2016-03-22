'''
This module runs the MCMC code on any single transit, 

Inputs:
   Name of object
   Lightcurve fits file location
   Initial estimates of lightcurve parameters
   Photometric colours OR directly determined input stellar parameters

Outputs the median value and errors for:
   Time of transit, Tcen
   Orbital Velocity, v
   Impact Parameter, b
   Planet to Star ratio, r
   Stellar flux level, F
And saves a triangle plot, output values and ''sampler'' file to disc
'''

import numpy as np
import pandas as pd
import glob
import pylab as p
#import me.planetlib as pl
#from StellarFits import *
from Monotransit import *
from planetlib import *
import scipy.optimize as optimize
import emcee
from os import sys, path
import csv
import pylab as p

#Getting working directory of Namaste
Namwd = path.dirname(path.realpath(__file__))
    
def Run(KIC, lc, guess, nstep=20000, StarDat=[], verbose=False, GP=False):
    '''
    INPUTS:
        KIC as EPIC (K2) or KIC (Kepler) identifier
        guess: list of [Tcen, Tdur, depth], or []
        nstep: number of steps in MCMC (default 15000)
        StarDat: Stellar data. Ts,eTs1,eTs2,Rs,eRs1,eRs2,Ms,eMs1,eMs2.
        lc: is input lightcurve filename
            In fits file, or in format time, normalised flux and flux errors.
        verbose: if 1, displays extra plots and text during MCMC run
    OUTPUTS:
        lsq_fit: Best-fit value
        sampler: samples from emcee output
        weights: Weights given to each point in lc
        finalmodel: Best-fitting model mapped to input lc
        StarDat: StarData, as inputted
        lccut: 'Cut' lightcurve
    '''

    #Getting lightcurve
    if lc==0:
        lc,h=getKeplerLC(KIC)
    elif type(lc)==str:
        lc,h=ReadLC(lc)
    print h

    #Doing manual transit search if transit unknown
    if 0 in guess:
        guess=FindTransitManual(KIC, lc, cut=25.0)
    NoPrevious=True
    if path.isfile(Namwd+'/Outputs/'+str(KIC)+"sampler.npy"):
        #Finding previous output file. Using this instead of guess
        NoPrevious=False
        sams=np.load(Namwd+'/Outputs/'+str(KIC)+"sampler.npy")
        sams[:, 1]=abs(sams[:, 1])
        #Taking last 1000 values in case previous run was poorly burned in
        params=np.median(sams[-1000:, :], axis=0)
        Tdur=(2*(1+params[3])*np.sqrt(1-(params[1]/(1+params[3]))**2))/params[2]
        Tdur=CalcTdur(np.median(sams[:,2]),np.median(sams[:,1]),np.median(sams[:,3]))
        Tcen=params[0]
        ratio=params[2]
        bestparams=params
        print 'Found Old Params for '+str(KIC)
    else:
        NoPrevious=True
        Tcen=guess[0]
        Tdur=guess[1]
        ratio=guess[2]**0.5
        params=[Tcen, 0.4, CalcVel(Tdur, 0.4, ratio), ratio, np.median(lc[abs(lc[:, 0]-Tcen)>1.1*Tdur, 1]), 0,0]
    if StarDat==[]:
        StarDat=getStarDat(KIC)
    
    #Getting Quadratic Limb Darkening Paramters from Temperature:
    LD=getKeplerLDs(StarDat[0])
    LDRange=np.array([getKeplerLDs(StarDat[0]+StarDat[1]), getKeplerLDs(StarDat[0]-StarDat[2])])
    params[-2:]=LD

    #Cutting lightcurve to 15 Tdurs. Then flattening lightcurve
    lc=lc[np.where(abs(lc[:, 0]-Tcen)<Tdur*25.0)[0]]

    uniformprior=np.array([[params[0]-2*Tdur, params[0]+2*Tdur], [-1.2, 1.2], [0.0, CalcVel(Tdur*0.95, 0.0, 0.3)], [0, 0.3], [0.95, 1.05], [LD[0], np.max(abs(LD[0]-LDRange[:, 0]))], [LD[1], np.max(abs(LD[1]-LDRange[:, 1]))]])
    #print len(uniformprior)
    #print [[uniformprior[5][0], uniformprior[5][1]], [uniformprior[6][0], uniformprior[6][1]]]
    xtol=1e-11; ftol=1e-9

    #Diverging the GP versions here:
    if not GP:
        print "Running Namaste without GP"
        #Flattening lightcurve
        print lc
        lc=k2f(lc, stepsize=Tdur/2.0, winsize=Tdur*8.5,  niter=40)
        #Further cutting lightcurve around transit
        lccut=lc[np.where(abs(lc[:, 0]-Tcen)<Tdur*6.0)[0]]

        #Guesing b as 0.4 to begin with.
        bestparams=np.array(params, copy=True)
        if verbose==1:
            PlotLC(lccut)
            p.plot(np.arange(lccut[0, 0], lccut[-1, 0], 0.001), modelmonotransit(params, np.arange(lccut[0, 0], lccut[-1, 0], 0.001)))
            p.title('Initial Fit with'+str(params))
            #p.pause(5)
            #p.clf()

        '''
            PARAMS are:
            the time of conjunction for each individual transit (Tc),  0.1
            the impact parameter (b = a cos i/Rstar)  0.5
            vel -- scalar. planetary perpendicular velocity (in units of stellar radii per day) 1
            planet-to-star radius ratio (Rp/Rstar),  0.03
            stellar flux (F0),   0.001
            the limb-darkening parameters u1 and u2   0.1
        '''

        #Uniform priors set for each value in params
        #Last two values now gaussian priors for limb darkening params:

        #Setting up emcee minimization. Do we need all this minimization??


        ndim = len(params)
        covar = np.zeros((ndim, ndim), dtype=float)
        diag=np.zeros((ndim), dtype=float)
        for ii in range(ndim):
            covar[ii,ii] = 0.35*(abs(bestparams[ii]-uniformprior[ii][0])+abs(bestparams[ii]-uniformprior[ii][1]))
            diag[ii]=0.35*(abs(bestparams[ii]-uniformprior[ii][0])+abs(bestparams[ii]-uniformprior[ii][1]))

        weights = np.array(lccut[:, 2]**-2, copy=True)
        #fitkw = dict(=uniformprior)
        fitargs = (modelmonotransit, lccut[:, 0], lccut[:, 1], weights, uniformprior) #time, NL, NP, errscale, smallplanet, svs, data, weights, fitkw)

        bestchisq = errfunc(bestparams, *fitargs)

        nwalkers =20 * ndim
        nthread=4

        # Initialize sampler:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfunc, args=fitargs, threads=4)

        p0 = np.random.multivariate_normal(bestparams, covar, nwalkers*5)
        badp0 = ((abs(p0[:,1]) >1.5) + (p0[:,2] < 0) + (p0[:,3] < 0) + (p0[:,4] < 0))
        p0 = np.vstack((bestparams, p0[np.nonzero(True-badp0)[0][0:nwalkers-1]]))
        pos = p0

        # Run burn-in; exclude and remove bad walkers:
        pos, lp, _ = sampler.run_mcmc(pos,1200)

        for ii in range(2):
            pos, prob, state = sampler.run_mcmc(pos, max(1, (ii+1)*nstep/5))
            if (bestchisq + 2*max(prob)) > ftol: #-2*prob < bestchisq).any(): # Found a better fit! Optimize:
                #if verbose: print "Found a better fit; re-starting MCMC... (%1.8f, %1.8f)" % (bestchisq, (bestchisq + 2*max(prob)))
                print str(KIC)+" - Found a better fit; re-starting MCMC..."
                #pdb.set_trace()
                bestparams = pos[(prob==prob.max()).nonzero()[0][0]].copy()
                lsq_fit = optimize.leastsq(devfuncup, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
                bestparams = lsq_fit[0]
                covar = lsq_fit[1]
                bestchisq = errfunc(bestparams, *fitargs)
                ii = 0 # Re-start the burn-in.
                #pos[prob < np.median(prob)] = bestparams
            badpos = (prob < np.median(prob))
            if (pos[True-badpos].std(0) <= 0).any():
                goodpos_unc = np.vstack((np.abs(bestparams / 50.), pos[True-badpos].std(0))).max(0)
                pos[badpos] = np.random.normal(bestparams, goodpos_unc, (badpos.sum(), ndim))
            else:
                pos[badpos]  = np.random.multivariate_normal(bestparams, np.cov(pos[True-badpos].transpose()), badpos.sum())

        pos[badpos] = bestparams

        # Run main MCMC run:
        #pdb.set_trace()
        print str(KIC)+" running initial MCMC"
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep)

        # Test if MCMC found a better chi^2 region of parameter space:
        mcmc_params = sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==sampler.lnprobability.ravel().max())[0][0]]
        mcmc_fit = optimize.leastsq(devfuncup, mcmc_params, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
        if errfunc(mcmc_fit[0], *fitargs) < errfunc(bestparams, *fitargs):
            bestparams = mcmc_fit[0]
        else:
            pass

        lsq_fit = optimize.leastsq(devfuncup, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
        bestparams = lsq_fit[0]
        covar = lsq_fit[1]
        bestchisq = errfunc(bestparams, *fitargs)
    elif GP:
        #Running the emcee with Gaussian Processes.
        print "Running Namaste with GP (George)"

        #Removing 'Flux' term from parameters (GP will fit this)
        params=np.hstack((params[0:3],params[4:]))
        uniformprior=np.hstack((uniformprior[0:3],uniformprior[4:]))

        if george.__version__[0]!='1':
            print "This script uses George 1.0."
            print "See http://github.com/dfm/george"
            print "Or download with pip using \"pip install git+http://github.com/dfm/george.git@1.0-dev\" "
            return None
        #Running it on 15-Tdur-wide window
        lccut=lc[np.where(abs(lc[:, 0]-Tcen)<Tdur*15.0)[0]]
        gp, res, lnlikfit = TrainGP(lc,Tcen,h[3])
        newmean,newwn,a,tau=res

        #Initialising Monotransit class.
        mono=Monotransit(params)
        newparams=np.hstack((params,newwn,a,tau))
        mono.set_vector(newparams[:6])

        #Setting up george variable
        model=george.GP(np.exp(newparams[-2])*george.kernels.ExpSquaredKernel(np.exp(newparams[-1])),mean=mono,fit_mean=True,white_noise=newparams[-3],fit_white_noise=True)
        model.compute(lccut[:,0],lccut[:,2])
        #Setting up emcee run
        fitargs = (lccut[:,1], uniformprior,model)
        initial = model.get_vector()

        '''
        PARAMS (initial) are:
            the time of conjunction for each individual transit (Tc),  0.1
            the impact parameter (b = a cos i/Rstar)  0.5
            vel -- scalar. planetary perpendicular velocity (in units of stellar radii per day) 1
            planet-to-star radius ratio (Rp/Rstar),  0.03
            the limb-darkening parameters u1 and u2   0.1
            white noise (ln(whitenoise)^2)
            a (ExpSquared GP amplitude)
            tau (ExpSquared GP timescale)
        '''

        ndim, nwalkers = len(initial), 32
        covar = np.zeros((ndim, ndim), dtype=float)
        diag=np.zeros((ndim), dtype=float)
        for ii in range(ndim):
            if ii<len(uniformprior):
                covar[ii,ii] = 0.25*(abs(initial[ii]-uniformprior[ii][0])+abs(initial[ii]-uniformprior[ii][1]))
                diag[ii]=0.25*(abs(initial[ii]-uniformprior[ii][0])+abs(initial[ii]-uniformprior[ii][1]))
            else:
                #GP objects dont have priors... Using square root of absolute GP params
                covar[ii,ii] = 0.25*np.sqrt(abs(initial[ii]))*(1.0 if initial[ii]>0 else -1.0)
                diag[ii]= 0.25*np.sqrt(abs(initial[ii]))*(1.0 if initial[ii]>0 else -1.0)

        print initial
        print covar
        p0 = np.vstack((initial, np.random.multivariate_normal(initial, covar, nwalkers)))
        #p0 = initial + 1e-6 * np.random.randn(nwalkers, ndim) #Replaced

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobGP, args=fitargs)


        print("Running burn-in...")
        p0, prob, state = sampler.run_mcmc(p0, nstep/5)
        print sampler
        print p0
        PlotModel(lccut, sampler, GP=True)
        sampler.reset()

        print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, nstep);
        mcmc_params = sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==sampler.lnprobability.ravel().max())[0][0]]
        mcmc_fit = optimize.leastsq(devfuncup, mcmc_params, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)

        if lnprobGP(mcmc_fit[0], *fitargs) < lnprobGP(bestparams, *fitargs):
            bestparams = mcmc_fit[0]
        else:
            pass

    finalmodel = modelmonotransit(bestparams, fitargs[1])
    print str(KIC)+" finished MCMC"
    return lsq_fit, sampler, weights, finalmodel, StarDat,  lccut

def Pmin(lc, Tcen, Tdur):
    #Calculating minimum period from the lightcurve
    return np.max(abs(Tcen-lc[:, 0]))-Tdur

def nll(p,gp,y):
    gp.set_vector(p)
    ll = gp.lnlikelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p,gp,y):
    gp.set_vector(p)
    return -gp.grad_lnlikelihood(y, quiet=True)

def FixingExtent(pos):
    #Finding out if the MCMC has converged on a single value anywhere, and fixing this to add artificial variation
    NoExtent=np.where(np.std(pos, axis=0)<1e-6)[0]
    if len(NoExtent)>0:
        for p in NoExtent:
            #Finding out how many times less than 1e-6 the standard deviation is, and multiplying by that difference to get it back to acceptable levels
            ratiobelow=1e-6/np.std(pos[:, p])
            pos[:, p]*=np.random.normal(1.5*ratiobelow, 0.5*ratiobelow, len(pos[:, p]))
    return pos

def CalcTdur(vel, b, p):
    '''Caculates a velocity (in v/Rs) from the input transit duration Tdur, impact parameter b and planet-to-star ratio p'''
    # In Rs per day
    return (2*(1+p)*np.sqrt(1-(b/(1+p))**2))/vel
    
def CalcVel(Tdur, b, p):
    '''Caculates a velocity (in v/Rs) from the input transit duration Tdur, impact parameter b and planet-to-star ratio p'''
    # In Rs per day
    return (2*(1+p)*np.sqrt(1-(b/(1+p))**2))/Tdur

def VelToOrbit(Vel, Rs, Ms, ecc=0, omega=0):
    '''Takes in velocity (in units of stellar radius), Stellar radius estimate and (later) eccentricity & angle of periastron.
    Returns Semi major axis (AU) and period (days)'''
    import me.constants as c
    Rs=Rs*c.Rsun if Rs<5 else Rs
    Ms=Ms*c.Msun if Ms<5 else Ms
    SMA=(c.G*Ms)/((Vel*Rs/86400.)**2)
    Per=(2*np.pi*SMA)/(Vel*Rs/86400)
    return SMA/c.AU, Per

def PerToVel(Period, Rs, Ms):
    return ((2*np.pi*6.67e-11*Ms*1.96e30)/(Period*86400*(Rs*695000000)**3))**(1/3.)

def PlotMCMC(params,  EPIC, sampler, lc, KepName=''):
    import pylab as p
    import triangle
    if type(sampler)!=np.ndarray:
        samples = sampler.chain[:, -10000:, :].reshape((-1, len(params)))
    else:
        samples=sampler
    #Turning b into absolute...
    samples[:,1]=abs(samples[:,1])
    #np.savetxt('Kep103B.txt', samples)
    p.figure(1)
    fig = triangle.corner(samples, labels=["$T_{c}$","$b$","$v$","$Rp/Rs$","$u1$","$u2$","$tau$","$a$"],quantiles=[0.16, 0.5, 0.84], plot_datapoints=False)
    #Plotting model in top right
    p.subplot(7,7,10).axis('off')
    if KepName=='':
        if str(int(EPIC)).zfill(9)[0]=='2':
            KepName='EPIC'+str(EPIC)
        else:
            KepName='KIC'+str(EPIC)
    p.title(KepName, fontsize=22)
    ax = p.subplot2grid((7,7), (0, 4), rowspan=2, colspan=3)
    #p.errorbar(lc[:, 0], lc[:, 1], yerr=lc[:, 1], fmt='.',color='#EEEEEE')
    #p.plot(lc[:, 0], lc[:, 1], '.',color='#')
    modelfits=PlotModel(lc,  samples,scale=3)
    #plotting residuals:
    ax = p.subplot2grid((7,7), (2, 4), rowspan=1, colspan=3)
    modelfits=PlotModel(lc,  samples, modelfits, residuals=True, scale=3)
    if os.path.exists('/home/astro/phrnbe/SinglesSC/TestMCMCs/Pcorner_fit_'+str(EPIC)+'Aug.pdf'):
        if os.path.exists('/home/astro/phrnbe/SinglesSC/TestMCMCs/Pcorner_fit_'+str(EPIC)+'2Aug.pdf'):
            fname='/home/astro/phrnbe/SinglesSC/TestMCMCs/Pcorner_fit_'+str(EPIC)+'3Aug.pdf'
        else:
            fname='/home/astro/phrnbe/SinglesSC/TestMCMCs/Pcorner_fit_'+str(EPIC)+'2Aug.pdf'
    else:
        fname='/home/astro/phrnbe/SinglesSC/TestMCMCs/Pcorner_fit_'+str(EPIC)+'Aug.pdf'
    p.savefig(fname,Transparent=True,dpi=300)
    return modelfits
    
def GetVel(sampler, StarDat, nsamp):
    #Taking random Nsamples from samples to put through calculations
    #Need to form assymetric gaussians of Star Dat parameters if not equal
    Rstardist=np.random.normal(StarDat[3], (StarDat[5]+StarDat[4])/2, nsamp)
    Mstardist=np.random.normal(StarDat[6], (StarDat[7]+StarDat[8])/2, nsamp)
    #Rstardist2=np.hstack((np.sort(Rstardist[:, 0])[0:int(nsamp/2)], np.sort(Rstardist[:, 1])[int(nsamp/2):] ))
    Rplans, Ps, Smas, Krvs, Mps=[], [], [], [], []
    for n in range(nsamp):
        rn=np.random.randint(len(sampler[:,3]))
        Rplans+=[(sampler[rn, 3]*695500000*Rstardist[n])/7.15e7]
        sma, P = VelToOrbit(sampler[rn, 2], Rstardist[n], Mstardist[n])
        Ps+=[P/86400.]
        Smas+=[sma]
        #Krv = ((2.*np.pi*6.67e-11)/P)**(1./3.)*((Mp*np.sin(i))/(Ms**(2./3.)))) 
        Mp=(PlanetRtoM((sampler[rn, 3]*695500000*Rstardist[n])/6371000.0))*5.6e24
        Mps+=[Mp]
        Krvs+=[((2.*np.pi*6.67e-11)/P)**(1./3.)*((Mp)/((1.96e30*Mstardist[n])**(2./3.)))]
    #Putting gaussians through
    #Rtests=np.random.rand
    #.argsort()[int(np.round(0.16*5000))]
    sigs=[6.76676416,   18.39397206,   50.,81.60602794,93.23323584]
    #print np.percentile(np.array(Krvs), sigs)
    #print np.percentile(np.array(Mps)/c.Mjup, sigs)
    #print np.percentile(np.array(Rplans), sigs)
    return np.percentile(np.array(Rplans), sigs),  np.percentile(np.array(Ps), sigs), np.percentile(np.array(Smas), sigs), np.percentile(np.array(Krvs), sigs)

def PlotModel(lccut, sampler, models='', residuals=False, scale=1, nx=10000,GP=0,monomodel=0):
    '''This plots the single transit, with the uncertainty area of the fit (scaled by ''scale'' parameter for ease of view)
    If residuals, the model is subtracted with only residuals left.
    #Inputs:
    lccut - lightcurve cut around transit
    sampler - MCMC output
    models - percentiles of the best-fit model for each timestamp
    residuals - specifies whether to plot residuals or normal
    scale - allows the 'filled' 1-sigma area to be scaled by a factor for ease of view
    #Outputs:
    models (in columns of -2.-1,0,1 and 2 -sigma uncertainties) generated by sampling nx (10000) random samples from the MCMC output
    '''
    #Getting the times that are not in t (eg due to time jumps)
    t=np.arange(lccut[0,0],lccut[-1,0]+0.01,0.020431700249901041)

    if not GP:
        #Generating model fit regions
        if len(models)==0:
            #Getting models by looping threw samples and adding flux for each to large array
            modelouts=np.zeros((len(t), nx))
            idx=np.random.randint(len(sampler[:,0]), size=nx)
            for i in range(nx):
                modelouts[:,i]=modelmonotransit(sampler[idx[i],:],t)
            # Turning this nx*nt array into a 5*t array of-2,-1,0,+1 & +2 sigma models
            modelfits=np.percentile(modelouts,[6.76676416,   18.39397206,   50., 81.60602794,93.23323584],axis=1)
            modelfits=np.swapaxes(modelfits, 1, 0)
        else:
            modelfits=models
    else:
        params=np.median(sampler,axis=0)
        if monomodel==0:
            monomodel=Monotransit(params)
        model=george.GP(np.exp(params[-2])*george.kernels.ExpSquaredKernel(np.exp(params[-1])),mean=monomodel,fit_mean=True,white_noise=params[-3],fit_white_noise=True)
        model.compute(lc[:,1],lc[:,2])
        ypreds,varpreds=model.predict(lc[:,1], np.arange(lc[0,0],lc[-1,0],0.01))
        stds=np.sqrt(np.diag(varpreds))
        #Turing into y-2s,y-1s,y,y+1s,y+2s format.
        modelfits=np.column_stack((ypreds-stds*2,ypreds-stds,ypreds,ypreds+stds,ypreds+stds*2))
            #,np.percentile(modelouts,[6.76676416,   18.39397206,   50., 81.60602794,93.23323584],axis=1)
        modelfits=np.swapaxes(modelfits, 1, 0)

    newmodelfits=np.copy(modelfits)
    if residuals:
        #Subtracting bestfit model from both flux and model to give residuals
        newmodelfits=modelfits-np.tile(modelfits[:,2], (5, 1)).swapaxes(0, 1) #subtracting median fit
        lccut[:,1]-=modelfits[:, 2][(np.round((lccut[:,0]-lccut[0,0])/0.020431700249901041)).astype(int)]
    #p.xlim([t[np.where(redfits[:,2]==np.min(redfits[:,2]))]-1.6, t[np.where(redfits[:,2]==np.min(redfits[:,2]))]+1.6])

    #Plotting data
    p.errorbar(lccut[:, 0], lccut[:,1], yerr=lccut[:, 2], fmt='.',color='#999999')
    p.plot(lccut[:, 0], lccut[:,1], '.',color='#333399')

    #Plotting 1-sigma error region and models
    p.fill(np.hstack((t,t[::-1])),np.hstack((newmodelfits[:,2]-(newmodelfits[:,2]-newmodelfits[:,1]),(newmodelfits[:,2]+(newmodelfits[:,3]-newmodelfits[:,2]))[::-1])),'#33BBFF', linewidth=0,label='1-sigma uncertainties scaled by '+str(scale*100)+'%',alpha=0.5)
    p.plot(t,newmodelfits[:,2],'-',color='#003333',linewidth=2.0,label='Median model fit')

    if not residuals:
        #Putting title on upper (non-residuals) graph
        p.title('Best fit model')
        p.legend()
    return modelfits


def StartSingleRun(kic):
    '''
    This script uses the KOI data table and associated parameters to start a 'Namaste' run.
    Finding the transit is still required
    '''
    kic=int(kic)
    AllKicData=KicData(kic)
    lccut, TransData=FindTransitManual(kic)
    savedlcname=Sourcedir+'KeplerFits/KOI'+str(AllKicData['kepoi_name'].values[0])+'/'+'Transit_LC_at_'+str(TransData[0])+'_lc.txt'
    np.savetxt(savedlcname, lccut)
    StarDat=[AllKicData['koi_steff'].values[0], AllKicData['koi_steff_err1'].values[0],  AllKicData['koi_steff_err2'].values[0], AllKicData['koi_srad'].values[0], AllKicData['koi_srad_err1'].values[0],  AllKicData['koi_srad_err2'].values[0], AllKicData['koi_smass'].values[0], AllKicData['koi_smass_err1'].values[0],  AllKicData['koi_smass_err2'].values[0]]
    #Currently just fudging it to run the other python script from the command line
    print 'python McmcSingles.py \''+str(kic)+', '+savedlcname+', '+str(TransData).replace(' ','')+' , 28000, '+str(StarDat).replace(' ','')+'\'' #['+TransData[0]+', '+TransData[1]+', '+TransData[2]+']


def SaveOutputs(KIC, lsq_fit, weights, finalmodel, lc, samplercut, Rps, Ps, As, StarDat, sigs=np.array([2.2750131948178987, 15.865525393145707, 50.0, 84.13447460685429, 97.7249868051821])):
    print 'Saving '+str(KIC)+' outputs to file'
    np.save(Namwd+'/Outputs/'+str(KIC)+"lsq_fitAug", lsq_fit)
    np.save(Namwd+'/Outputs/'+str(KIC)+"weightsAug", weights)
    np.save(Namwd+'/Outputs/'+str(KIC)+"finalmodelAug", finalmodel)
    np.save(Namwd+'/Outputs/'+str(KIC)+"lightcurveAug", lc)
    np.save(Namwd+'/Outputs/'+str(KIC)+"samplerAug", samplercut)
    #Tcen, b, v, p0, F0, u1, u2, Rp, P, sma, Ts, Rs, Ms - 1- and 2-sigma boundaries for each.
    params=np.vstack((np.percentile(samplercut[:, 0], sigs), np.percentile(samplercut[:, 1], sigs), np.percentile(samplercut[:, 2], sigs), np.percentile(samplercut[:, 3], sigs), np.percentile(samplercut[:, 4], sigs), np.percentile(samplercut[:, 5], sigs), np.percentile(samplercut[:, 6], sigs), Rps, Ps, As, np.array([StarDat[0]-2*StarDat[1], StarDat[0]-StarDat[1], StarDat[0], StarDat[0]+StarDat[2], StarDat[0]+2*StarDat[2]]), np.array([StarDat[3]-2*StarDat[4], StarDat[3]-StarDat[4], StarDat[3], StarDat[3]+StarDat[5], StarDat[3]+2*StarDat[5]]), np.array([StarDat[6]-2*StarDat[7], StarDat[6]-StarDat[7], StarDat[6], StarDat[6]+StarDat[8], StarDat[6]+2*StarDat[8]]) ))

    np.savetxt(Namwd+'/Outputs/'+str(KIC)+"params.txt", params)
    #Putting parameters in a master file of all Namaster runs
    filename=(Namwd+'/Outputs/FullMcmcmOutput.csv')
    if path.exists(filename):
        c = csv.writer(open(filename,'a'))
        c.writerow(list(params))
    else:
        #Writing header
        c = csv.writer(open(filename, "wb"))
        c.writerow(['#KEPID','_','_','T_cen','_','_','_','_','b','_','_','_','_','v','_','_','_','_','p0','_','_','_','_','F0','_','_','_','_','u1','_','_','_','_','u2','_','_','_','_','Rp','_','_','_','_','P','_','_','_','_','A','_','_','_','_','K_rv','_','_','_','_','T_s','_','_','_','_','R_s','_','_','_','_','M_s','_','_','COMEPLTE?','COMMENT?'])
        c.writerow(['#','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig','-2sig','-1sig','med','+1sig','+2sig'])
        c.writerow(list(params))


if __name__ == '__main__':
    '''
    #Running from the command line
    #Syntax: python Namaste.py KIC lcfilename, [tc , td , d , T,Terr , R,Rerr , M,Merr , nstep=15000]

    '''

    #NEW IMPORT STYLE:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    #Required args:
    parser.add_argument("KIC",type=int,help="Kepler (KIC) or K2 (EPIC) id")
    parser.add_argument("lcurve",type=str, help="File location of lightcurve file. Type \"get\" to download lightcurve")

    #Optional args:
    parser.add_argument("-n", "--nstep", dest="nstep", default=15000,
                      help="Number of MCMC steps. Default 15000")
    parser.add_argument("-g", "--gproc", action="store_true", dest="GP", default=False,
                      help="Use to turn on Gaussian Processes")

    parser.add_argument("-tc", "--tcen", dest="Tcen", default=0,type=float,
                      help="Initial central transit time estimate")
    parser.add_argument("-td", "--tdur", dest="Tdur", default=0,type=float,
                      help="Initial transit length/duration estimate")
    parser.add_argument("-d", "--dep", dest="depth", default=0,type=float,
                      help="Initial transit depth estimate")

    parser.add_argument("-ts", "--stemp", dest="Ts", default='0,0',type=str,
                      help="Stellar temperature and error (eg 5500,150). Can be estimated from colours")
    parser.add_argument("-rs", "--srad", dest="Rs", default='0,0',type=str,
                      help="Stellar radius and error (eg 1.1,0.1). Can be estimated from colours")
    parser.add_argument("-ms", "--smass", dest="Ms", default='0,0',type=str,
                      help="Stellar mass and error (eg 1.2,0.2). Can be estimated from colours")

    parser.add_argument("-q", "--quiet",
                      action="store_false", dest="verbose", default=True,
                    help="don't print status messages to stdout")
    args = parser.parse_args()

    #Assembing stellar parameters into array:
    try:
        temp=np.array(str(args.Ts).split(',')).astype(float)
        rad=np.array(str(args.Rs).split(',')).astype(float)
        mass=np.array(str(args.Ms).split(',')).astype(float)
    except ValueError:
        print "Error in temp, rad or mass. Assuming not known"
        temp,rad,mass=[0,0],[0,0],[0,0]
    StarDat=[temp[0],temp[1],temp[1],rad[0],rad[1],rad[1],mass[0],mass[1],mass[1]]
    if args.lcurve=="get":
        args.lcurve=getKeplerLC(args.KIC,getlc=False)
    if 0 in StarDat:
        if StarDat[0]!=0:
            #There's a zero somewhere, bu
            temp[1]=150 if temp[1]<150 else temp[1]
            rad=RfromT(temp[0],temp[1])
            mass=MfromR(rad[0],rad[1])
            StarDat=[temp[0],temp[1],temp[1],rad[0],rad[1],rad[1],mass[0],mass[1],mass[1]]
        else:
            #Get stardat from kic/fits file
            StarDat=getStarDat(args.KIC,args.lcurve)
    #Running Fit!
    print args.GP
    lsq_fit, sampler, weights, finalmodel, StarDat, lc = Run(args.KIC, args.lcurve, np.array((args.Tcen,args.Tdur,args.depth)).astype(float), float(args.nstep), StarDat,GP=args.GP)
    samplercut=sampler.chain[:,-10000:,:].reshape((-1,7))

    #Doing period calculations from velocity distribution
    Rps, Ps, As,  Krvs = GetVel(samplercut, StarDat, 5000)
    print str(args.KIC)+" - Period of "+str(Ps[3])+" +"+str(Ps[4]-Ps[3])+"/ -"+str(Ps[3]-Ps[2])
    print str(args.KIC)+" - Semi Major Axis of "+str(As[3])+" +"+str(As[4]-As[3])+"/ -"+str(As[3]-As[2])
    print str(args.KIC)+" - Radius of "+str(Rps[3])+" +"+str(Rps[4]-Rps[3])+"/ -"+str(Rps[3]-Rps[2])

    #Saving outputs to file
    SaveOutputs(args.KIC, lsq_fit, weights, finalmodel, lc, samplercut, Rps, Ps, As, StarDat)

    #Plotting mcmc
    fig=PlotMCMC(lsq_fit[0],  args.KIC, sampler, lc)
