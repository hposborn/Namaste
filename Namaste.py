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
import pylab as p
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
    
def Run(KIC, lc, guess, nstep=20000, StarDat=[], verbose=True, GP=False):
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

    #Doing manual transit search if transit unknown
    if 0 in guess:
        guess=FindTransitManual(KIC, lc, cut=25.0)
    NoPrevious=True

    #Finding previous output file. Using this instead of guess
    FindPrev=False
    if path.isfile(Namwd+'/Outputs/'+str(KIC)+"sampler.npy") and FindPrev==True:
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
    print str(KIC)+" Stellar data = "+str(StarDat) if verbose else 0

    #Getting Quadratic Limb Darkening Paramters from Temperature:
    LD=getKeplerLDs(StarDat[0])
    LDRange=np.array([getKeplerLDs(StarDat[0]+StarDat[1]), getKeplerLDs(StarDat[0]-StarDat[2])])
    params[-2:]=LD

    #Initialising Priors in form [uniform-,uniform+,gauss_mean,gauss_fwhm]
    priors=np.array([[params[0]-2*Tdur, params[0]+2*Tdur,0,0], \
                     [-1.2, 1.2,0,0], \
                     [0.0, CalcVel(Tdur*0.75, 0.0, 0.3),0,0], \
                     [0, 0.3,0,0], \
                     [0.95, 1.05,0,0], \
                     [0,1.0,LD[0], np.max(abs(LD[0]-LDRange[:, 0]))], \
                     [0,1.0,LD[1], np.max(abs(LD[1]-LDRange[:, 1]))]\
                     ])#reshaping into columns
    '''
    Priors:
    tcen:   +/- 2*Tdur
    b:      -1.2 : +1.2
    vel:    0 : max vel for 75% Tdur at b=0 and p0=0.3
    p0:     0 : 0.3
    u1:     Use Gauss u1 pm err
    u2:     Use Gauss u2 pm err
    wn:     Gamma prior +=(e-100*wn)
    a:      Gamma prior +=(e-100*a)
    tau:    Gamma prior +=(e-100*tau)
    '''

    #Finding jumps in lightcurve (where diff t > 1.0. GP or lcf may fail here). Cutting around these.
    jumps=np.hstack((0,np.where(np.diff(lc[:,0])>1.0)[0]+1))
    if len(jumps)>2:
        idx=np.searchsorted(lc[jumps,0],Tcen)
        lc=lc[jumps[idx-1]:jumps[idx],:]
    else:
        lc=lc[abs(Tcen-lc[:,0])<30.0,:]

    xtol=1e-11; ftol=1e-9

    #Setting number of threads based on number of cores...
    try:
        import multiprocessing
        nthread=int(multiprocessing.cpu_count())
    except:
        #Default=4
        nthread=4

    #Diverging the GP versions here:
    if not GP:
        print "Running Namaste without GP" if verbose else 0

        #Flattening lightcurve
        lc=k2f(lc, stepsize=Tdur/2.0, winsize=Tdur*6.7,  niter=40)
        #Further cutting lightcurve around transit
        lccut=lc[abs(lc[:, 0]-Tcen)<Tdur*6.0,:]

        #Guesing b as 0.4 to begin with.
        bestparams=np.array(params, copy=True)

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

        #Setting up emcee minimization.
        ndim = len(params)
        covar = np.zeros((ndim, ndim), dtype=float)
        diag=np.zeros((ndim), dtype=float)
        for ii in range(ndim):
            covar[ii,ii] = 0.35*(abs(bestparams[ii]-priors[ii][0])+abs(bestparams[ii]-priors[ii][1]))
            diag[ii]=0.35*(abs(bestparams[ii]-priors[ii][0])+abs(bestparams[ii]-priors[ii][1]))
        print "Diagonal of Covar = "+str(np.diag(covar)) if verbose else 0
        weights = np.array(lccut[:, 2]**-2, copy=True)
        #fitkw = dict(=priors)
        fitargs = (modelmonotransit, lccut[:, 0], lccut[:, 1], weights, priors) #time, NL, NP, errscale, smallplanet, svs, data, weights, fitkw)

        bestchisq = errfunc(bestparams, *fitargs)

        nwalkers = 20 * ndim

        # Initialize sampler:
        print "Initializing sampler" if verbose else 0
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfunc, args=fitargs, threads=nthread)

        p0 = np.random.multivariate_normal(bestparams, (1e-5)*covar, nwalkers*5)
        p0 = np.vstack((bestparams, p0[0:nwalkers-1]))

        print "Running first burn-in" if verbose else 0
        pos, lp, _ = sampler.run_mcmc(p0,nstep/5)
        sampler.reset()
        #Checking if Pos actually has variation with this call
        pos=FixingExtent(pos)

        # Run second burn-in.
        print "Running second burn-in" if verbose else 0
        pos, lp, _ = sampler.run_mcmc(pos,nstep/5)
        pos=FixingExtent(pos)

        # Run main MCMC run:
        #pdb.set_trace()
        print str(KIC)+" running main MCMC" if verbose else 0
        sampler.reset()
        pos, prob, state = sampler.run_mcmc(pos, nstep)

        # Test if MCMC found a better chi^2 region of parameter space:
        mcmc_params = sampler.chain[:,:,:].reshape((-1,len(bestparams)))#sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==sampler.lnprobability.ravel().max())[0][0]]
        mcmc_fit = optimize.leastsq(devfuncup, mcmc_params, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)

        if errfunc(mcmc_fit[0], *fitargs) < errfunc(bestparams, *fitargs):
            bestparams = mcmc_fit[0]
            print "Found a better fit from minimizing." if verbose else 0
        else:
            pass

        lsq_fit = optimize.minimize(errfunc, bestparams, args=fitargs)#, xtol=xtol, ftol=ftol)
        bestparams = lsq_fit['x']

        finalmodel = PlotModel(lccut, mcmc_params,verbose=verbose)

    elif GP:
        import george
        #Running the emcee with Gaussian Processes.
        print "Running Namaste with GP (George)" if verbose else 0

        #Removing 'Flux' term from parameters (GP will fit this)
        params=np.hstack((params[0:4],params[5:]))
        priors=np.vstack((priors[0:4,:],priors[5:,:]))

        if george.__version__[0]!='1':
            print "This script uses George 1.0."
            print "See http://github.com/dfm/george"
            print "Or download with pip using \"pip install git+http://github.com/dfm/george.git@1.0-dev\" "
            return None
        #Running it on 15-Tdur-wide window
        lccut=lc[np.where(abs(lc[:, 0]-Tcen)<Tdur*8.0)[0]]
        print "Training GP on out-of-transit data" if verbose else 0
        gp, res, lnlikfit = TrainGP(lc,Tcen,h[3])

        newmean,newwn,a,tau=res

        #Initialising Monotransit class.
        mono=Monotransit(params)
        newparams=np.hstack((params,newwn,a,tau))
        mono.set_vector(newparams[:6])

        #Setting up george variable
        print "Initialising george fitting model" if verbose else 0
        model=george.GP(np.exp(newparams[-2])*george.kernels.ExpSquaredKernel(np.exp(newparams[-1])),mean=mono,fit_mean=True,white_noise=newparams[-3],fit_white_noise=True)
        model.compute(lccut[:,0],lccut[:,2])
        #Setting up emcee run
        fitargs = (lccut[:,1], priors, model)
        bestparams = model.get_vector()
        initial = np.copy(bestparams)
        initial_fit = optimize.minimize(nll, initial, args=(model, lccut[:,1]), method="L-BFGS-B")#optimize.leastsq(devfuncup, mcmc_params, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
        print "Initial params = "+str(initial) if verbose else 0
        print "Initial minimized = "+str(initial_fit['x']) if verbose else 0
        print "Initial minimized ratio = "+str(initial_fit['x']/initial) if verbose else 0

        if lnprobGP(initial_fit['x'], *fitargs) > lnprobGP(initial, *fitargs) and lnpriorGP(initial_fit['x'],priors)!=-np.inf:
            initial=initial_fit['x']
            print lnprobGP(initial_fit['x'], *fitargs)/lnprobGP(initial, *fitargs)
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
            if ii<4:
                covar[ii,ii] = 0.05*(abs(initial[ii]-np.min(priors[ii]))+abs(initial[ii]+np.max(priors[ii])))
                diag[ii]=0.05*(abs(initial[ii]-priors[ii][0])+abs(initial[ii]-priors[ii][1]))
            else:
                #GP objects dont have priors... Using square root of absolute GP params
                covar[ii,ii] = 0.05*np.sqrt(abs(initial[ii]))*(1.0 if initial[ii]>0 else -1.0)
                diag[ii]= 0.05*np.sqrt(abs(initial[ii]))*(1.0 if initial[ii]>0 else -1.0)


        print "Covariance diagonal = "+str(np.diag(covar)) if verbose else 0
        p0 = np.vstack((initial, np.random.multivariate_normal(initial, (1e-5)*covar, nwalkers-1)))
        #p0 = initial + 1e-6 * np.random.randn(nwalkers, ndim) #Replaced

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobGP, args=fitargs,threads=nthread)


        print("Running first burn-in with "+str(nstep/5)+" iterations...") if verbose else 0
        newp0, _, _ = sampler.run_mcmc(p0, nstep/5)
        newp0=FixingExtent(newp0)
        sampler.reset()

        print("Running second burn-in with "+str(nstep/5)+" iterations...") if verbose else 0
        newp0, _, _ = sampler.run_mcmc(newp0, nstep/5)
        newp0=FixingExtent(newp0)
        PlotModel(lccut, sampler.chain[:,:,:].reshape((-1,len(initial))), GP=True,verbose=verbose)

        print "sampler stds"+str(np.std(sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==sampler.lnprobability.ravel().max())[0][0]],axis=0))
        sampler.reset()

        print("Running production...") if verbose else 0
        pos, prob, state = sampler.run_mcmc(newp0, nstep);
        mcmc_params = sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==sampler.lnprobability.ravel().max())[0][0]]
        newbestparams=np.median(sampler.chain[:,:,:].reshape((-1,len(initial))),axis=0)

        #Running another minimization and taking the new paramteres if theyre better (and within priors)
        lsq_fit = optimize.minimize(nll, newbestparams, args=(model, lccut[:,1], priors), method="L-BFGS-B")#optimize.leastsq(devfuncup, mcmc_params, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
        if lnprobGP(lsq_fit['x'], *fitargs) > lnprobGP(newbestparams, *fitargs) and lnpriorGP(initial_fit['x'],priors)!=-np.inf:
            bestparams = lsq_fit['x']
        else:
            bestparams = newbestparams

        #Plotting and returning the model:
        ymodel=PlotModel(lccut, sampler.chain[:,:,:].reshape((-1,len(initial))), GP=True,verbose=verbose)
        model.set_vector(bestparams)
        finalmodel=ymodel

    #finalmodel = modelmonotransit(bestparams, fitargs[1])
    print str(KIC)+" finished MCMC" if verbose else 0
    return lsq_fit, sampler, finalmodel, StarDat,  lccut

def Pmin(lc, Tcen, Tdur):
    #Calculating minimum period from the lightcurve
    return np.max(abs(Tcen-lc[:, 0]))-Tdur

def FixingExtent(pos):
    #Finding out if the MCMC has converged on a single value anywhere, and fixing this to add artificial variation
    NoExtent=np.where(np.std(pos, axis=0)<1e-4)[0]
    if len(NoExtent)>0:
        for p in NoExtent:
            #Bumping pos back up to variations of 10e-5 if there is NoExtent
            pos[:, p]*=np.random.normal(1.0, 1.0e-5, len(pos[:, p]))
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
    Rs=Rs*695500000 if Rs<5 else Rs
    Ms=Ms*1.96e30 if Ms<5 else Ms
    SMA=(6.67e-11*Ms)/((Vel*Rs/86400.)**2)
    Per=(2*np.pi*SMA)/(Vel*Rs/86400)
    return SMA/1.49e11, Per

def PerToVel(Period, Rs, Ms,dens=0):
    #Check this from paper...
    return ((2*np.pi*6.67e-11*Ms*1.96e30)/(Period*86400*(Rs*695000000)**3))**(1/3.)

def PlotMCMC(params,  EPIC, sampler, lc, GP=False, KepName=''):
    import corner
    if type(sampler)!=np.ndarray:
        samples = sampler.chain[:, -10000:, :].reshape((-1, len(params)))
    else:
        samples=sampler

    #Turning b into absolute...
    samples[:,1]=abs(samples[:,1])
    p.figure(1)

    #Clipping extreme values (top.bottom 0.1 percentiles)
    toclip=np.array([(np.percentile(samples[:,t],99.9)>samples[:,t]) // (samples[:,t]>np.percentile(samples[:,t],0.1)) for t in range(len(params))]).all(axis=0)
    samples=samples[toclip]

    #Earmarking the difference between GP and non
    labs = ["$T_{c}$","$b$","$v$","$Rp/Rs$","$u1$","$u2$","$\sigma_{white}$","$tau$","$a$"] if GP\
        else ["$T_{c}$","$b$","$v$","$Rp/Rs$","$F_0$","$u1$","$u2$"]

    #This plots the corner:
    fig = corner.corner(samples, label=labs, quantiles=[0.16, 0.5, 0.84], plot_datapoints=False)

    #Making sure the lightcuvre plot doesnt overstep the corner
    ndim=np.shape(samples)[1]
    rows=(ndim-1)/2
    cols=(ndim-1)/2

    #Printing Kepler name on plot
    p.subplot(ndim,ndim,ndim+3).axis('off')
    if KepName=='':
        if str(int(EPIC)).zfill(9)[0]=='2':
            KepName='EPIC'+str(EPIC)
        else:
            KepName='KIC'+str(EPIC)
    p.title(KepName, fontsize=22)

    #This plots the model on the same plot as the corner
    ax = p.subplot2grid((ndim,ndim), (0, ndim-cols), rowspan=rows-1, colspan=cols)
    modelfits=PlotModel(lc, samples, scale=1.0, GP=True)

    #plotting residuals beneath:
    ax = p.subplot2grid((ndim,ndim), (rows-1, ndim-cols), rowspan=1, colspan=cols)
    modelfits=PlotModel(lc,  samples, modelfits, residuals=True, GP=True, scale=1.0)

    #Saving as pdf. Will save up to 3 unique files.
    if os.path.exists(Namwd+'/Outputs/Corner_'+str(EPIC)+'_1.pdf'):
        if os.path.exists(Namwd+'/Outputs/Corner_'+str(EPIC)+'_2.pdf'):
            fname=Namwd+'/Outputs/Corner_'+str(EPIC)+'_3.pdf'
        else:
            fname=Namwd+'/Outputs/Corner_'+str(EPIC)+'_2.pdf'
    else:
        fname=Namwd+'/Outputs/Corner_'+str(EPIC)+'_1.pdf'
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
    sigs=np.array([2.2750131948178987, 15.865525393145707, 50.0, 84.13447460685429, 97.7249868051821]) #68,95,99.7% percentile bounds.
    return np.percentile(np.array(Rplans), sigs),  np.percentile(np.array(Ps), sigs), np.percentile(np.array(Smas), sigs), np.percentile(np.array(Krvs), sigs)

def PlotModel(lccut, sampler, models='', residuals=False, scale=1, nx=10000,GP=False,monomodel=0, verbose=True):
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
            monomodel=Monotransit(params[:-3])
        #Setting up GP to model
        model=george.GP(np.exp(params[-2])*george.kernels.ExpSquaredKernel(np.exp(params[-1])),mean=monomodel,fit_mean=True,white_noise=params[-3],fit_white_noise=True)
        model.compute(lccut[:,0],lccut[:,2])
        ypreds,varpreds=model.predict(lccut[:,1], t)
        stds=np.sqrt(np.diag(varpreds))
        #Turning into y-2s,y-1s,y,y+1s,y+2s format.
        modelfits=np.column_stack((ypreds-stds*2,ypreds-stds,ypreds,ypreds+stds,ypreds+stds*2))
            #,np.percentile(modelouts,[6.76676416,   18.39397206,   50., 81.60602794,93.23323584],axis=1)
    newmodelfits=np.copy(modelfits)

    if residuals:
        #Subtracting bestfit model from both flux and model to give residuals
        newmodelfits=modelfits-np.tile(modelfits[:,2], (5, 1)).swapaxes(0, 1) #subtracting median fit
        lccut[:,1]-=modelfits[:, 2][(np.round((lccut[:,0]-lccut[0,0])/0.020431700249901041)).astype(int)]
    #p.xlim([t[np.where(redfits[:,2]==np.min(redfits[:,2]))]-1.6, t[np.where(redfits[:,2]==np.min(redfits[:,2]))]+1.6])

    if verbose:
        #Plotting data
        p.errorbar(lccut[:, 0], lccut[:,1], yerr=lccut[:, 2], fmt='.',color='#999999')
        p.plot(lccut[:, 0], lccut[:,1], '.',color='#333399')

        print np.shape(np.hstack((newmodelfits[:,2]-(newmodelfits[:,2]-newmodelfits[:,1]),(newmodelfits[:,2]+(newmodelfits[:,3]-newmodelfits[:,2]))[::-1])))
        print np.shape(np.hstack((t,t[::-1])))
        #Plotting 1-sigma error region and models
        p.fill(np.hstack((t,t[::-1])),np.hstack((newmodelfits[:,2]-(newmodelfits[:,2]-newmodelfits[:,1]),(newmodelfits[:,2]+(newmodelfits[:,3]-newmodelfits[:,2]))[::-1])),'#33BBFF', linewidth=0,label='$1-\sigma$ region ('+str(scale*100)+'% scaled)',alpha=0.5)
        p.plot(t,newmodelfits[:,2],'-',color='#003333',linewidth=2.0,label='Median model fit')
        p.pause(5)

    if not residuals:
        #Putting title on upper (non-residuals) graph
        p.title('Best fit model')
        p.legend(loc=3)
    return modelfits

'''
def StartSingleRun(kic):
    #This script uses the KOI data table and associated parameters to start a 'Namaste' run.
    #Finding the transit is still required

    kic=int(kic)
    AllKicData=KicData(kic)
    lccut, TransData=FindTransitManual(kic)
    savedlcname=Namwd+'KeplerFits/KOI'+str(AllKicData['kepoi_name'].values[0])+'/'+'Transit_LC_at_'+str(TransData[0])+'_lc.txt'
    np.savetxt(savedlcname, lccut)
    StarDat=[AllKicData['koi_steff'].values[0], AllKicData['koi_steff_err1'].values[0],  AllKicData['koi_steff_err2'].values[0], AllKicData['koi_srad'].values[0], AllKicData['koi_srad_err1'].values[0],  AllKicData['koi_srad_err2'].values[0], AllKicData['koi_smass'].values[0], AllKicData['koi_smass_err1'].values[0],  AllKicData['koi_smass_err2'].values[0]]
    #Currently just fudging it to run the other python script from the command line
    print 'python McmcSingles.py \''+str(kic)+', '+savedlcname+', '+str(TransData).replace(' ','')+' , 28000, '+str(StarDat).replace(' ','')+'\'' #['+TransData[0]+', '+TransData[1]+', '+TransData[2]+']
'''

def SaveOutputs(KIC, lsq_fit, finalmodel, lc, samplercut, Rps, Ps, As, StarDat, sigs=np.array([2.2750131948178987, 15.865525393145707, 50.0, 84.13447460685429, 97.7249868051821])):
    #Creating Output directory
    if not os.path.exists(Namwd+'/Outputs/'):
        os.system('mkdir Namwd/Outputs/')

    print 'Saving '+str(KIC)+' outputs to file'
    np.save(Namwd+'/Outputs/'+str(KIC)+"lsq_fit", lsq_fit)
    np.save(Namwd+'/Outputs/'+str(KIC)+"finalmodel", finalmodel)
    np.save(Namwd+'/Outputs/'+str(KIC)+"lightcurve", lc)
    np.save(Namwd+'/Outputs/'+str(KIC)+"sampler", samplercut)
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
    #Currently only works for Kepler and K2 Lightcurves

    '''

    #NEW IMPORT STYLE:
    from argparse import ArgumentParser

    parser = ArgumentParser()
    #Required args:
    parser.add_argument("KIC",type=int,help="Kepler (KIC) or K2 (EPIC) id")
    parser.add_argument("lcurve",type=str, help="File location of lightcurve file. Type \"get\" to download lightcurve")

    #Optional args:
    parser.add_argument("-n", "--nstep", dest="nstep", default=5000,
                      help="Number of MCMC steps. Default 5000")
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
        print "Error in temp, rad or mass. Assuming not known" if args.verbose else 0
        temp,rad,mass=[0,0],[0,0],[0,0]
    StarDat=[temp[0],temp[1],temp[1],rad[0],rad[1],rad[1],mass[0],mass[1],mass[1]]
    if args.lcurve=="get":
        args.lcurve=getKeplerLC(args.KIC,getlc=False)
    if 0 in StarDat:
        if StarDat[0]!=0:
            #There's a zero somewhere, but not for temp!
            temp[1]=150 if temp[1]<150 else temp[1]
            rad=RfromT(temp[0],temp[1])
            mass=MfromR(rad[0],rad[1])
            StarDat=[temp[0],temp[1],temp[1],rad[0],rad[1],rad[1],mass[0],mass[1],mass[1]]
        else:
            #Get stardat from kic/fits file
            print "Getting Star Data from Kepler fits file and photometric colours" if args.verbose else 0
            StarDat=getStarDat(args.KIC,args.lcurve)

    #Running Fit!
    import datetime
    tstart=datetime.datetime.now()
    lsq_fit, sampler, finalmodel, StarDat, lc = Run(args.KIC, args.lcurve, np.array((args.Tcen,args.Tdur,args.depth)).astype(float), float(args.nstep), StarDat,GP=args.GP,verbose=args.verbose)
    print "Run took: "+str(datetime.datetime.now()-tstart) if args.verbose else 0

    #Taking only last 10000 if more than 10000 samples
    cut=-1*np.shape(sampler.chain)[1] if np.shape(sampler.chain)[1]<10000 else -10000
    samplercut=sampler.chain[:,cut:,:].reshape((-1,len(lsq_fit['x'])))

    #Doing period calculations from velocity distribution
    Rps, Ps, As,  Krvs = GetVel(samplercut, StarDat, 5000)
    print str(args.KIC)+" - Period of "+str(Ps[3])+" +"+str(Ps[4]-Ps[3])+"/ -"+str(Ps[3]-Ps[2]) if args.verbose else 0
    print str(args.KIC)+" - Semi Major Axis of "+str(As[3])+" +"+str(As[4]-As[3])+"/ -"+str(As[3]-As[2]) if args.verbose else 0
    print str(args.KIC)+" - Radius of "+str(Rps[3])+" +"+str(Rps[4]-Rps[3])+"/ -"+str(Rps[3]-Rps[2]) if args.verbose else 0

    #Saving outputs to file
    SaveOutputs(args.KIC, lsq_fit, finalmodel, lc, samplercut, Rps, Ps, As, StarDat)

    #Plotting mcmc
    fig=PlotMCMC(lsq_fit['x'],  args.KIC, sampler, lc)
