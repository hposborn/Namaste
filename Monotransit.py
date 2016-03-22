import numpy as np
import pandas as pd
import glob
from os import path
import csv
import Crossfield_transit as transit
import george
from george import kernels

#ADAPTED FROM IAN CROSSFIELD'S PHASECURVES:
def errfunc(*arg,**kw):
    #weighted residuals squared and summed - ChiSq
    return (devfuncup(*arg, **kw)**2).sum()

def lnprobfunc(*arg, **kw):
    #residuals multiplied by -0.5 - effectives lnlikelihood
    return abs(-0.5 * errfunc(*arg, **kw))

def lnprobfunc2(*arg, **kw):
    return abs(lnprobfunc(*arg, **kw))

def devfuncup(params,*arg, **kw):
    #args=modelmonotransit, lccut[:, 0], lccut[:, 1], weights, uniformprior
    uniformprior = arg[-1]

    #Manually switching u1 and u2 from uniform priors to gauss priors:
    gaussprior=[[0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [0, np.inf], [uniformprior[5][0], uniformprior[5][1]], [uniformprior[6][0], uniformprior[6][1]]]
    uniformprior[-2]=[0,1]
    uniformprior[-1]=[0, 1]
    depvar = arg[2]
    weights = arg[3]
    functions = arg[0]
    #arg[2:len(arg)-2]
    model = functions(params,arg[1])
    
    residuals = np.sqrt(weights)*(model-depvar)
    # Compute 1D and N-D gaussian, and uniform, prior penalties:
    additionalChisq = 0.
    #Setting Max Velocity from p
    additionalChisq += np.sum([((param0 - gprior[0])/gprior[1])**2 for param0, gprior in zip(params, gaussprior)])
    #Transit probability linearly proportional to velocity. Multiplying additional ChiSq by the velocity. This will translate into residuals 2* larger for velocities 2* higher.
    additionalChisq *= params[2]**2
    
    for param0, uprior in zip(params, uniformprior):
        if (param0 < uprior[0]) or (param0 > uprior[1]):
            residuals *= 1e9

    # Scale up the residuals so as to impose priors in chi-squared
    # space:
    thisChisq = np.sum(weights * (model - depvar)**2)
    if thisChisq>0:
        scaleFactor = 1. + additionalChisq / thisChisq
        residuals *= np.sqrt(scaleFactor)
#    p.figure(2)
#    p.plot(model,'-')
#    p.plot(depvar, '.')
#    p.title('v='+str(params[2])+' ChiSq='+str(thisChisq))
#    p.pause(0.1)
    if np.isnan(residuals).sum()>1:
        print 'NaNs'
    return residuals
    

def modelmonotransit(params, t):
    """Model a transit light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN period.
    :INPUTS:
      params -- (5+N)-sequence with the following:
        the time of conjunction for each individual transit (Tc),
        the impact parameter (b = a cos i/Rstar)
        vel -- scalar. planetary perpendicular velocity (in units of stellar radii per day)
        planet-to-star radius ratio (Rp/Rstar), 
        stellar flux (F0),
        the limb-darkening parameters u1 and u2:
          EITHER:
            gamma1,  gamma2  -- quadratic limb-darkening coefficients
          OR:
            c1, c2, c3, c4 -- nonlinear limb-darkening coefficients
          OR:
            Nothing at all (i.e., only 5 parameters).
      func -- function to fit to data, e.g. transit.occultquad
      t -- numpy array.  Time of observations.
    """
    #import Ian.transit
    func=transit.occultquad
    #Getting fine cadence (cad/100). Integrating later:
    cad=np.median(np.diff(t))
    finetime=np.empty(0)
    for i in range(len(t)):
        finetime=np.hstack((finetime, np.arange(t[i]-0.495*cad, t[i]+0.505*cad, cad/100.0)))
    z = v2z(params[0], params[2], params[1], finetime) 
    #z[abs(((t - params[0] + params[1]*.25)/per % 1) - 0.5) < 0.43] = 10.
    #pdb.set_trace()
    if len(params)>5:
        model = params[4] * func(z, params[3], params[5::])
    try:  # Limb-darkened
        model = params[4] * func(z, params[3], params[5::])
    except:  # Uniform-disk
        model = params[4] * (1. - func(z, params[3]))
    return np.average(np.resize(model, (len(t), 100)), axis=1)
'''
def lnprobfuncGP(*arg, **kw):

    params = np.array(arg[0], copy=False)
    gp=george.GP(params[]*kernels.Matern32Kernel(params[]))
    #time and weights from args
    gp.compute(arg[1],arg[3]**(-0.5))
    #residuals multiplied by -0.5 - effectives lnlikelihood
    return gp.lnlikelihood(y - modelmonotransitGP())
        abs(-0.5 * (devfuncup(*arg, **kw)**2).sum())


def lnlikeGP(params,t,y,yerr,wn,resid=0,gp=0):
    #Log Likelihood. Last two parameters are now alpha and tau - used to fit background flux.
    #params: tcen,b,v,p0,u1,u1,a,tau
    #Removed "Flux" parameter.
    #Stolen from "George" example script and bastardised for this purpose
    #print arg
    a, tau = np.exp(params[-2:])
    if wn!=0:
        gp = george.GP(a * kernels.ExpSquaredKernel(tau),mean=np.median(y),fit_mean=True, white_noise=np.log(wn), fit_white_noise=True)
    else:
        gp = george.GP(a * kernels.ExpSquaredKernel(tau),mean=np.median(y),fit_mean=True)
    gp.compute(t, yerr)
    if resid==0:
        return gp.lnlikelihood(y - modelmonotransitGP(params, t))
    else:
        return y - modelmonotransitGP(params, t)


def lnpriorGP(params, priors):
    #Working out priors from uniform (all), gaussian (LD coeffs) and linear (velocity)
    #gps=priors[-4:-2]
    #Doing gaussian priors for LD params
    addOut = -((params[-4] - priors[-4][0])/priors[-4][1])**2-((params[-3] - priors[-3][0])/priors[-3][1])**2
    #Prior from velocity:
    addOut *= params[2]**2
    for param0, uprior in zip(params, unipriors):
        if (param0 < uprior[0]) or (param0 > uprior[1]):
            addOut=-1e25
    return addOut

def lnprobGP(params,t,y,yerr,priors):
    lp = lnpriorGP(params, priors)
    return lp + lnlikeGP(params,t,y,yerr) if np.isfinite(lp) else -1e25
'''

def modelmonotransitGP(params, t):
    #Removes Flux usage to give out-of-transit flux changes over to GP
    """Model a transit light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN period.
    :INPUTS:
      params -- (5+N)-sequence with the following:
        the time of conjunction for each individual transit (Tc),
        the impact parameter (b = a cos i/Rstar)
        vel -- scalar. planetary perpendicular velocity (in units of stellar radii per day)
        planet-to-star radius ratio (Rp/Rstar),
        stellar flux (F0),
        the limb-darkening parameters u1 and u2:
          EITHER:
            gamma1,  gamma2  -- quadratic limb-darkening coefficients
          OR:
            c1, c2, c3, c4 -- nonlinear limb-darkening coefficients
          OR:
            Nothing at all (i.e., only 5 parameters).
      func -- function to fit to data, e.g. transit.occultquad
      t -- numpy array.  Time of observations.
    """
    #import Ian.transit
    func=transit.occultquad
    #Getting fine cadence (cad/100). Integrating later:
    cad=np.median(np.diff(t))
    fineind=20.0
    finetime=np.empty(0)
    for i in range(len(t)):
        finetime=np.hstack((finetime, np.arange(t[i]-1/(2*fineind)*cad, t[i]+1/(2*fineind)*cad, cad/fineind)))
    z = v2z(params[0], params[2], params[1], finetime)
    #z[abs(((t - params[0] + params[1]*.25)/per % 1) - 0.5) < 0.43] = 10.
    cad=np.median(np.diff(t))
    finetime=np.arange(t[0]-0.5*cad, cad/fineind)
    #Removed the flux component below. Can be done by GP
    model = func(z, params[3], params[5:7])
    return np.average(np.resize(model, (len(t), fineind)), axis=1)


def v2z(tcen,  vel, b, times):
    """Convert HJD (time) to transit crossing parameter z.
    :INPUTS:
        tcen --  scalar. transit ephemeris
        vel --  scalar. planetary perpendicular velocity (in units of stellar radii per day)
        b --impact paramter,
        times -- scalar or array of times, typically heliocentric or
               barycentric julian date.
"""
    return np.sqrt(b**2+(vel*(times - tcen))**2)



from george import ModelingMixin

class Monotransit(ModelingMixin):
    #George-style model class:

    def __init__(self,params):
        self.parameter_names = ['tcen','b','vel','p0','Ld0','LD1']
        self.parameters = params
        #[n for i, n in enumerate(self.parameter_names) if self.unfrozen[i]]
        self.unfrozen = np.ones_like(self.parameters, dtype=bool)

    def __len__(self):
        """
        Return the number of (un-frozen) parameters exposed by this
        object.
        """
        return np.sum(self.unfrozen)

    def get_parameter_names(self):
        """
        Return a list with the names of the parameters.

        """
        return ['tcen','b','vel','p0','Ld0','LD1']

    def get_vector(self):
        """
        Returns a numpy array with the current settings of all the
        non-frozen parameters.
        """
        return self.parameters[self.unfrozen]

    def set_vector(self, vector):
        """
        Update the vector with a numpy array of the same shape and order
        as ``get_vector``.
        """
        self.parameters = vector

    def get_value(self, t):
        tcen,b,vel,p0,LD0,LD1=self.get_vector()
        func=transit.occultquad
        #Getting fine cadence (cad/100). Integrating later:
        cad=np.median(np.diff(t))
        fineind=20.0
        finetime=np.empty(0)
        for i in range(len(t)):
            finetime=np.hstack((finetime, np.linspace(t[i]-(1-1/fineind)*(cad/2.), t[i]+(1-1/fineind)*(cad/2.), fineind) ))
        finetime=np.sort(finetime)
        z = np.sqrt(b**2+(vel*(finetime - tcen))**2)
        #Removed the flux component below. Can be done by GP
        model = func(z, p0, np.array((LD0,LD1)))
        return np.average(np.resize(model, (len(t), fineind)), axis=1)

    def get_gradient(self, t):
        """
        Get the gradient of ``get_value`` with respect to the parameter
        vector returned by ``get_vector``.
        # Ok, I dont quite understand this. Seems to be the differential of the function wrt to the paramters (in the eg case, x1 and x2).
        # Hmmm. In the eg case, y=f(x1,x2). Grad=dy/dx (or dfdx)
        # In my case, y=f(t). Grad=dfdt
        # So grad is the rate of change of f at point t...
        # Lets just estimate the gradient by averaging the fine cadence trick we use anyway...
        NOPE - there needs to be one gradient per parameter (not one per t input) ...
        Ok, I have no idea how to integrate the transit model with respect to each parameter (so I'm not going to bother).
        """
        print "Actually using the gradient. Shit."
        return np.ones(self.__len__())[self.unfrozen]

    def freeze_parameter(self, parameter_name):
        """
        Fix the value of some parameter by name at its current value. This
        parameter should no longer be returned by ``get_vector`` or
        ``get_gradient``.
        """
        self.unfrozen[self.parameter_names.index(parameter_name)] = False

    def thaw_parameter(self, parameter_name):
        """
        The opposite of ``freeze_parameter``.
        """
        self.unfrozen[self.parameter_names.index(parameter_name)] = True

def nll(p,gp,y):
    #inverted LnLikelihood function for GP training
    gp.set_vector(p)
    ll = gp.lnlikelihood(y, quiet=True)
    return -ll if np.isfinite(ll) else 1e25


def grad_nll(p,gp,y):
    #Gradient of the objective function for TrainGP
    gp.set_vector(p)
    return -gp.grad_lnlikelihood(y, quiet=True)

def TrainGP(lc,Tcen,Vmag):
    '''
    #This optimizes the gaussian process on out-of-transit data. This is then held with a gaussian prior during modelling
    #Args:
        #lc: lightcurve
        #Tcen: Central transit time

    #Returns:
    #    a, tau: Squared Exponential kernel hyperparameters
    '''
    import scipy.optimize as op
    #Cutting lc
    updown= -1 if np.max(lc[:,0]-Tcen)>abs(np.min(lc[:,0]-Tcen)) else 1
    lcs=lc[np.where((lc[:,0]-Tcen)*updown>0.4)[0],:]

    # initialising GP parameters
    WNmag14 = 2.42e-4 #White Noise at 14th magnitude for Kepler.
    wn=WNmag14/np.sqrt(10**((14-Vmag)/2.514)) #Calculating White Noise from sqrt flux (eg assuming poisson)
    a,tau=1.0,0.3

    #Setting up GP
    kernel=a*(george.kernels.ExpSquaredKernel(tau))
    gp = george.GP(kernel, mean=np.mean(lcs[:,1]), fit_mean=True, white_noise=np.log(wn**2), fit_white_noise=True)
    gp.compute(lcs[:,0],lcs[:,2])

    # Run the optimization routine.
    p0 = gp.get_vector()
    results = op.minimize(nll, p0,args=(gp,lcs[:,1]), jac=grad_nll, method="L-BFGS-B")

    # Update the kernel and print the final log-likelihood.
    gp.set_vector(results.x)

    return gp, results.x, gp.lnlikelihood(lcs[:,1])

def lnpriorGP(params,priors):
    c,addOut=0,0
    for par0, pri0 in zip(params, priors):
        if c in [4,5]:
            #The better a fit, the higher the value. Must be added
            #From Kipping: q_1 = (u_1+u_2)^2 and q_2 = 0.5u_1/(u_1+u_2)
            #Turning gaussian priors in q1/q2 into gaussian priors in u1/u2...
            # according to paper:
            # u_1 = 2*sqrt(q_1)*q_2
            # u_2 = sqrt(q1)*(1-2*q2)
            addOut+=scipy.stats.norm(pri0[0],pri0[1]).pdf(par0)
        elif c>=6:
            #Gamma priors on wn, a and tau as shown in Evans et al 2015
            addOut+=np.exp(-100*np.exp(par0))
        elif c<4 and [(par0 < pri0[0]) or (par0 > pri0[1])]:
            #To maximise lnlikelihood, parameters outside priors need to be subtracted.
            addOut=-np.inf
        c+=1
    #To prioritise shorter orbits (eg higher vels), the higher the velprior, the better the fit.
    addOut += params[2]**2
    return addOut


def lnprobGP(p, y, priors, model):
    model.set_vector(p)
    lp=lnpriorGP(p,priors)
    ret=lp+model.lnlikelihood(y, quiet=True)
    #print '| '+str(ret)+' | '+str(p)
    return ret
