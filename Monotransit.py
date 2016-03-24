import numpy as np
import pandas as pd
import glob
from os import path
import csv
import Crossfield_transit as transit
import george
from george import kernels
from scipy import stats

#ADAPTED FROM IAN CROSSFIELD'S PHASECURVES:
def errfunc(*arg,**kw):
    #weighted residuals squared and summed - ChiSq
    return (devfuncup(*arg, **kw)**2).sum()

def lnprobfunc(*arg, **kw):
    #residuals multiplied by -0.5 - effectives lnlikelihood
    return abs(-0.5 * errfunc(*arg, **kw))

def lnprobfunc2(*arg, **kw):
    return abs(lnprobfunc(*arg, **kw))

def devfuncup(params,func,t,y,weights,priors):
    #args=modelmonotransit, lccut[:, 0], lccut[:, 1], weights, uniformprior
    model = func(params,t)
    
    residuals = np.sqrt(weights)*(model-y)
    # Compute 1D and N-D gaussian, and uniform, prior penalties:

    #Transit probability linearly proportional to velocity. Multiplying additional ChiSq by the velocity. This will translate into residuals 2* larger for velocities 2* higher.
    additionalChisq = params[2]**2
    
    for p in range(len(params)):
        param0=params[p]
        uprior=priors[p,:2]
        if (param0 < uprior[0]) or (param0 > uprior[1]):
            additionalChisq += 1e25
        if p in [5,6]:
            gprior=priors[p,2:]
            #Gaussian Priors for LDs:
            additionalChisq += ((param0 - gprior[0])/gprior[1])**2
    # Scale up the residuals so as to impose priors in chi-squared
    # space:
    thisChisq = np.sum(weights * (model - y)**2)
    if thisChisq>0:
        scaleFactor = 1. + additionalChisq / thisChisq
        residuals *= np.sqrt(scaleFactor)
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
        gamma1,  gamma2  -- quadratic limb-darkening coefficients

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
        print "Actually using the gradient. Shit, I havent actually calculated that yet"
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

def nll(p,gp,y,prior=[]):
    #inverted LnLikelihood function for GP training
    gp.set_vector(p)
    if prior!=[]:
        prob=lnprobGP(p,y,prior,gp)
        return -prob if np.isfinite(prob) else 1e25
    else:
        ll = gp.lnlikelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

def grad_nll(p,gp,y,prior=[]):
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
    '''
    Args:
        params: 9-parameter array
        priors: 9x4 array of uniform and gaussian priors

    Returns:

    '''
    c,addOut=0,0
    for n in range(len(params)):
        par0=params[n]
        if c in [4,5]:
            pri0=priors[n,:2]
            #The better a fit, the higher the value. Must be added
            #From Kipping: q_1 = (u_1+u_2)^2 and q_2 = 0.5u_1/(u_1+u_2)
            #Turning gaussian priors in q1/q2 into gaussian priors in u1/u2...
            # according to paper:
            # u_1 = 2*sqrt(q_1)*q_2
            # u_2 = sqrt(q1)*(1-2*q2)
            addOut+=stats.norm(pri0[0],pri0[1]).pdf(par0)
        elif c>=6:
            #Gamma priors on wn, a and tau as shown in Evans et al 2015
            addOut+=np.exp(-100*np.exp(par0))
        if c<6:
            pri0=priors[n,:2]
            if par0 < np.min(pri0) or par0 > np.max(pri0):
                #To maximise lnlikelihood, parameters outside priors need to be subtracted.
                addOut=-np.inf
                #print "inf from "+str(c)+" par0: "+str(par0)+" pri: "+str(pri0)
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
