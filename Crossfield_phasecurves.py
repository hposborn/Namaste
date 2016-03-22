"""Functions for fitting phase curves.

Use of 'errfunc' is encouraged.

:REQUIREMENTS:
  :doc:`transit`
"""
# 2009-12-15 13:26 IJC: Created
# 2011-04-21 16:13 IJMC: Added tsin function and nsin function

#try:
#    import psyco
#    psyco.full()
#except ImportError:
#    print 'Psyco not installed, the program will just run slower'



from numpy import *
import numpy as np

def errfunc(*arg,**kw):
    """Generic function to give the chi-squared error on a generic function:

    INPUTS:
       (fitparams, function, arg1, arg2, ... , depvar, weights)

    OPTIONAL INPUT:
       retres : bool 
              If set to True, return a vector of residuals rather than
              the sum of this vector (for :func:`scipy.optimize.leastsq`).
       sumconstraint : tuple -- ([indices], val, tol)
             Add a value to chisq of the form:  (abs(fitparams[indices].sum()-val)/tol)**2
       prodconstraint : tuple -- ([indices], val, tol)
             Add a value to chisq of the form:  (((1.+fitparams[indices]).prod()-val)/tol)**2
             


    EXAMPLE: 
      ::

       from numpy import *
       import phasecurves
       def sinfunc(period, x): return sin(2*pi*x/period)
       snr = 10
       x = arange(30.)
       y = sinfunc(9.5, x) + randn(len(x))/snr
       guess = 8.
       period = optimize.fmin(phasecurves.errfunc,guess,args=(sinfunc,x, y, ones(x.shape)*snr**2))

    :SEE ALSO:
       :func:`resfunc`
    """

    # 2009-12-15 13:39 IJC: Created
    # 2010-03-29 13:29 IJC: Switched to helper function.
    # 2010-07-01 10:40 IJC: Added sumconstraint feature
    # 2011-01-19 22:06 IJC: Added "retres" option to return a vector
    #        of residuals, for use with scipy.optimize.leastsq
    # 2011-05-16 08:59 IJC: Chisq involves a squaring, not a fourth power...
    # 2012-05-01 01:54 IJMC: Now much simplified: just call resfunc!

    # phasecurves.errfunc(): 
    return (devfunc(*arg, **kw)**2).sum()

def devfunc(*arg, **kw):
    """Generic function to give the weighted residuals on a function or functions:

    :INPUTS:
       (fitparams, function, arg1, arg2, ... , depvar, weights)

      OR:
       
       (fitparams, function, arg1, arg2, ... , depvar, weights, kw)

      OR:
       
       (allparams, (args1, args2, ..), npars=(npar1, npar2, ...))

       where allparams is an array concatenation of each functions
       input parameters.

      If the last argument is of type dict, it is assumed to be a set
      of keyword arguments: this will be added to resfunc's direct
      keyword arguments, and will then be passed to the fitting
      function **kw.  This is necessary for use with various fitting
      and sampling routines (e.g., kapteyn.kmpfit and emcee.sampler)
      which do not allow keyword arguments to be explicitly passed.
      So, we cheat!  Note that any keyword arguments passed in this
      way will overwrite keywords of the same names passed in the
      standard, Pythonic, way.


    :OPTIONAL INPUTS:
      jointpars -- list of 2-tuples.  
                   For use with multi-function calling (w/npars
                   keyword).  Setting jointpars=[(0,10), (0,20)] will
                   always set params[10]=params[0] and
                   params[20]=params[0].

      gaussprior -- list of 2-tuples (or None values), same length as "fitparams."
                   The i^th tuple (x_i, s_i) imposes a Gaussian prior
                   on the i^th parameter p_i by adding ((p_i -
                   x_i)/s_i)^2 to the total chi-squared.  Here in
                   :func:`devfunc`, we _scale_ the error-weighted
                   deviates such that the resulting chi-squared will
                   increase by the desired amount.

      uniformprior -- list of 2-tuples (or 'None's), same length as "fitparams."
                   The i^th tuple (lo_i, hi_i) imposes a uniform prior
                   on the i^th parameter p_i by requiring that it lie
                   within the specified "high" and "low" limits.  We
                   do this (imprecisely) by multiplying the resulting
                   deviates by 1e9 for each parameter outside its
                   limits.

      ngaussprior -- list of 3-tuples of Numpy arrays.
                   Each tuple (j_ind, mu, cov) imposes a multinormal
                   Gaussian prior on the parameters indexed by
                   'j_ind', with mean values specified by 'mu' and
                   covariance matrix 'cov.' This is the N-dimensional
                   generalization of the 'gaussprior' option described
                   above. Here in :func:`devfunc`, we _scale_ the
                   error-weighted deviates such that the resulting
                   chi-squared will increase by the desired amount.

                   For example, if parameters 0 and 3 are to be
                   jointly constrained (w/unity means), set: 
                     jparams = np.array([0, 3])
                     mu = np.array([1, 1])
                     cov = np.array([[1, .9], [9., 1]])
                     ngaussprior=[[jparams, mu, cov]]  # Double brackets are key!


    EXAMPLE: 
      ::

       from numpy import *
       import phasecurves
       def sinfunc(period, x): return sin(2*pi*x/period)
       snr = 10
       x = arange(30.)
       y = sinfunc(9.5, x) + randn(len(x))/snr
       guess = 8.
       period = optimize.fmin(phasecurves.errfunc,guess,args=(sinfunc,x, y, ones(x.shape)*snr**2))
    """
    # 2009-12-15 13:39 IJC: Created
    # 2010-11-23 16:25 IJMC: Added 'testfinite' flag keyword
    # 2011-06-06 10:52 IJMC: Added 'useindepvar' flag keyword
    # 2011-06-24 15:03 IJMC: Added multi-function (npars) and
    #                        jointpars support.
    # 2011-06-27 14:34 IJMC: Flag-catching for multifunc calling
    # 2012-03-23 18:32 IJMC: testfinite and useindepvar are now FALSE
    #                        by default.
    # 2012-05-01 01:04 IJMC: Adding surreptious keywords, and GAUSSIAN
    #                        PRIOR capability.
    # 2012-05-08 16:31 IJMC: Added NGAUSSIAN option.
    # 2012-10-16 09:07 IJMC: Added 'uniformprior' option.
    # 2013-02-26 11:19 IJMC: Reworked return & concatenation in 'npars' cases.
    # 2013-03-08 12:54 IJMC: Added check for chisq=0 in penalty-factor cases.

    import pdb

    params = np.array(arg[0], copy=False)

    if isinstance(arg[-1], dict): 
        print 'Dictionary found'
        # Surreptiously setting keyword arguments:
        kw2 = arg[-1]
        kw.update(kw2)
        arg = arg[0:-1]
    else:
        pass


    if len(arg)==2:
        residuals = devfunc(params, *arg[1], **kw)

    else:
        if kw.has_key('testfinite'):
            testfinite =  kw['testfinite']
        else:
            testfinite = False
        if not kw.has_key('useindepvar'):
            kw['useindepvar'] = False

        if kw.has_key('gaussprior'):
            # If any priors are None, redefine them:
            temp_gaussprior =  kw['gaussprior']
            gaussprior = []
            for pair in temp_gaussprior:
                if pair is None:
                    gaussprior.append([0, np.inf])
                else:
                    gaussprior.append(pair)
        else:
            gaussprior = None

        if kw.has_key('uniformprior'):
            # If any priors are None, redefine them:
            temp_uniformprior =  kw['uniformprior']
            uniformprior = []
            for pair in temp_uniformprior:
                if pair is None:
                    uniformprior.append([-np.inf, np.inf])
                else:
                    uniformprior.append(pair)
        else:
            uniformprior = None

        if kw.has_key('ngaussprior') and kw['ngaussprior'] is not None:
            # If any priors are None, redefine them:
            temp_ngaussprior =  kw['ngaussprior']
            ngaussprior = []
            for triplet in temp_ngaussprior:
                if len(triplet)==3:
                    ngaussprior.append(triplet)
        else:
            ngaussprior = None


        #print "len(arg)>>", len(arg),

        if kw.has_key('npars'):
            npars = kw['npars']
            ret = np.array([])
            # Excise "npars" kw for recursive calling:
            lower_kw = kw.copy()
            junk = lower_kw.pop('npars')

            # Keep fixed pairs of joint parameters:
            if kw.has_key('jointpars'):
                jointpars = kw['jointpars']
                for jointpar in jointpars:
                    params[jointpar[1]] = params[jointpar[0]]

            for ii in range(len(npars)):
                i0 = sum(npars[0:ii])
                i1 = i0 + npars[ii]
                these_params = arg[0][i0:i1]
                #ret.append(devfunc(these_params, *arg[1][ii], **lower_kw))
                ret = np.concatenate((ret, devfunc(these_params, *arg[ii+1], **lower_kw).ravel()))

            return ret

        else: # Single function-fitting
            depvar = arg[-2]
            weights = arg[-1]

            if not kw['useindepvar']:
                functions = arg[1]
                helperargs = arg[2:len(arg)-2]
            else:
                functions = arg[1]
                helperargs = arg[2:len(arg)-3]
                indepvar = arg[-3]



        if testfinite:
            finiteind = isfinite(indepvar) * isfinite(depvar) * isfinite(weights)
            indepvar = indepvar[finiteind]
            depvar = depvar[finiteind]
            weights = weights[finiteind]


        if not kw['useindepvar'] or arg[1].__name__=='multifunc' or arg[1].__name__=='sumfunc':
            model = functions(*((params,)+helperargs))
        else:  # (i.e., if useindepvar is True!)
            model = functions(*((params,)+helperargs + (indepvar,)))

        # Compute the weighted residuals:
        residuals = np.sqrt(weights)*(model-depvar)


        # Compute 1D and N-D gaussian, and uniform, prior penalties:
        additionalChisq = 0.
        if gaussprior is not None:
            additionalChisq += np.sum([((param0 - gprior[0])/gprior[1])**2 for \
                                   param0, gprior in zip(params, gaussprior)])

        if ngaussprior is not None:
            for ind, mu, cov in ngaussprior:
                dvec = params[ind] - mu
                additionalChisq += \
                    np.dot(dvec.transpose(), np.dot(np.linalg.inv(cov), dvec))

        if uniformprior is not None:
            for param0, uprior in zip(params, uniformprior):
                if (param0 < uprior[0]) or (param0 > uprior[1]):
                    residuals *= 1e9

        # Scale up the residuals so as to impose priors in chi-squared
        # space:
        thisChisq = np.sum(weights * (model - depvar)**2)
        if thisChisq>0:
            scaleFactor = 1. + additionalChisq / thisChisq
            residuals *= np.sqrt(scaleFactor)

    return residuals

def lnprobfunc(*arg, **kw):
    """Return natural logarithm of posterior probability: i.e., -chisq/2.

    Inputs are the same as for :func:`errfunc`.

    :SEE ALSO:
      :func:`gaussianprocess.negLogLikelihood`
    """
    # 2012-03-23 18:17 IJMC: Created for use with :doc:`emcee` module.

    return -0.5 * errfunc(*arg, **kw)
