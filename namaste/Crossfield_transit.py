"""
-------------------------------------------------------
The Mandel & Agol (2002) transit light curve equations.
-------------------------------------------------------

:FUNCTIONS:
   :func:`occultuniform` -- uniform-disk transit light curve

   :func:`occultquad` -- quadratic limb-darkening

   :func:`occultnonlin` -- full (4-parameter) nonlinear limb-darkening

   :func:`occultnonlin_small` -- small-planet approximation with full
                                 nonlinear limb-darkening.
p
   :func:`t2z` -- convert dates to transiting z-parameter for circular
                  orbits.


:REQUIREMENTS:
   `numpy <http://www.numpy.org/>`_

   `scipy.special <http://www.scipy.org/>`_


:NOTES:
    Certain values of p (<0.09, >0.5) cause some routines to hang;
    your mileage may vary.  If you find out why, please let me know!

    Cursory testing suggests that the Python routines contained within
     are slower than the corresponding IDL code by a factor of 5-10.

    For :func:`occultquad` I relied heavily on the IDL code of E. Agol
    and J. Eastman.

    Function :func:`appellf1` comes from the mpmath compilation, and
    is adopted (with modification) for use herein in compliance with
    its BSD license (see function documentation for more details).

:REFERENCE:
    The main reference is that seminal work by `Mandel and Agol (2002)
    <http://adsabs.harvard.edu/abs/2002ApJ...580L.171M>`_.

:LICENSE:
    Created by `Ian Crossfield <http://www.astro.ucla.edu/~ianc/>`_ at
    UCLA.  The code contained herein may be reused, adapted, or
    modified so long as proper attribution is made to the original
    authors.

:REVISIONS:
   2011-04-22 11:08 IJMC: Finished, renamed occultation functions.
                          Cleaned up documentation. Published to
                          website.

   2011-04-25 17:32 IJMC: Fixed bug in :func:`ellpic_bulirsch`.

   2012-03-09 08:38 IJMC: Several major bugs fixed, courtesy of
                          S. Aigrain at Oxford U.

   2012-03-20 14:12 IJMC: Fixed modeleclipse_simple based on new
                          format of :func:`occultuniform.  `

"""

import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from scipy import special, misc
#import pdb

eps = np.finfo(float).eps
zeroval = eps*1e6

def appelf1_ac(a, b1, b2, c, z1, z2, **kwargs):
    """Analytic continuations of the Appell hypergeometric function of 2 variables.

    :REFERENCE:
       Olsson 1964, Colavecchia et al. 2001
    """
    # 2012-03-09 12:05 IJMC: Created

def appellf1(a,b1,b2,c,z1,z2,**kwargs):
    """Give the Appell hypergeometric function of two variables.

    :INPUTS:
       six parameters, all scalars.

    :OPTIONS:
       eps -- scalar, machine tolerance precision.  Defaults to 1e-10.

    :NOTES:
       Adapted from the `mpmath <http://code.google.com/p/mpmath/>`_
       module, but using the scipy (instead of mpmath) Gauss
       hypergeometric function speeds things up.

    :LICENSE:
       MPMATH Copyright (c) 2005-2010 Fredrik Johansson and mpmath
       contributors.  All rights reserved.

       Redistribution and use in source and binary forms, with or
       without modification, are permitted provided that the following
       conditions are met:

       a. Redistributions of source code must retain the above
          copyright notice, this list of conditions and the following
          disclaimer.

       b. Redistributions in binary form must reproduce the above
          copyright notice, this list of conditions and the following
          disclaimer in the documentation and/or other materials
          provided with the distribution.

       c. Neither the name of mpmath nor the names of its contributors
          may be used to endorse or promote products derived from this
          software without specific prior written permission.


       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
       CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
       INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
       MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
       DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE
       LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
       EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
       TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
       DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
       ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
       LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
       IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
       THE POSSIBILITY OF SUCH DAMAGE.
    """
    #2011-04-22 10:15 IJMC: Adapted from mpmath, but using scipy Gauss
    #   hypergeo. function
    # 2013-03-11 13:34 IJMC: Added a small error-trap for 'nan' hypgf values

    if kwargs.has_key('eps'):
        eps = kwargs['eps']
    else:
        eps = 1e-9

    # Assume z1 smaller
    # We will use z1 for the outer loop
    if abs(z1) > abs(z2):
        z1, z2 = z2, z1
        b1, b2 = b2, b1
    def ok(x):
        return abs(x) < 0.99
    # IJMC: Ignore the finite cases for now....
    ## Finite cases
    #if ctx.isnpint(a):
    #    pass
    #elif ctx.isnpint(b1):
    #    pass
    #elif ctx.isnpint(b2):
    #    z1, z2, b1, b2 = z2, z1, b2, b1
    #else:
    #    #z1, z2
    #    # Note: ok if |z2| > 1, because
    #    # 2F1 implements analytic continuation
    if not ok(z1):
        u1 = (z1-z2)/(z1-1)
        if not ok(u1):
            raise ValueError("Analytic continuation not implemented")
        #print "Using analytic continuation"
        return (1-z1)**(-b1)*(1-z2)**(c-a-b2)*\
            appellf1(c-a,b1,c-b1-b2,c,u1,z2,**kwargs)

    #print "inner is", a, b2, c
    ##one = ctx.one
    s = 0
    t = 1
    k = 0

    while 1:
        #h = ctx.hyp2f1(a,b2,c,z2,zeroprec=ctx.prec,**kwargs)
        #print a.__class__, b2.__class__, c.__class__, z2.__class__
        h = special.hyp2f1(float(a), float(b2), float(c), float(z2))
        if not np.isfinite(h):
            break
        term = t * h
        if abs(term) < eps and abs(h) > 10*eps:
            break
        s += term
        k += 1
        t = (t*a*b1*z1) / (c*k)
        c += 1 # one
        a += 1 # one
        b1 += 1 # one
        #print k, h, term, s
        #if (k/200.)==int(k/200.) or k==171: pdb.set_trace()


    return s

def ellke2(k, tol=100*eps, maxiter=100):
    """Compute complete elliptic integrals of the first kind (K) and
    second kind (E) using the series expansions."""
    # 2011-04-24 21:14 IJMC: Created

    k = np.array(k, copy=False)
    ksum = 0*k
    kprevsum = ksum.copy()
    kresidual = ksum + 1
    #esum = 0*k
    #eprevsum = esum.copy()
    #eresidual = esum + 1
    n = 0
    sqrtpi = np.sqrt(np.pi)
    #kpow = k**0
    #ksq = k*k

    while (np.abs(kresidual) > tol).any() and n <= maxiter:
        #kpow *= (ksq)
        #print kpow==(k**(2*n))
        ksum += ((misc.factorial2(2*n - 1)/misc.factorial2(2*n))**2) * k**(2*n)
        #ksum += (special.gamma(n + 0.5)/special.gamma(n + 1) / sqrtpi) * k**(2*n)
        kresidual = ksum - kprevsum
        kprevsum = ksum.copy()
        n += 1
        #print n, kresidual

    return ksum * (np.pi/2.)

def ellke(k):
    """Compute Hasting's polynomial approximation for the complete
    elliptic integral of the first (ek) and second (kk) kind.

    :INPUTS:
       k -- scalar or Numpy array

    :OUTPUTS:
       ek, kk

    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJC: Adapted from J. Eastman's IDL code.

    m1 = 1. - k**2
    logm1 = np.log(m1)

    # First kind:
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    ee1 = 1. + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ee2 = m1 * (b1 + m1*(b2 + m1*(b3 + m1*b4))) * (-logm1)

    # Second kind:
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0 + m1*(a1 + m1*(a2 + m1*(a3 + m1*a4)))
    ek2 = (b0 + m1*(b1 + m1*(b2 + m1*(b3 + m1*b4)))) * logm1

    return ee1 + ee2, ek1 - ek2

def ellpic_bulirsch(n, k, tol=1000*eps, maxiter=1e4):
    """Compute the complete elliptical integral of the third kind
    using the algorithm of Bulirsch (1965).

    :INPUTS:
       n -- scalar or Numpy array

       k-- scalar or Numpy array

    :NOTES:
       Adapted from the IDL function of the same name by J. Eastman (OSU).
       """
    # 2011-04-19 09:15 IJMC: Adapted from J. Eastman's IDL code.
    # 2011-04-25 11:40 IJMC: Set a more stringent tolerance (from 1e-8
    #                  to 1e-14), and fixed tolerance flag to the
    #                  maximum of all residuals.
    # 2013-04-13 21:31 IJMC: Changed 'max' call to 'any'; minor speed boost.

    # Make p, k into vectors:
    #if not hasattr(n, '__iter__'):
    #    n = array([n])
    #if not hasattr(k, '__iter__'):
    #    k = array([k])

    if not hasattr(n,'__iter__'):
        n = np.array([n])
    if not hasattr(k,'__iter__'):
        k = np.array([k])

    if len(n)==0 or len(k)==0:
        return np.array([])

    kc = np.sqrt(1. - k**2)
    p = n + 1.

    if min(p) < 0:
        print("Negative p")

    # Initialize:
    m0 = np.array(1.)
    c = np.array(1.)
    p = np.sqrt(p)
    d = 1./p
    e = kc.copy()

    outsideTolerance = True
    iter = 0
    while outsideTolerance and iter<maxiter:
        f = c.copy()
        c = d/p + c
        g = e/p
        d = 2. * (f*g + d)
        p = g + p;
        g = m0.copy()
        m0 = kc + m0
        if ((np.abs(1. - kc/g)) > tol).any():
            kc = 2. * np.sqrt(e)
            e = kc * m0
            iter += 1
        else:
            outsideTolerance = False
        #if (iter/10.) == (iter/10):
        #    print iter, (np.abs(1. - kc/g))
        #pdb.set_trace()
        ## For debugging:
        #print min(np.abs(1. - kc/g)) > tol
        #print 'tolerance>>', tol
        #print 'minimum>>  ', min(np.abs(1. - kc/g))
        #print 'maximum>>  ', max(np.abs(1. - kc/g)) #, (np.abs(1. - kc/g))

    return .5 * np.pi * (c*m0 + d) / (m0 * (m0 + p))

def z2dt_circular(per, inc, ars, z):
    """ Convert transit crossing parameter z to a time offset for circular orbits.

    :INPUTS:
        per --  scalar. planetary orbital period

        inc -- scalar. orbital inclination (in degrees)

        ars -- scalar.  ratio a/Rs,  orbital semimajor axis over stellar radius

        z -- scalar or array; transit crossing parameter z.

    :RETURNS:
        |dt| -- magnitude of the time offset from transit center at
                which specified z occurs.
        """
    # 2011-06-14 11:26 IJMC: Created.

    numer = (z / ars)**2 - 1.
    denom = np.cos(inc*np.pi/180.)**2 - 1.
    dt = (per / (2*np.pi)) * np.arccos(np.sqrt(numer / denom))

    return dt


def t2z(tt, per, inc, hjd, ars, ecc=0, longperi=0):
    """Convert HJD (time) to transit crossing parameter z.

    :INPUTS:
        tt --  scalar. transit ephemeris

        per --  scalar. planetary orbital period

        inc -- scalar. orbital inclination (in degrees)

        hjd -- scalar or array of times, typically heliocentric or
               barycentric julian date.

        ars -- scalar.  ratio a/Rs,  orbital semimajor axis over stellar radius

        ecc -- scalar.  orbital eccentricity.

        longperi=0 scalar.  longitude of periapse (in radians)

    :ALGORITHM:
       At zero eccentricity, z relates to physical quantities by:

       z = (a/Rs) * sqrt(sin[w*(t-t0)]**2+[cos(i)*cos(w*[t-t0])]**2)
       """
    # 2010-01-11 18:18 IJC: Created
    # 2011-04-19 15:20 IJMC: Updated documentation.
    # 2011-04-22 11:27 IJMC: Updated to avoid reliance on planet objects.
    # 2011-05-22 16:51 IJMC: Temporarily removed eccentricity
    #                        dependence... I'll deal with that later.

    #if not p.transit:
    #    print "Must use a transiting exoplanet!"
    #    return False
    import Analysis as an


    if ecc != 0:
        ecc = 0
        print("WARNING: setting ecc=0 for now until I get this function working")


    if ecc==0:
        omega_orb = 2*np.pi/per
        omega_tdiff = omega_orb * (hjd - tt)
        z = ars * np.sqrt(np.sin(omega_tdiff)**2 + \
                              (np.cos(inc*np.pi/180.)*np.cos(omega_tdiff))**2)
    else:
        if longperi is None:
            longperi = 180.
        f = an.trueanomaly(ecc, (2*np.pi/per) * (hjd - tt))
        z = ars * (1. - ecc**2) * np.sqrt(1. - (np.sin(longperi + f) * np.sin(inc*np.pi/180.))**2) / \
            (1. + ecc * np.cos(f))

    return z

def uniform(*arg, **kw):
    """Placeholder for my old code; the new function is called
    :func:`occultuniform`.
    """
    # 2011-04-19 15:06 IJMC: Created
    print("The function 'transit.uniform()' is deprecated.")
    print("Please use transit.occultuniform() in the future.")
    return occultuniform(*arg, **kw)

def occultuniform(z, p, complement=False, verbose=False):
    """Uniform-disk transit light curve (i.e., no limb darkening).

    :INPUTS:
       z -- scalar or sequence; positional offset values of planet in
            units of the stellar radius.

       p -- scalar;  planet/star radius ratio.

       complement : bool
         If True, return (1 - occultuniform(z, p))

    :SEE ALSO:  :func:`t2z`, :func:`occultquad`, :func:`occultnonlin_small`
    """
    # 2011-04-15 16:56 IJC: Added a tad of documentation
    # 2011-04-19 15:21 IJMC: Cleaned up documentation.
    # 2011-04-25 11:07 IJMC: Can now handle scalar z input.
    # 2011-05-15 10:20 IJMC: Fixed indexing check (size, not len)
    # 2012-03-09 08:30 IJMC: Added "complement" argument for backwards
    #                        compatibility, and fixed arccos error at
    #                        1st/4th contact point (credit to
    #                        S. Aigrain @ Oxford)
    # 2013-04-13 21:28 IJMC: Some code optimization; ~20% gain.

    z = np.abs(np.array(z,copy=True))
    fsecondary = np.zeros(z.shape,float)
    if p < 0:
        pneg = True
        p = np.abs(p)
    else:
        pneg = False

    p2 = p*p

    if len(z.shape)>0: # array entered
        i1 = (1+p)<z
        i2 = (np.abs(1-p) < z) * (z<= (1+p))
        i3 = z<= (1-p)
        i4 = z<=(p-1)

        any2 = i2.any()
        any3 = i3.any()
        any4 = i4.any()
        #print i1.sum(),i2.sum(),i3.sum(),i4.sum()

        if any2:
            zi2 = z[i2]
            zi2sq = zi2*zi2
            arg1 = 1 - p2 + zi2sq
            acosarg1 = (p2+zi2sq-1)/(2.*p*zi2)
            acosarg2 = arg1/(2*zi2)
            acosarg1[acosarg1 > 1] = 1.  # quick fix for numerical precision errors
            acosarg2[acosarg2 > 1] = 1.  # quick fix for numerical precision errors
            k0 = np.arccos(acosarg1)
            k1 = np.arccos(acosarg2)
            k2 = 0.5*np.sqrt(4*zi2sq-arg1*arg1)
            fsecondary[i2] = (1./np.pi)*(p2*k0 + k1 - k2)

        fsecondary[i1] = 0.
        if any3: fsecondary[i3] = p2
        if any4: fsecondary[i4] = 1.

        if verbose:
            if not (i1+i2+i3+i4).all():
                print("warning -- some input values not indexed!")
            if (i1.sum()+i2.sum()+i3.sum()+i4.sum() != z.size):
                print("warning -- indexing didn't get the right number of values")



    else:  # scalar entered
        if (1+p)<=z:
            fsecondary = 0.
        elif (np.abs(1-p) < z) * (z<= (1+p)):
            z2 = z*z
            k0 = np.arccos((p2+z2-1)/(2.*p*z))
            k1 = np.arccos((1-p2+z2)/(2*z))
            k2 = 0.5*np.sqrt(4*z2-(1+z2-p2)**2)
            fsecondary = (1./np.pi)*(p2*k0 + k1 - k2)
        elif z<= (1-p):
            fsecondary = p2
        elif z<=(p-1):
            fsecondary = 1.

    if pneg:
        fsecondary *= -1

    if complement:
        return fsecondary
    else:
        return 1. - fsecondary

def depthchisq(z, planet, data, ddepth=[-.1,.1], ndepth=20, w=None):
    #z = transit.t2z(planet, planet.i, hjd, 0.211)
    nobs = z.size
    depths = np.linspace(ddepth[0],ddepth[1], ndepth)
    print(depths)
    chisq = np.zeros(ndepth, float)
    for ii in range(ndepth):
        tr = -(transit.occultuniform(z, np.sqrt(planet.depth))/depths[ii])
        if w is None:
            w = np.ones(nobs,float)/data[tr==0].std()
        print('w>>',w[0])
        baseline = np.ones(nobs,float) * an.wmean(data[tr==0], w[tr==0])
        print('b>>',baseline[0])
        print('d[ii]>>',depths[ii])
        model = baseline + tr*depths[ii]
        plot(model)
        chisq[ii] = (w*(model-data)**2).sum()
    return depths, chisq

def integral_smallplanet_nonlinear(z, p, cn, lower, upper):
    """Return the integral in I*(z) in Eqn. 8 of Mandel & Agol (2002).
    -- Int[I(r) 2r dr]_{z-p}^{1}, where:

    :INPUTS:
         z = scalar or array.  Distance between center of star &
             planet, normalized by the stellar radius.

         p = scalar.  Planet/star radius ratio.

         cn = 4-sequence.  Nonlinear limb-darkening coefficients,
              e.g. from Claret 2000.

         lower, upper -- floats. Limits of integration in units of mu

    :RETURNS:
         value of the integral at specified z.

         """
    # 2010-11-06 14:12 IJC: Created
    # 2012-03-09 08:54 IJMC: Added a cheat for z very close to zero

    #import pdb

    z = np.array(z, copy=True)
    z[z==0] = zeroval
    lower = np.array(lower, copy=True)
    upper = np.array(upper, copy=True)
    a = (z - p)**2
    return eval_int_at_limit(upper, cn) - eval_int_at_limit(lower, cn)

def eval_int_at_limit(limit, cn):
    """Evaluate the integral at a specified limit (upper or lower)"""
    # 2013-04-17 22:27 IJMC: Implemented some speed boosts; added a
    #                        bug; fixed it again.

    # The old way:
    #term1 = cn[0] * (1. - 0.8 * np.sqrt(limit))
    #term2 = cn[1] * (1. - (2./3.) * limit)
    #term3 = cn[2] * (1. - (4./7.) * limit**1.5)
    #term4 = cn[3] * (1. - 0.5 * limit**2)
    #goodret = -(limit**2) * (1. - term1 - term2 - term3 - term4)

    # The new, somewhat faster, way:
    sqrtlimit = np.sqrt(limit)
    sqlimit = limit*limit
    total = 1. - cn[0] * (1. - 0.8 * sqrtlimit)
    total -= cn[1] * (1. - (2./3.) * limit)
    total -= cn[2] * (1. - (4./7.) * limit*sqrtlimit)
    total -= cn[3] * (1. - 0.5 * sqlimit)
    return -(sqlimit) * total

def smallplanet_nonlinear(*arg, **kw):
    """Placeholder for backwards compatibility with my old code.  The
     function is now called :func:`occultnonlin_small`.
    """
    # 2011-04-19 15:10 IJMC: Created

    print("The function 'transit.smallplanet_nonlinear()' is deprecated.")
    print("Please use transit.occultnonlin_small() in the future.")

    return occultnonlin_small(*arg, **kw)


def occultnonlin_small(z,p, cn):
    """Nonlinear limb-darkening light curve in the small-planet
    approximation (section 5 of Mandel & Agol 2002).

    :INPUTS:
        z -- sequence of positional offset values

        p -- planet/star radius ratio

        cn -- four-sequence nonlinear limb darkening coefficients.  If
              a shorter sequence is entered, the later values will be
              set to zero.

    :NOTE:
       I had to divide the effect at the near-edge of the light curve
       by pi for consistency; this factor was not in Mandel & Agol, so
       I may have coded something incorrectly (or there was a typo).

    :EXAMPLE:
       ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         cns = vstack((zeros(4), eye(4)))
         figure()
         for coef in cns:
             f = transit.occultnonlin_small(z, 0.1, coef)
             plot(z, f, '--')

    :SEE ALSO:
       :func:`t2z`
    """
    # 2010-11-06 14:23 IJC: Created
    # 2011-04-19 15:22 IJMC: Updated documentation.  Renamed.
    # 2011-05-24 14:00 IJMC: Now check the size of cn.
    # 2012-03-09 08:54 IJMC: Added a cheat for z very close to zero
    # 2013-04-17 10:51 IJMC: Mild code optimization

    #import pdb

    cn = np.array([cn], copy=True).ravel()
    if cn.size < 4:
        cn = np.concatenate((cn, [0.]*(4-cn.size)))

    z = np.array(z, copy=False)
    F = np.ones(z.shape, float)
    z[z==0] = zeroval # cheat!

    a = (z - p)**2
    b = (z + p)**2
    c0 = 1. - np.sum(cn)
    Omega = 0.25 * c0 + np.sum( cn / np.arange(5., 9.) )

    ind1 = ((1. - p) < z) * ((1. + p) > z)
    ind2 = z <= (1. - p)

    # Need to specify limits of integration in terms of mu (not r)
    aind1 = 1. - a[ind1]
    zind1 = z[ind1]
    zind1m1 = zind1 - 1.
    Istar_edge = integral_smallplanet_nonlinear(zind1, p, cn, \
                                                np.sqrt(aind1), 0.) / aind1

    zind2 = z[ind2]
    Istar_inside = integral_smallplanet_nonlinear(zind2, p, cn, \
                                              np.sqrt(1. - a[ind2]), \
                                              np.sqrt(1. - b[ind2])) / \
                                              (4. * zind2 * p)


    term1 = 0.25 * Istar_edge / (np.pi * Omega)
    term2 = p*p * np.arccos((zind1m1) / p)
    term3 = (zind1m1) * np.sqrt(p*p - (zind1m1*zind1m1))

    F[ind1] = 1. - term1 * (term2 - term3)
    F[ind2] = 1. - 0.25 * p*p * Istar_inside / Omega

    #pdb.set_trace()
    return F

def occultquad(z,p0, gamma, retall=False, verbose=False):
    """Quadratic limb-darkening light curve; cf. Section 4 of Mandel & Agol (2002).

    :INPUTS:
        z -- sequence of positional offset values

        p0 -- planet/star radius ratio

        gamma -- two-sequence.
           quadratic limb darkening coefficients.  (c1=c3=0; c2 =
           gamma[0] + 2*gamma[1], c4 = -gamma[1]).  If only a single
           gamma is used, then you're assuming linear limb-darkening.

    :OPTIONS:
        retall -- bool.
           If True, in addition to the light curve return the
           uniform-disk light curve, lambda^d, and eta^d parameters.
           Using these quantities allows for quicker model generation
           with new limb-darkening coefficients -- the speed boost is
           roughly a factor of 50.  See the second example below.

    :EXAMPLE:
       ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 100)
         gammavals = [[0., 0.], [1., 0.], [2., -1.]]
         figure()
         for gammas in gammavals:
             f = transit.occultquad(z, 0.1, gammas)
             plot(z, f)

       ::

         # Calculate the same geometric transit with two different
         #    sets of limb darkening coefficients:
         from pylab import *
         import transit
         p, b = 0.1, 0.5
         x = (arange(300.)/299. - 0.5)*2.
         z = sqrt(x**2 + b**2)
         gammas = [.25, .75]
         F1, Funi, lambdad, etad = transit.occultquad(z, p, gammas, retall=True)

         gammas = [.35, .55]
         F2 = 1. - ((1. - gammas[0] - 2.*gammas[1])*(1. - F1) +
            (gammas[0] + 2.*gammas[1])*(lambdad + 2./3.*(p > z)) + gammas[1]*etad) /
            (1. - gammas[0]/3. - gammas[1]/6.)
         figure()
         plot(x, F1, x, F2)
         legend(['F1', 'F2'])


    :SEE ALSO:
       :func:`t2z`, :func:`occultnonlin_small`, :func:`occultuniform`

    :NOTES:
       In writing this I relied heavily on the occultquad IDL routine
       by E. Agol and J. Eastman, especially for efficient computation
       of elliptical integrals and for identification of several
       apparent typographic errors in the 2002 paper (see comments in
       the source code).

       From some cursory testing, this routine appears about 9 times
       slower than the IDL version.  The difference drops only
       slightly when using precomputed quantities (i.e., retall=True).
       A large portion of time is taken up in :func:`ellpic_bulirsch`
       and :func:`ellke`, but at least as much is taken up by this
       function itself.  More optimization (or a C wrapper) is desired!
    """
    # 2011-04-15 15:58 IJC: Created; forking from smallplanet_nonlinear
    # 2011-05-14 22:03 IJMC: Now linear-limb-darkening is allowed with
    #                        a single parameter passed in.
    # 2013-04-13 21:06 IJMC: Various code tweaks; speed increased by
    #                        ~20% in some cases.
    #import pdb

    # Initialize:
    z = np.array(z, copy=False)
    lambdad = np.zeros(z.shape, float)
    etad = np.zeros(z.shape, float)
    F = np.ones(z.shape, float)

    p = np.abs(p0) # Save the original input

    # Define limb-darkening coefficients:
    if len(gamma) < 2 or not hasattr(gamma, '__iter__'):  # Linear limb-darkening
        gamma = np.concatenate([np.ravel(gamma), [0.]])
        c2 = gamma[0]
    else:
        c2 = gamma[0] + 2 * gamma[1]

    c4 = -gamma[1]



    # Test the simplest case (a zero-sized planet):
    if p==0:
        if retall:
            ret = np.ones(z.shape, float), np.ones(z.shape, float), \
                  np.zeros(z.shape, float), np.zeros(z.shape, float)
        else:
            ret = np.ones(z.shape, float)
        return ret

    # Define useful constants:
    fourOmega = 1. - gamma[0]/3. - gamma[1]/6. # Actually 4*Omega
    a = (z - p)*(z - p)
    b = (z + p)*(z + p)
    k = 0.5 * np.sqrt((1. - a) / (z * p))
    p2 = p*p
    z2 = z*z
    ninePi = 9*np.pi

    # Define the many necessary indices for the different cases:
    pgt0 = p > 0

    i01 = pgt0 * (z >= (1. + p))
    i02 = pgt0 * (z > (.5 + np.abs(p - 0.5))) * (z < (1. + p))
    i03 = pgt0 * (p < 0.5) * (z > p) * (z < (1. - p))
    i04 = pgt0 * (p < 0.5) * (z == (1. - p))
    i05 = pgt0 * (p < 0.5) * (z == p)
    i06 = (p == 0.5) * (z == 0.5)
    i07 = (p > 0.5) * (z == p)
    i08 = (p > 0.5) * (z >= np.abs(1. - p)) * (z < p)
    i09 = pgt0 * (p < 1) * (z > 0) * (z < (0.5 - np.abs(p - 0.5)))
    i10 = pgt0 * (p < 1) * (z == 0)
    i11 = (p > 1) * (z >= 0.) * (z < (p - 1.))
    #any01 = i01.any()
    #any02 = i02.any()
    #any03 = i03.any()
    any04 = i04.any()
    any05 = i05.any()
    any06 = i06.any()
    any07 = i07.any()
    #any08 = i08.any()
    #any09 = i09.any()
    any10 = i10.any()
    any11 = i11.any()
    #print n01, n02, n03, n04, n05, n06, n07, n08, n09, n10, n11
    if verbose:
        allind = i01 + i02 + i03 + i04 + i05 + i06 + i07 + i08 + i09 + i10 + i11
        nused = (i01.sum() + i02.sum() + i03.sum() + i04.sum() + \
                     i05.sum() + i06.sum() + i07.sum() + i08.sum() + \
                     i09.sum() + i10.sum() + i11.sum())

        print("%i/%i indices used" % (nused, i01.size))
        if not allind.all():
            print("Some indices not used!")

    #pdb.set_trace()


    # Lambda^e and eta^d are more tricky:
    # Simple cases:
    lambdad[i01] = 0.
    etad[i01] = 0.

    if any06:
        lambdad[i06] = 1./3. - 4./ninePi
        etad[i06] = 0.09375 # = 3./32.

    if any11:
        lambdad[i11] = 1.
        # etad[i11] = 1.  # This is what the paper says
        etad[i11] = 0.5 # Typo in paper (according to J. Eastman)


    # Lambda_1:
    ilam1 = i02 + i08
    q1 = p2 - z2[ilam1]
    ## This is what the paper says:
    #ellippi = ellpic_bulirsch(1. - 1./a[ilam1], k[ilam1])
    # ellipe, ellipk = ellke(k[ilam1])

    # This is what J. Eastman's code has:

    # 2011-04-24 20:32 IJMC: The following codes act funny when
    #                        sqrt((1-a)/(b-a)) approaches unity.
    qq = np.sqrt((1. - a[ilam1]) / (b[ilam1] - a[ilam1]))
    ellippi = ellpic_bulirsch(1./a[ilam1] - 1., qq)
    ellipe, ellipk = ellke(qq)
    lambdad[ilam1] = (1./ (ninePi*np.sqrt(p*z[ilam1]))) * \
        ( ((1. - b[ilam1])*(2*b[ilam1] + a[ilam1] - 3) - \
               3*q1*(b[ilam1] - 2.)) * ellipk + \
              4*p*z[ilam1]*(z2[ilam1] + 7*p2 - 4.) * ellipe - \
              3*(q1/a[ilam1])*ellippi)

    # Lambda_2:
    ilam2 = i03 + i09
    q2 = p2 - z2[ilam2]

    ## This is what the paper says:
    #ellippi = ellpic_bulirsch(1. - b[ilam2]/a[ilam2], 1./k[ilam2])
    # ellipe, ellipk = ellke(1./k[ilam2])

    # This is what J. Eastman's code has:
    ailam2 = a[ilam2] # Pre-cached for speed
    bilam2 = b[ilam2] # Pre-cached for speed
    omailam2 = 1. - ailam2 # Pre-cached for speed
    ellippi = ellpic_bulirsch(bilam2/ailam2 - 1, np.sqrt((bilam2 - ailam2)/(omailam2)))
    ellipe, ellipk = ellke(np.sqrt((bilam2 - ailam2)/(omailam2)))

    lambdad[ilam2] = (2. / (ninePi*np.sqrt(omailam2))) * \
        ((1. - 5*z2[ilam2] + p2 + q2*q2) * ellipk + \
             (omailam2)*(z2[ilam2] + 7*p2 - 4.) * ellipe - \
             3*(q2/ailam2)*ellippi)


    # Lambda_3:
    #ellipe, ellipk = ellke(0.5/ k)  # This is what the paper says
    if any07:
        ellipe, ellipk = ellke(0.5/ p)  # Corrected typo (1/2k -> 1/2p), according to J. Eastman
        lambdad[i07] = 1./3. + (16.*p*(2*p2 - 1.)*ellipe -
                                (1. - 4*p2)*(3. - 8*p2)*ellipk / p) / ninePi


    # Lambda_4
    #ellipe, ellipk = ellke(2. * k)  # This is what the paper says
    if any05:
        ellipe, ellipk = ellke(2. * p)  # Corrected typo (2k -> 2p), according to J. Eastman
        lambdad[i05] = 1./3. + (2./ninePi) * (4*(2*p2 - 1.)*ellipe + (1. - 4*p2)*ellipk)

    # Lambda_5
    ## The following line is what the 2002 paper says:
    #lambdad[i04] = (2./(3*np.pi)) * (np.arccos(1 - 2*p) - (2./3.) * (3. + 2*p - 8*p2))
    # The following line is what J. Eastman's code says:
    if any04:
        lambdad[i04] = (2./3.) * (np.arccos(1. - 2*p)/np.pi - \
                                      (6./ninePi) * np.sqrt(p * (1.-p)) * \
                                      (3. + 2*p - 8*p2) - \
                                      float(p > 0.5))

    # Lambda_6
    if any10:
        lambdad[i10] = -(2./3.) * (1. - p2)**1.5

    # Eta_1:
    ilam3 = ilam1 + i07 # = i02 + i07 + i08
    z2ilam3  = z2[ilam3]    # pre-cache for better speed
    twoZilam3  = 2*z[ilam3] # pre-cache for better speed
    #kappa0 = np.arccos((p2+z2ilam3-1)/(p*twoZilam3))
    #kappa1 = np.arccos((1-p2+z2ilam3)/(twoZilam3))
    #etad[ilam3] = \
    #    (0.5/np.pi) * (kappa1 + kappa0*p2*(p2 + 2*z2ilam3) - \
    #                    0.25*(1. + 5*p2 + z2ilam3) * \
    #                    np.sqrt((1. - a[ilam3]) * (b[ilam3] - 1.)))
    etad[ilam3] = \
        (0.5/np.pi) * ((np.arccos((1-p2+z2ilam3)/(twoZilam3))) + (np.arccos((p2+z2ilam3-1)/(p*twoZilam3)))*p2*(p2 + 2*z2ilam3) - \
                        0.25*(1. + 5*p2 + z2ilam3) * \
                        np.sqrt((1. - a[ilam3]) * (b[ilam3] - 1.)))


    # Eta_2:
    etad[ilam2 + i04 + i05 + i10] = 0.5 * p2 * (p2 + 2. * z2[ilam2 + i04 + i05 + i10])


    # We're done!


    ## The following are handy for debugging:
    #term1 = (1. - c2) * lambdae
    #term2 = c2*lambdad
    #term3 = c2*(2./3.) * (p>z).astype(float)
    #term4 = c4 * etad
    # Lambda^e is easy:
    lambdae = 1. - occultuniform(z, p)
    F = 1. - ((1. - c2) * lambdae + \
                  c2 * (lambdad + (2./3.) * (p > z).astype(float)) - \
                  c4 * etad) / fourOmega

    #pdb.set_trace()
    if retall:
        ret = F, lambdae, lambdad, etad
    else:
        ret = F

    #pdb.set_trace()
    return ret

def occultnonlin(z,p0, cn):
    """Nonlinear limb-darkening light curve; cf. Section 3 of Mandel & Agol (2002).

    :INPUTS:
        z -- sequence of positional offset values

        p0 -- planet/star radius ratio

        cn -- four-sequence. nonlinear limb darkening coefficients

    :EXAMPLE:
        ::

         # Reproduce Figure 2 of Mandel & Agol (2002):
         from pylab import *
         import transit
         z = linspace(0, 1.2, 50)
         cns = vstack((zeros(4), eye(4)))
         figure()
         for coef in cns:
             f = transit.occultnonlin(z, 0.1, coef)
             plot(z, f)

    :SEE ALSO:
       :func:`t2z`, :func:`occultnonlin_small`, :func:`occultuniform`, :func:`occultquad`

    :NOTES:
        Scipy is much faster than mpmath for computing the Beta and
        Gauss hypergeometric functions.  However, Scipy does not have
        the Appell hypergeometric function -- the current version is
        not vectorized.
    """
    # 2011-04-15 15:58 IJC: Created; forking from occultquad
    #import pdb

    # Initialize:
    cn0 = np.array(cn, copy=True)
    z = np.array(z, copy=True)
    F = np.ones(z.shape, float)

    p = np.abs(p0) # Save the original input


    # Test the simplest case (a zero-sized planet):
    if p==0:
        ret = np.ones(z.shape, float)
        return ret

    # Define useful constants:
    c0 = 1. - np.sum(cn0)
    # Row vectors:
    c = np.concatenate(([c0], cn0))
    n = np.arange(5, dtype=float)
    # Column vectors:
    cc = c.reshape(5, 1)
    nn = n.reshape(5,1)
    np4 = n + 4.
    nd4 = n / 4.
    twoOmega = 0.5*c[0] + 0.4*c[1] + c[2]/3. + 2.*c[3]/7. + 0.25*c[4]

    a = (z - p)**2
    b = (z + p)**2
    am1 = a - 1.
    bma = b - a

    k = 0.5 * np.sqrt(-am1 / (z * p))
    p2 = p**2
    z2 = z**2


    # Define the many necessary indices for the different cases:
    i01 = (p > 0) * (z >= (1. + p))
    i02 = (p > 0) * (z > (.5 + np.abs(p - 0.5))) * (z < (1. + p))
    i03 = (p > 0) * (p < 0.5) * (z > p) * (z <= (1. - p))  # also contains Case 4
    #i04 = (z==(1. - p))
    i05 = (p > 0) * (p < 0.5) * (z == p)
    i06 = (p == 0.5) * (z == 0.5)
    i07 = (p > 0.5) * (z == p)
    i08 = (p > 0.5) * (z >= np.abs(1. - p)) * (z < p)
    i08a = (p == 1) * (z == 0)
    i09 = (p > 0) * (p < 1) * (z > 0) * (z < (0.5 - np.abs(p - 0.5)))
    i10 = (p > 0) * (p < 1) * (z == 0)
    i11 = (p > 1) * (z >= 0.) * (z < (p - 1.))

    iN = i02 + i08
    iM = i03 + i09

    # Compute N and M for the appropriate indices:
    #  (Use the slow, non-vectorized appellf1 function:)
    myappellf1 = np.frompyfunc(appellf1, 6, 1)
    N = np.zeros((5, z.size), float)
    M = np.zeros((3, z.size), float)
    if iN.any():
        termN = myappellf1(0.5, 1., 0.5, 0.25*nn + 2.5, am1[iN]/a[iN], -am1[iN]/bma[iN])
        N[:, iN] = ((-am1[iN])**(0.25*nn + 1.5)) / np.sqrt(bma[iN]) * \
            special.beta(0.25*nn + 2., 0.5) * \
            (((z2[iN] - p2) / a[iN]) * termN - \
                 special.hyp2f1(0.5, 0.5, 0.25*nn + 2.5, -am1[iN]/bma[iN]))

    if iM.any():
        termM = myappellf1(0.5, -0.25*nn[1:4] - 1., 1., 1., -bma[iM]/am1[iM], -bma[iM]/a[iM])
        M[:, iM] = ((-am1[iM])**(0.25*nn[1:4] + 1.)) * \
            (((z2[iM] - p2)/a[iM]) * termM - \
                 special.hyp2f1(-0.25*nn[1:4] - 1., 0.5, 1., -bma[iM]/am1[iM]))


    # Begin going through all the cases:

    # Case 1:
    F[i01] = 1.

    # Case 2: (Gauss and Appell hypergeometric functions)
    F[i02] = 1. - (1. / (np.pi*twoOmega)) * \
        (N[:, i02] * cc/(nn + 4.) ).sum(0)

    # Case 3 : (Gauss and Appell hypergeometric functions)
    F[i03] = 1. - (0.5/twoOmega) * \
        (c0*p2 + 2*(M[:, i03] * cc[1:4]/(nn[1:4] + 4.)).sum(0) + \
             c[-1]*p2*(1. - 0.5*p2 - z2[i03]))

    #if i04.any():
    #    F[i04] = occultnonlin_small(z[i04], p, cn)
    #    print "Value found for z = 1-p: using small-planet approximation "
    #    print "where Appell F2 function will not otherwise converge."

    #pdb.set_trace()
    #F[i04] = 0.5 * (occultnonlin(z[i04]+p/2., p, cn) + occultnonlin(z[i04]-p/2., p, cn))

    # Case 5: (Gauss hypergeometric function)
    F[i05] = 0.5 + \
        ((c/np4) * special.hyp2f1(0.5, -nd4 - 1., 1., 4*p2)).sum() / twoOmega

    # Case 6:  Gamma function
    F[i06] = 0.5 + (1./(np.sqrt(np.pi) * twoOmega)) * \
        ((c/np4) * special.gamma(1.5 + nd4) / special.gamma(2. + nd4)).sum()

    # Case 7: Gauss hypergeometric function, beta function
    F[i07] = 0.5 + (0.5/(p * np.pi * twoOmega)) * \
        ((c/np4) * special.beta(0.5, nd4 + 2.) * \
             special.hyp2f1(0.5, 0.5, 2.5 + nd4, 0.25/p2)).sum()

    # Case 8: (Gauss and Appell hypergeometric functions)
    F[i08a] = 0.
    F[i08] =  -(1. / (np.pi*twoOmega)) * (N[:, i02] * cc/(nn + 4.) ).sum(0)

    # Case 9: (Gauss and Appell hypergeometric functions)
    F[i09] = (0.5/twoOmega) * \
        (c0 * (1. - p2) + c[-1] * (0.5 - p2*(1. - 0.5*p2 - z2[i09])) - \
             2*(M[:, i09] * cc[1:4] / (nn[1:4] + 4.)).sum(0))

    # Case 10:
    F[i10] = (2. / twoOmega) * ((c/np4) * (1. - p2)**(nd4 + 1.)).sum()

    # Case 11:
    F[i11] = 0.


    # We're done!

    return F
def modeltransit(params, func, per, t):
    """Model a transit light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN period.

    :INPUTS:
      params -- (5+N)-sequence with the following:
        the time of conjunction for each individual transit (Tc),

        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

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

      per -- float.  Orbital period, in days.

      t -- numpy array.  Time of observations.
    """
    # 2011-05-22 16:14 IJMC: Created.
    # 2011-05-24 10:52 IJMC: Inserted a check for cos(i) > 1

    ecc = 0.
    nparam = len(params)


    if (params[1] * params[2]) > 1:  # cos(i) > 1: impossible!
        return -1
    else:
        z = t2z(params[0], per, (180./np.pi)*np.arccos(params[1]*params[2]), t, 1./params[2], 0.)

    # Mask out secondary eclipses:
    #z[abs(((t - params[0] + params[1]*.25)/per % 1) - 0.5) < 0.43] = 10.

    #pdb.set_trace()

    if len(params)>5:
        model = params[4] * func(z, params[3], params[5::])
    try:  # Limb-darkened
        model = params[4] * func(z, params[3], params[5::])
    except:  # Uniform-disk
        model = params[4] * (1. - func(z, params[3]))

    return model
def modeltransit_general(params, t, NL, NP=1, errscale=1, smallplanet=True, svs=None):
    """Model a transit light curve of arbitrary type to a flux time
    series, assuming zero eccentricity.

    :INPUTS:
      params -- (6 + NL + NP + NS)-sequence with the following:
        Tc, the time of conjunction for each individual transit,

        P, the orbital period (in units of "t")

        i, the orbital inclination (in degrees; 90 is edge-on)

        R*/a, the stellar radius in units of orbital distance,

        Rp/R*, planet-to-star radius ratio,

        the NP polynomial coefficients to normalize the data.

          EITHER:
            F0 -- stellar flux _ONLY_ (set NP=1)

          OR:
            [p_1, p_2, ..., p_(NP)] -- coefficients for polyval, to be
            used as: numpy.polyval([p_1, ...], t)

        the NL limb-darkening parameters (cf. Claret+2011):

          EITHER:
            u      -- linear limb-darkening (set NL=1)

          OR:
            a, b   -- quadratic limb-darkening (set NL=2)

          OR:
            c,  d  -- root-square limb-darkening (set NL= -2)
              where
            I(mu) = 1 - c * (1 - mu) - d * (1 - mu^0.5)

          OR:
            a1, a2, a3, a4 -- nonlinear limb-darkening  (set NL=4)
              where
            I(mu) = 1 - a1 * (1 - mu^0.5) - a2 * (1 - mu) - \
                        a3 * (1 - mu^1.5) - a4 * (1 - mu^2)

          OR:
            Nothing at all -- uniform limb-darkening (set NL=0)

        the NS state vectors, passed in as 'svs'

      t -- numpy array.  Time of observations.

      smallplanet : bool
        This only matters if NL>2.  If "smallplanet" is True, use
        :func:`occultnonlin_small`.  Otherwise, use
        :func:`occultnonlin`

      errscale : int
        If certain conditions (see below) are met, the resulting
        transit light curve is scaled by this factor.  When fitting,
        set errscale to a very large value (e.g., 1e6) to use as an
        extremely crude hard-edged filter.

        A better way would be to incorporate constrained fitting...

      svs : None or list of 1D NumPy Arrays
        State vectors, applied with coefficients as defined above. To
        avoid degeneracies with the NP polynomial terms (especially
        the constant offset term), it is preferable that the state
        vectors are all mean-subtracted.

    :NOTES:

      If quadratic or linear limb-darkening (L.D.) is used, the sum of
      the L.D. coefficients cannot exceed 1.  If they do, this routine
      normalizes the coefficients [g1,g2] to:  g_i = g_i / (g1 + g2).

      If "Rp/R*", or "R*/a" are < 0, they will be set to zero.

      If "P" < 0.01, it will be set to 0.01.

      If "inc" > 90, it will be set to 90.
    """
    # 2012-04-30 03:41 IJMC: Created
    # 2012-05-08 16:59 IJMC: NL can be negative (for root-square profiles)
    # 2013-01-31 18:21 IJMC: Added 'errscale' option
    # 2013-04-02 09:07 IJMC: Added 'svs' option
    # 2013-04-18 10:22 IJMC: Apply penalty scaling in 'sqrt' case if
    #                        coefficients give nonphysical intensity values.


    ecc = 0.
    longperi = 0.

    if svs is None:
        nsvs = 0
    else:
        if isinstance(svs, np.ndarray) and svs.ndim==1:
            svs = svs.reshape(1, svs.size)
        nsvs = len(svs)
    nparam = len(params) - nsvs

    verbose = False

    tc, per, inc, ra, k = params[0:5]
    if NP>0:
        poly_params = params[5:5+NP]
    else:
        poly_params = [1]

    pNL = np.abs(NL)
    if pNL>0:
        ld_params = params[5+NP:5+NP+pNL]

    penalty_factor = 1.
    # Enforce various normalization constraints:
    if inc > 90:
        inc = 90.
        penalty_factor *= errscale

    if per < 0.01:
        per = 0.01
        penalty_factor *= errscale

    if k < 0:
        k = 0.
        penalty_factor *= errscale

    if ra <= 0:
        ra = 1e-6
        penalty_factor *= errscale

    # Enforce the constraint that cos(i) <= 1.
    #print ("%1.5f "*4) % (tc, per, inc, ra)
    z = t2z(tc, per, inc, t, 1./ra, ecc=ecc, longperi=longperi)


    if NL!=0 and (sum(ld_params)>1 or sum(ld_params)<0):
        penalty_factor *= errscale

    if NL==0:  # Uniform
        model = occultuniform(z, k)

    elif NL==2 or NL==1:  # Quadratic or Linear
        model = occultquad(z, k, ld_params, verbose=verbose)

    elif NL==-2: # Root-square (i.e., nonlinear)
        new_ld_params = [ld_params[1], ld_params[0], 0., 0.]
        if smallplanet:
            model = occultnonlin_small(z, k, new_ld_params)
        else:
            model = occultnonlin(z, k, new_ld_params)

    elif NL==4 or NL==3: # Nonlinear
        if smallplanet:
            model = occultnonlin_small(z, k, ld_params)
        else:
            model = occultnonlin(z, k, ld_params)

    model *= np.polyval(poly_params, t)
    for ii in xrange(nsvs):
        model += params[-ii-1] * svs[-ii-1]

    model *= penalty_factor
    return model

def modeleclipse(params, func, per, t):
    """Model an eclipse light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN period.

    :INPUTS:
      params -- (6-or-7)-sequence with the following:
        the time of conjunction for each individual eclipse (Tc),

        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar),

        eclipse depth (dimensionless),

        stellar flux (F0),

        orbital period (OPTIONAL!)

      func -- function to fit to data; presumably :func:`transit.occultuniform`

      per -- float.
        Orbital period,  OR

        None, if period is included in params

      t -- numpy array.
         Time of observations (same units as Tc and per)
    """
    # 2011-05-30 16:56 IJMC: Created from modeltransit()
    # 2012-01-31 22:14 IJMC: Period can be included in parameters, for
    #                        fitting purposes.

    ecc = 0.
    nparam = len(params)

    if per is None:
        per = params[6]

    if (params[1] * params[2]) > 1:  # cos(i) > 1: impossible!
        return -1
    else:
        z = t2z(params[0], per, (180./np.pi)*np.arccos(params[1]*params[2]), t, 1./params[2], 0.)

#    if len(params)>6:
#        model = params[4] * func(z, params[3], params[6::])
    try:  # Limb-darkened
        TLC = func(z, params[3], params[6::])
    except:  # Uniform-disk
        TLC =  (func(z, params[3]))

    # Appropriately scale eclipse depth:
    #pdb.set_trace()
    model = params[5] * (1. - params[4] * (TLC ) / params[3]**2)

    return model

def modellightcurve(params, t, tfunc=occultuniform, nlimb=0, nchan=0):
    """Model a full planetary light curve: transit, eclipse, and
    (sinusoidal) phase variation. Accept independent eclipse and
    transit times-of-center, but otherwise assume a circular orbit
    (and thus symmetric transits and eclipses).

    :INPUTS:
      params -- (M+10+N)-sequence with the following:

        OPTIONALLY:

          sensitivity variations for each of M channels (e.g.,
          SST/MIPS).  This assumes the times in 't' are in the order
          (T_{1,0}, ... T_{1,M-1}, ... T_{2,0}, ...).  The parameters
          affect the data multiplicatively as (1 + c_i), with the
          constraint that Prod_i(1+c_i) = 1.

        the time of conjunction for each individual transit (T_t),

        the time of conjunction for each individual eclipse (T_e),

        the orbital period (P),

        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar),

        stellar flux (F0),

        maximum (hot-side) planet flux (Fbright),

        minimum (cold-side) planet flux (Fdark),

        phase curve offset (phi_0; 0 implies maximum flux near eclipse)

        OPTIONALLY:
          limb-darkening parameters (depending on tfunc):

          EITHER:
            gamma1,  gamma2  -- quadratic limb-darkening coefficients

          OR:
            c1, c2, c3, c4 -- nonlinear limb-darkening coefficients

      t -- numpy array.  Time of observations; same units as orbital
                         period and ephemerides.  If nchan>0, t should
                         be of shape (nchan, L), or a .ravel()ed
                         version of that.

    :OPTIONS:
      tfunc : model transit function
          One of :func:`occultuniform`, :func:`occultnonlin_small`,
          :func:`occultquad`, or :func:`occultnonlin`

      nlimb : int
         number of limb-darkening parameters; these should be the last
         values of params.

      nchan : int
         number of photometric channel sensitivity perturbations;
         these should be the first 'nchan' values of params.

    :EXAMPLE:
       TBW

    """
    # 2011-06-10 11:10 IJMC: Created.
    # 2011-06-14 13:18 IJMC: Sped up with creation of z2dt()
    # 2011-06-30 21:00 IJMC: Fixed functional form of phase curve.
    from scipy import optimize
    import pdb

    ecc = 0.
    if nchan>0:
        cparams = params[0:nchan].copy()
        params = params[nchan::].copy()
        cparams[0] = 1./(1. + cparams[1::]).prod() - 1.

    nparam = len(params)
    tt, te, per, b, ra, k, fstar, fbright, fdark, phi = params[0:10]
    if nparam > 10:
        limbdarkening = params[10::]
    else:
        limbdarkening = None

    cosi = b * ra
    if (cosi) > 1:  # cos(i) > 1: impossible!
        return -1
    else:
        zt = t2z(tt, per, (180./np.pi)*np.arccos(b * ra), t, 1./ra, ecc)
        ze = t2z(te, per, (180./np.pi)*np.arccos(b * ra), t, 1./ra, ecc)
        inc = np.arccos(cosi)


    # Solve for t given z0, such that  z0 - t2z(t) = 0.
    def residualz(tguess, z0):
        return z0 - t2z(te, per, (180./np.pi)*np.arccos(b * ra), tguess, 1./ra, ecc)

    # Mask out secondary eclipses:
    #z[abs(((t - params[0] + params[1]*.25)/per % 1) - 0.5) < 0.43] = 10.

    sep = (tt - te) % per
    transit_times = (((t - tt) % per) < sep/2.) + (((t - tt) % per) > (per - sep/2.))
    eclipse_times = (((t - te) % per) < sep/2.) + (((t - te) % per) > (per - sep/2.))

    #pdb.set_trace()

    # Model phase curve flux
    def phaseflux(time):
        return 0.5*(fbright + fdark) + \
            0.5*(fbright - fdark) * np.cos((2*np.pi*(time - tt))/per + phi)

    phas = phaseflux(t)

    # Model transit:
    trans = np.ones(zt.shape, dtype=float)
    if limbdarkening is None:
        trans[transit_times] = (1. - tfunc(zt[transit_times], k))
    else:
        trans[transit_times] = tfunc(zt[transit_times], k, limbdarkening)

    transit_curve = trans*fstar + phas

    # Model eclipse:
    feclip = phaseflux(te) / (fstar + phaseflux(te))
    eclip = np.ones(ze.shape, dtype=float)
    eclip[eclipse_times] =  (1. - occultuniform(ze[eclipse_times], k))
    eclip = 1. + feclip * (eclip - 1.) / k**2


    # A lot of hokey cheats to keep the eclipse bottom flat, but
    #    ingress and egress continuous:

    ## The following code is deprecated with the creation of z2dt()
    #t14 = (per/np.pi) * np.arcsin(ra * np.sqrt((1. + k**2) - b**2)/np.sin(np.arccos(cosi)))
    #t23 = (per/np.pi) * np.arcsin(ra * np.sqrt((1. - k**2) - b**2)/np.sin(np.arccos(cosi)))
    #t12 = 0.5 * (t14 - t23)

    #zzz = [t2z(tt, per, (180./np.pi)*np.arccos(b * ra), thist, 1./ra, ecc) for thist in [te-t14, te, te+t14]]
    #aaa,bbb = residualz(te-t14, 1. + k), residualz(te, 1. + k)
    #ccc,ddd = residualz(te-t14, 1. - k), residualz(te, 1. - k)
    #if (aaa >= 0 and bbb >= 0) or (aaa <= 0 and bbb <= 0):
    #    print aaa, bbb
    #    print te, t14, t23, k, ra, b, per
    #    pdb.set_trace()
    #if (ccc >= 0 and ddd >= 0) or (ccc <= 0 and ddd <= 0):
    #    print ccc, ddd
    #    print te, t14, t23, k, ra, b, per
    #    pdb.set_trace()
    #pdb.set_trace()
    #t5 = optimize.bisect(residualz, te - 2*t14, te + t14, args=(1. + k,))
    ##t5 = optimize.newton(residualz, te - t23 - t12, args=(1. + k,))
    ##pdb.set_trace()
    ##t6 = optimize.newton(residualz, te - t23 + t12, args=(1. - k,))
    #t6 = optimize.bisect(residualz, te - 2*t14, te + t14, args=(1. - k,))

    t5 = te - z2dt_circular(per, inc*180./np.pi, 1./ra, 1. + k)
    t6 = te - z2dt_circular(per, inc*180./np.pi, 1./ra, 1. - k)
    t7 = te + (te - t6)
    t8 = te + (te - t5)
    #z58 = [t2z(tt, per, (180./np.pi)*np.arccos(b * ra), thist, 1./ra, ecc) for thist in [t5,t6,t7,t8]]

    eclipse_ingress = eclipse_times * (((t - t5) % per) < (t6 - t5))
    if eclipse_ingress.any():
        inscale = np.zeros(ze.shape, dtype=float)
        tei = t[eclipse_ingress]
        inscale[eclipse_ingress] = ((fstar + phaseflux(t6)) * (1. - feclip) - fstar) * \
            ((tei - tei.min()) / (tei.max() - tei.min()))
    else:
        inscale = 0.

    eclipse_egress = eclipse_times * (((t - t7) % per) < (t8 - t7))
    if eclipse_egress.any():
        egscale = np.zeros(ze.shape, dtype=float)
        tee = t[eclipse_egress]
        egscale[eclipse_egress] = ((fstar + phaseflux(t7)) * (1. - feclip) - fstar) * \
            ((tee - tee.max()) / (tee.max() - tee.min()))
    else:
        egscale = 0.

    # Now compute the full light curve:
    full_curve = transit_curve * eclip
    full_curve[eclipse_times * (ze < (1. - k))] = fstar
    full_curve = full_curve - inscale + egscale

    if nchan>0:
        if len(t.shape)==2:  # Data entered as 2D
            full_curve *= (1. + cparams.reshape(nchan, 1))
        else: # Data entered as 1D
            full_curve = (full_curve.reshape(nchan, full_curve.size/nchan) * \
                (1. + cparams.reshape(nchan, 1))).ravel()

    return full_curve

def modeleclipse_simple(params, tparams, func, t):
    """Model an eclipse light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN orbit.

    :INPUTS:
      params -- (3)-sequence with eclipse parameters to FIT:
        the time of conjunction for each individual eclipse (Tc),

        eclipse depth (dimensionless),

        stellar flux (F0),

      tparams -- (4)-sequence of transit parameters to HOLD FIXED:
        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar),

        orbital period (same units as Tc and t)

      func -- function to fit to data; presumably :func:`transit.occultuniform`

      t -- numpy array.  Time of observations.
    """
    # 2011-05-31 08:35 IJMC: Created anew, specifically for eclipses.

    ecc = 0.


    if (tparams[0] * tparams[1]) > 1:  # cos(i) > 1: impossible!
        return -1
    else:
        z = t2z(params[0], tparams[3], (180./np.pi)*np.arccos(tparams[0]*tparams[1]), \
                    t, 1./tparams[1], ecc=ecc)

    # Uniform-disk occultation:
    TLC =  (func(z, tparams[2]))

    # Appropriately scale eclipse depth:
    model = params[2] * (1. + params[1] * (TLC - 1.) / tparams[2]**2)

    return model

def modeleclipse_simple14(params, tparams, func, t):
    """Model an eclipse light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN orbit.

    :INPUTS:
      params -- (14+3)-sequence with eclipse parameters to FIT:
        the multiplicative sensitivity effects (c0, ..., c13), which
        affect each bit of data as (1. + c_j) * ...  HOWEVER, to keep
        these from becoming degenerate with the overall stellar flux
        level, only 13 of these are free parameters: the first (c0)
        will always be set such that the product PROD_j(1 + c_j) = 1.

        the time of conjunction for each individual eclipse (Tc),

        eclipse depth (dimensionless),

        stellar flux (F0),

      tparams -- (4)-sequence of transit parameters to HOLD FIXED:
        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

        planet-to-star radius ratio (Rp/Rstar),

        orbital period (same units as Tc and t)

      func -- function to fit to data; presumably :func:`transit.occultuniform`

      t -- numpy array.  Time of observations.
         Must either be of size (14xN), or if a 1D vector then
         t.reshape(14, N) must correctly reformat the data into data
         streams at 14 separate positions.
    """
    # 2011-05-31 08:35 IJMC: Created anew, specifically for eclipses.


    # Separate the c (sensitivity) and t (transit) parameters:
    cparams = params[0:14].reshape(14, 1)
    params = params[14::]

    cparams[0] = 1./(1.+cparams[1::]).prod() - 1.

    tis1D = False # we want "t" to be 2D

    if len(t.shape)==1:
        t = t.reshape(14, t.size/14)
        tis1D = True  # "t" is 1D
    elif len(t.shape)>2:
        print("t is of too high a dimension (>2)")
        return -1

    # Get the vanilla transit light curve:
    model = modeleclipse_simple(params, tparams, func, t)

    # Apply sensitivity calibrations:
    model *= (1. + cparams)

    if tis1D:
        model = model.ravel()

    return model

def modeltransit14(params, func, per, t):
    """Model a transit light curve of arbitrary type to a flux time
    series, assuming zero eccentricity and a fixed, KNOWN period, and
    assuming MIPS-type data with 14 separate sensitivity dependencies.

    :INPUTS:
      params -- (14+5+N)-sequence with the following:

        the multiplicative sensitivity effects (c0, ..., c13), which
        affect each bit of data as (1. + c_j) * ...  HOWEVER, to keep
        these from becoming degenerate with the overall stellar flux
        level, only 13 of these are free parameters: the first (c0)
        will always be set such that the product PROD_j(1 + c_j) = 1.

        the time of conjunction for each individual transit (Tc),

        the impact parameter (b = a cos i/Rstar)

        the stellar radius in units of orbital distance (Rstar/a),

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

      per -- float.  Orbital period, in days.

      t -- numpy array.  Time of observations.
         Must either be of size (14xN), or if a 1D vector then
         t.reshape(14, N) must correctly reformat the data into data
         streams at 14 separate positions.

    :SEE ALSO:
      :func:`modeltransit`
    """
    # 2011-05-26 13:37 IJMC: Created, from the 'vanilla' modeltransit.

    # Separate the c (sensitivity) and t (transit) parameters:
    cparams = params[0:14].reshape(14, 1)
    tparams = params[14::]

    cparams[0] = 1./(1.+cparams[1::]).prod() - 1.

    tis1D = False # we want "t" to be 2D

    if len(t.shape)==1:
        t = t.reshape(14, t.size/14)
        tis1D = True  # "t" is 1D
    elif len(t.shape)>2:
        print("t is of too high a dimension (>2)")
        return -1

    # Get the vanilla transit light curve:
    model = modeltransit(tparams, func, per, t)
    if np.sum(model + 1)==0:
        model = -np.ones(t.shape, dtype=float)

    # Apply sensitivity calibrations:
    model *= (1. + cparams)

    if tis1D:
        model = model.ravel()

    return model

def mcmc_transit_single(flux, sigma, t, per, func, params, stepsize, numit, nstep=1, posdef=None, holdfixed=None):
    """MCMC for 5-parameter eclipse function of transit with KNOWN period

    :INPUTS:
        flux : 1D array
                    Contains dependent data

        sigma : 1D array
                    Contains standard deviation (uncertainties) of flux data

        t : 1D array
                    Contains independent data: timing info

        per : scalar
                    Known orbital period (same units as t)

        func : function
                    Function to model transit (e.g., transit.occultuniform)

        params : 5+N parameters to be fit
          [T_center, b, Rstar/a, Rp/Rstar, Fstar] + (limb-darkening parameters?)
              #[Fstar, t_center, b, v (in Rstar/day), p (Rp/Rs)]

        stepsize :  1D or 2D array
                if 1D: array of 1-sigma change in parameter per iteration
                if 2D: array of covariances for new parameters

        numit : int
                Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

        posdef : None, 'all', or sequences of indices.
                Which elements should be restricted to positive definite?
                If indices, it should be of the form (e.g.): [0, 1, 4]

        holdfixed : None, or sequences of indices.
                    Which elements should be held fixed in the analysis?
                    If indices, it should be of the form (e.g.): [0, 1, 4]

    :RETURNS:
        allparams : 2D array
                Contains all parameters at each step

        bestp : 1D array
                Contains best paramters as determined by lowest Chi^2

        numaccept: int
                Number of accepted steps

        chisq: 1D array
                Chi-squared value at each step

    :REFERENCES:
        Numerical Recipes, 3rd Edition (Section 15.8);  Wikipedia
    """
    # 2011-05-14 16:06 IJMC: Adapted from upsand phase curve routines;
    #                        also adapting Agol et al. 2010's Spitzer
    #                        work, and from K. Stevenson's MCMC
    #                        example implementation.
    # 2011-05-24 15:52 IJMC: testing the new 'holdfixed' option.
    # 2011-11-02 22:08 IJMC: Now cast numit as int

    import numpy as np

    #Initial setup
    numaccept  = 0
    nout = numit/nstep
    bestp      = np.copy(params)
    params     = np.copy(params)
    original_params = np.copy(params)
    numit = int(numit)

    # Set indicated parameters to be positive definite:
    if posdef=='all':
        params = np.abs(params)
        posdef = np.arange(params.size)
    elif posdef is not None:
        posdef = np.array(posdef)
        params[posdef] = np.abs(params[posdef])
    else:
        posdef = np.zeros(params.size, dtype=bool)

    # Set indicated parameters to be positive definite:
    if holdfixed is not None:
        holdfixed = np.array(holdfixed)
        params[holdfixed] = np.abs(params[holdfixed])
    else:
        holdfixed = np.zeros(params.size, dtype=bool)

    weights = 1./sigma**2
    allparams  = np.zeros((len(params), nout))
    allchi     = np.zeros(nout,float)

    #Calc chi-squared for model type using current params
    zmodel = modeltransit(params, func, per, t)
    currchisq  = (((zmodel - flux)**2)*weights).ravel().sum()
    bestchisq  = currchisq

#Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for j in range(numit):
    #Take step in random direction for adjustable parameters
            if len(stepsize.shape)==1:
                nextp    = np.random.normal(params,stepsize)
            else:
                nextp = np.random.multivariate_normal(params, stepsize)

            nextp[posdef] = np.abs(nextp[posdef])
            nextp[holdfixed] = original_params[holdfixed]
            #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
            zmodel     = modeltransit(nextp, func, per, t)

            nextchisq  = (((zmodel - flux)**2)*weights).ravel().sum()

            accept = np.exp(0.5 * (currchisq - nextchisq))
            if (accept >= 1) or (np.random.uniform(0, 1) <= accept):
                    #Accept step
                    numaccept += 1
                    params  = np.copy(nextp)
                    currchisq  = nextchisq
            if (currchisq < bestchisq):
                            #New best fit
                            bestp     = np.copy(params)
                            bestchisq = currchisq

            if (j%nstep)==0:
                allparams[:, j/nstep] = params
                allchi[j/nstep] = currchisq
    return allparams, bestp, numaccept, allchi

def mcmc_transit_single14(flux, sigma, t, per, func, params, stepsize, numit, nstep=1, posdef=None, holdfixed=None):
    """MCMC for 5-parameter eclipse function of transit with KNOWN period

    :INPUTS:
        flux : 1D array
                    Contains dependent data

        sigma : 1D array
                    Contains standard deviation (uncertainties) of flux data

        t : 1D array
                    Contains independent data: timing info

        per : scalar
                    Known orbital period (same units as t)

        func : function
                    Function to model transit (e.g., transit.occultuniform)

        params : 14+5+N parameters to be fit
          [c0,...,c13] + [T_center, b, Rstar/a, Rp/Rstar, Fstar] + (limb-darkening parameters?)

        stepsize :  1D or 2D array
                    If 1D: 1-sigma change in parameter per iteration
                    If 2D: covariance matrix for parameter changes.

        numit : int
                Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

        posdef : None, 'all', or sequences of indices.
                Which elements should be restricted to positive definite?
                If indices, it should be of the form (e.g.): [0, 1, 4]

        holdfixed : None, or sequences of indices.
                    Which elements should be held fixed in the analysis?
                    If indices, it should be of the form (e.g.): [0, 1, 4]

    :RETURNS:
        allparams : 2D array
                Contains all parameters at each step

        bestp : 1D array
                Contains best paramters as determined by lowest Chi^2

        numaccept: int
                Number of accepted steps

        chisq: 1D array
                Chi-squared value at each step

    :REFERENCES:
        Numerical Recipes, 3rd Edition (Section 15.8);  Wikipedia
    """
    # 2011-05-27 13:46 IJMC: Created
    # 2011-06-23 13:26 IJMC: Now accepts 2D covariance stepsize inputs.
    # 2011-11-02 22:09 IJMC: Cast numit as int

    import numpy as np

    #Initial setup
    numaccept  = 0
    nout = numit/nstep
    params     = np.copy(params)
    #params[0] = 1./(1.+params[1:14]).prod() - 1.
    bestp      = np.copy(params)
    original_params = np.copy(params)
    numit = int(numit)

    # Set indicated parameters to be positive definite:
    if posdef=='all':
        params = np.abs(params)
        posdef = np.arange(params.size)
    elif posdef is not None:
        posdef = np.array(posdef)
        params[posdef] = np.abs(params[posdef])
    else:
        posdef = np.zeros(params.size, dtype=bool)

    # Set indicated parameters to be held fixed:
    if holdfixed is not None:
        holdfixed = np.array(holdfixed)
        params[holdfixed] = np.abs(params[holdfixed])
    else:
        holdfixed = np.zeros(params.size, dtype=bool)

    weights = 1./sigma**2
    allparams  = np.zeros((len(params), nout))
    allchi     = np.zeros(nout,float)

    #Calc chi-squared for model type using current params
    zmodel = modeltransit14(params, func, per, t)
    currchisq  = (((zmodel - flux)**2)*weights).ravel().sum()
    bestchisq  = currchisq
    print("zmodel [0,1,2]=", zmodel.ravel()[0:3])
    print("Initial chisq is %5.1f" % currchisq)

#Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for j in range(numit):
    #Take step in random direction for adjustable parameters
            if len(stepsize.shape)==1:
                nextp    = np.random.normal(params,stepsize)
            else:
                nextp = np.random.multivariate_normal(params, stepsize)

            nextp[posdef] = np.abs(nextp[posdef])
            nextp[holdfixed] = original_params[holdfixed]
            #nextp[0] = 1./(1. + nextp[1:14]).prod() - 1.
            #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
            zmodel     = modeltransit14(nextp, func, per, t)

            nextchisq  = (((zmodel - flux)**2)*weights).ravel().sum()

            accept = np.exp(0.5 * (currchisq - nextchisq))
            if (accept >= 1) or (np.random.uniform(0, 1) <= accept):
                    #Accept step
                    numaccept += 1
                    params  = np.copy(nextp)
                    currchisq  = nextchisq
            if (currchisq < bestchisq):
                            #New best fit
                            bestp     = np.copy(params)
                            bestchisq = currchisq

            if (j%nstep)==0:
                allparams[:, j/nstep] = params
                allchi[j/nstep] = currchisq
    return allparams, bestp, numaccept, allchi

def mcmc_eclipse(flux, sigma, t, func, params, tparams, stepsize, numit, nstep=1, posdef=None, holdfixed=None):
    """MCMC for 3-parameter eclipse function with KNOWN orbit

    :INPUTS:
        flux : 1D array
                    Contains dependent data

        sigma : 1D array
                    Contains standard deviation (uncertainties) of flux data

        t : 1D array
                    Contains independent data: timing info

        func : function
                    Function to model eclipse (e.g., :func:`transit.occultuniform`)

        params : parameters to be fit:
          EITHER:
             [T_center, depth, Fstar]
          OR:
             [c0, ..., c13, T_center, depth, Fstar]

        params : 4 KNOWN, CONSTANT orbital parameters
          [b, Rstar/a, Rp/Rstar, period]

        stepsize :  1D array
                Array of 1-sigma change in parameter per iteration

        numit : int
                Number of iterations to perform

        nstep : int
                Saves every "nth" step of the chain

        posdef : None, 'all', or sequences of indices.
                Which elements should be restricted to positive definite?
                If indices, it should be of the form (e.g.): [0, 1, 4]

        holdfixed : None, or sequences of indices.
                    Which elements should be held fixed in the analysis?
                    If indices, it should be of the form (e.g.): [0, 1, 4]

    :RETURNS:
        allparams : 2D array
                Contains all parameters at each step

        bestp : 1D array
                Contains best paramters as determined by lowest Chi^2

        numaccept: int
                Number of accepted steps

        chisq: 1D array
                Chi-squared value at each step

    :REFERENCES:
        Numerical Recipes, 3rd Edition (Section 15.8);  Wikipedia
    """
    # 2011-05-31 10:48 IJMC: Created from mcmc_transit
    # 2011-11-02 17:14 IJMC: Now cast numit as int

    import numpy as np

    #Initial setup
    if len(params) > 14:
        modelfunc = modeleclipse_simple14
    else:
        modelfunc = modeleclipse_simple

    numaccept  = 0
    nout = numit/nstep
    bestp      = np.copy(params)
    params     = np.copy(params)
    original_params = np.copy(params)
    numit = int(numit)

    # Set indicated parameters to be positive definite:
    if posdef=='all':
        params = np.abs(params)
        posdef = np.arange(params.size)
    elif posdef is not None:
        posdef = np.array(posdef)
        params[posdef] = np.abs(params[posdef])
    else:
        posdef = np.zeros(params.size, dtype=bool)

    # Set indicated parameters to be positive definite:
    if holdfixed is not None:
        holdfixed = np.array(holdfixed)
        params[holdfixed] = np.abs(params[holdfixed])
    else:
        holdfixed = np.zeros(params.size, dtype=bool)

    weights = 1./sigma**2
    allparams  = np.zeros((len(params), nout))
    allchi     = np.zeros(nout,float)

    #Calc chi-squared for model type using current params
    zmodel = modelfunc(params, tparams, func, t)
    currchisq  = (((zmodel - flux)**2)*weights).ravel().sum()
    bestchisq  = currchisq

#Run Metropolis-Hastings Monte Carlo algorithm 'numit' times
    for j in range(numit):
    #Take step in random direction for adjustable parameters
            nextp    = np.random.normal(params,stepsize)
            nextp[posdef] = np.abs(nextp[posdef])
            nextp[holdfixed] = original_params[holdfixed]
            #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
            zmodel     = modelfunc(nextp, tparams, func, t)

            nextchisq  = (((zmodel - flux)**2)*weights).ravel().sum()

            accept = np.exp(0.5 * (currchisq - nextchisq))
            if (accept >= 1) or (np.random.uniform(0, 1) <= accept):
                    #Accept step
                    numaccept += 1
                    params  = np.copy(nextp)
                    currchisq  = nextchisq
            if (currchisq < bestchisq):
                            #New best fit
                            bestp     = np.copy(params)
                            bestchisq = currchisq

            if (j%nstep)==0:
                allparams[:, j/nstep] = params
                allchi[j/nstep] = currchisq
    return allparams, bestp, numaccept, allchi


#def t14(per, ars, p0,
#
    #t14 = (per/np.pi) * np.arcsin(ra * np.sqrt((1. + k**2) - b**2)/np.sin(np.arccos(cosi)))
    #t23 = (per/np.pi) * np.arcsin(ra * np.sqrt((1. - k**2) - b**2)/np.sin(np.arccos(cosi)))

def fiteclipse(data, sv, ords, tlc, edata=None, index=None, dotransit=True, dopb=True):
    """data: time series to fit using least-squares.

      sv:  state vectors (e.g., various instrumental parameters)

      ords: orders to raise each sv vector to: e.g., [1, [1,2], 3]

      tlc:  eclipse light curve

      edata: error on the data (for chisq ONLY! No weighted fits.)

      index: array index to apply to data, sv, and tlc

      dopb: do prayer-bead uncertainty analysis

      dotransit: include tlc in the fitting; otherwise, leave it out.
      """
    # 2012-01-05 11:25 IJMC: Created

    import analysis as an

    #Simple prayer-bead analysis routine (using matrix multiplication):
    def pbanal(data, xmatrix):
        nobs, ncoef = xmatrix.shape
        solns = np.zeros((nobs, ncoef), float)
        solns[0] = np.dot(np.linalg.pinv(xmatrix), data)
        model = np.dot(xmatrix, solns[0])
        residual = data - model
        for ii in range(1, nobs):
            fakedata = model + np.concatenate((residual[ii::], residual[0:ii]))
            solns[ii] = np.dot(np.linalg.pinv(xmatrix), fakedata)
        return solns



    nobs = len(data)
    if sv is None:
        sv = np.ones((0, nobs))
    else:
        sv = np.array(sv, copy=False)
    if sv.size>0 and sv.size==sv.shape[0]:
        sv = sv.reshape(1, len(sv))
    nsv = sv.shape[0]
    if index is None:
        index = np.ones(nobs, dtype=bool)
    else:
        index = np.array(index, copy=False)

    if edata is None:
        edata = np.ones(nobs)
    elif not hasattr(edata, '__iter__'):
        edata = np.tile(edata, nobs)
    else:
        edata = np.array(edata, copy=False)
    if len(edata.shape)==1:
        weights = np.diag(1./edata**2)
    elif len(edata.shape)==2:
        weights = 1./edata**2

    xmat = np.ones((1, nobs), float)
    if dotransit:
        xmat = np.vstack((xmat, tlc))
    for jj in range(nsv):
        if hasattr(ords[jj], '__iter__'):
            for ord in ords[jj]:
                xmat = np.vstack((xmat, sv[jj]**ord))
        else:
            xmat = np.vstack((xmat, sv[jj]**ords[jj]))

    xmat = xmat.transpose()
    nparam = xmat.shape[1]
    prayerbead = pbanal(np.log(data[index]), xmat[index])
    if dotransit:
        depth = prayerbead[0,1]
        udepth = an.dumbconf(prayerbead[:, 1], .683, type='central', mid=prayerbead[0,1])[0]
    else:
        edepth, udepth = 0., 0.
    model = np.exp(np.dot(xmat, prayerbead[0,:]))
    mods = np.exp(np.array([(np.dot(xmat, prayerbead[ii,:])) for ii in range(index.sum())]))
    chisq = ((np.diag(weights) * (data - model)**2)[index]).sum()
    bic   = chisq + nparam * np.log(index.sum())
    return (depth, udepth), (chisq, bic), prayerbead, model, mods

def analyzetransit_general(params, NL, NP, time, data, weights=None, dopb=False, domcmc=False, gaussprior=None, ngaussprior=None, nsigma=4, maxiter=10, parinfo=None, nthread=1, nstep=2000, nwalker_factor=20, xtol=1e-12, ftol=1e-10, errscale=1e6, smallplanet=True, svs=None, verbose=False):
    """
    Fit transit to data, and estimate uncertainties on the fit.

    :INPUTS:
     params : sequence
       A guess at the best-fit model transit parameters, to be passed
       to :func:`modeltransit_general`.

       If 'svs' are passed in (see below), then params should have an
       additional value concatenated on the end as the coefficient for
       each state vector.


     NL : int
       number of limb-darkening parameters (cf. :func:`modeltransit_general`)

     NP : int
       number of normalizing polynomial coefficients
       (cf. :func:`modeltransit_general`)

     time : sequence
       time values (e.g., BJD_TDB - BJD_0)

     data : sequence
       photometric values (i.e., the transit light curve) to be fit to.

     weights : sequence
       weights to the photometric values.  If None, weights will be
       set equal to the inverse square of the residuals to the
       best-fit model.  In either case, extreme outliers will be
       de-weighted in the fitting process.  This will not change the
       values of the input 'weights'.

     nsigma : scalar
       Residuals beyond this value of sigma-clipped standard
       deviations will be de-weighted.

     dopb : bool
       If True, run prayer-bead (residual permutation) error analysis.

     domcmc : bool
       If True, run Markov Chain Monte Carlo error analysis (requires EmCee)

     nstep : int
       Number of steps for EmCee MCMC run.  This should be *at least*
       several thousand.

     errscale: scalar
       See :func:`modeltransit_general`

     smallplanet: bool
       See :func:`modeltransit_general`

     svs : None, or sequence of 1D NumPy arrays
       State vectors, for additional decorrelation of data in a
       least-squares sense. See :func:`modeltransit_general`

    :OUTPUTS:
      (eventually, some object with useful fields)

    :SEE_ALSO:
       :func:`modeltransit_general`
    """
    # 2012-05-03 11:42 IJMC: Created
    # 2012-05-08 16:59 IJMC: Added ngaussprior option; NL can be negative.
    # 2013-04-01 09:38 IJMC: Added 'svs' option; rejiggered errscale, smallplanet
    # 2013-04-18 12:16 IJMC: Made this a bit smarter; now if MCMC
    #                        sampler finds a new best fit,
    #                        optimization is re-run and the sampler is
    #                        restarted.

    import emcee
    #from kapteyn import kmpfit
    #import transit
    from scipy import optimize
    import Crossfield_analysis as an
    import Crossfield_phasecurves as pc

    # Parse inputs:
    bestparams = np.array(params, copy=True)
    nparams = len(params)

    if parinfo is None:
        parinfo = [None] * nparams

    if gaussprior is None:
        gaussprior = [None] * nparams

    fitkw = dict(gaussprior=gaussprior, ngaussprior=ngaussprior)

    nobs = len(data)

    # Run through an initial fitting routine, to flag and de-weight outliers:
    goodind = np.isfinite(data)
    prev_ngood = goodind.sum()
    newBadPixels = True
    niter = 0
    if weights is None:
        weights = np.ones(nobs) / data[goodind].var()
        weights[True - goodind] = 1e-18
        scaleWeights = True
    else:
        weights = np.array(weights, copy=True)
        scaleWeights = False

    while newBadPixels and niter <= maxiter:
        fitargs = (transit.modeltransit_general, time, NL, NP, errscale, smallplanet, svs, data, weights, fitkw)
        mod = transit.modeltransit_general(bestparams, *fitargs[1:-3])

        # If using scipy.optimize:
        lsq_fit = optimize.leastsq(pc.devfunc, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
        bestparams = lsq_fit[0]
        covar = lsq_fit[1]
        bestchisq = pc.errfunc(bestparams, *fitargs)

        if covar is not None:
            p0 = np.random.multivariate_normal(bestparams, 4*covar, 50)
            testfits = [optimize.leastsq(pc.devfunc, pp, args=fitargs, full_output=True, xtol=xtol, ftol=ftol) for pp in p0]
            testchi = [pc.errfunc(thisfit[0], *fitargs) for thisfit in testfits]
            if min(testchi) < bestchisq:
                if verbose: print("Optimizer found a better fit...")
                bestchisq = min(testchi)
                bestparams = testfits[testchi.index(bestchisq)][0]
                covar = testfits[testchi.index(bestchisq)][1]
                niter -= 1

        # If using kapteyn.kmpfit:
        #fitter = kmpfit.Fitter(residuals=pc.devfunc, data=fitargs)
        #fitter.parinfo = parinfo
        #fitter.fit(params0=params)
        #params = fitter.params
        #bestchisq = fitter.chi2_min


        model = transit.modeltransit_general(bestparams, *fitargs[1:-3])
        residuals = data - model

        goodind = goodind * (True - (np.abs(residuals / an.stdr(residuals[goodind], nsigma=4)) > 4))
        ngood = goodind.sum()
        if scaleWeights:
            weights = np.ones(nobs) / residuals[goodind].var()
        weights[True - goodind] = 1e-18
        if ngood==prev_ngood:
            newBadPixels = False
        else:
            newBadPixels = True
            prev_ngood = ngood
        niter += 1

    # Redefine fitting arguments, with outliers de-weighted:
    fitargs = (transit.modeltransit_general, time, NL, NP, errscale, smallplanet, svs, data, weights, fitkw)

    # Now run a prayer-bead analysis.
    if dopb:
        print("Starting prayer-bead analysis")
        pb_fits = an.prayerbead(params, *fitargs, parinfo=parinfo, xtol=xtol)
        #pb_eparams = np.array([an.dumbconf(pb_fits[:,ii], .683, mid=bestparams[ii])[0] for ii in range(len(bestparams))])
        #pbchisq = [pc.errfunc(par0, *fitargs[:-1]) for par0 in pb_fits]
        #pbmodels = [fitargs[0](par0, *fitargs[1:-3]) for par0 in pb_fits]

        bestparams = pb_fits[0].copy()

    else:
        pb_fits = None

    # Now run an MCMC analysis:
    if domcmc:
        print("Starting MCMC analysis")
        ndim = len(params)
        nwalkers = nwalker_factor * ndim

        # Initialize sampler:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, pc.lnprobfunc, args=fitargs, threads=nthread)
        if covar is None and dopb: # leastsq didn't work as advertised!
            covar = np.cov(pb_fits.transpose())
        elif covar is None: # ... and we don't even have a good estimator!
            covar = np.zeros((nparams, nparams), dtype=float)
            for ii in range(nparams):
                if gaussprior[ii] is None:
                    covar[ii,ii] = (bestparams[ii]/100.)**2
                else:
                    covar[ii,ii] = gaussprior[ii][1]**2


        # Pick walker starting positions, excluding bad points:
        p0 = np.random.multivariate_normal(bestparams, covar, nwalkers*5)
        badp0 = ((p0[:,1] < 0) + (p0[:,2] < 0) + (p0[:,3] < 0) + (p0[:,4] < 0) + p0[:,2] > 90)
        p0 = np.vstack((bestparams, p0[np.nonzero(True-badp0)[0][0:nwalkers-1]]))

        # Run burn-in; exclude and remove bad walkers:
        pos = p0
        for ii in range(2):
            pos, prob, state = sampler.run_mcmc(pos, max(1, (ii+1)*nstep/5))
            if (bestchisq + 2*max(prob)) > ftol: #-2*prob < bestchisq).any(): # Found a better fit! Optimize:
                if verbose: print("Found a better fit; re-starting MCMC... (%1.8f, %1.8f)" % (bestchisq, (bestchisq + 2*max(prob))))
                #pdb.set_trace()
                bestparams = pos[(prob==prob.max()).nonzero()[0][0]].copy()
                lsq_fit = optimize.leastsq(pc.devfunc, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=ftol)
                bestparams = lsq_fit[0]
                covar = lsq_fit[1]
                bestchisq = pc.errfunc(bestparams, *fitargs)
                ii = 0 # Re-start the burn-in.
                #pos[prob < np.median(prob)] = bestparams

            pos[pos[:,2]>90,2] = pos[pos[:,2]>90,2] % 360.
            pos[pos[:,2]>90,2] = 180 - pos[pos[:,2]>90,2] # reset inclination
            badpos = (prob < np.median(prob))
            if (pos[True-badpos].std(0) <= 0).any():
                goodpos_unc = np.vstack((np.abs(bestparams / 100.), pos[True-badpos].std(0))).max(0)
                pos[badpos] = np.random.normal(bestparams, goodpos_unc, (badpos.sum(), ndim))
            else:
                pos[badpos]  = np.random.multivariate_normal(bestparams, np.cov(pos[True-badpos].transpose()), badpos.sum())

        pos[badpos] = bestparams

        # Run main MCMC run:
        #pdb.set_trace()
        pos, prob, state = sampler.run_mcmc(pos, max(1, (ii+1)*nstep/4))
        sampler.reset()
        steps_taken = 0
        while steps_taken < nstep:
            pos, prob, state = sampler.run_mcmc(pos, 1)
            steps_taken += 1
            if (bestchisq + 2*max(prob)) > ftol: # Found a better fit! Optimize:
                if verbose: print("Found a better fit; re-starting MCMC... (%1.8f, %1.8f)" % (bestchisq, (bestchisq + 2*max(prob))))
                #pdb.set_trace()
                bestparams = pos[(prob==prob.max()).nonzero()[0][0]].copy()
                lsq_fit = optimize.leastsq(pc.devfunc, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=1e-10)
                bestparams = lsq_fit[0]
                covar = lsq_fit[1]
                bestchisq = pc.errfunc(bestparams, *fitargs)
                steps_taken = 0 # Re-start the MCMC
                sampler.reset()
                pos[prob<np.median(prob)] = bestparams

        # Test if MCMC found a better chi^2 region of parameter space:
        mcmc_params = sampler.flatchain[np.nonzero(sampler.lnprobability.ravel()==sampler.lnprobability.ravel().max())[0][0]]
        mcmc_fit = optimize.leastsq(pc.devfunc, mcmc_params, args=fitargs, full_output=True, xtol=xtol, ftol=1e-10)
        if pc.errfunc(mcmc_fit[0], *fitargs) < pc.errfunc(bestparams, *fitargs):
            bestparams = mcmc_fit[0]
        else:
            pass

    else:
        sampler = None

    # Run a second fit w/Kapteyn:
    #fitter2 = kmpfit.Fitter(residuals=pc.devfunc, data=fitargs)
    #fitter2.parinfo = parinfo
    #fitter2.fit(params0=mcmc_params)
    #bestparams = fitter2.params
    #model2 = transit.modeltransit_general(bestparams, *fitargs[1:-2])

    # Run a second fit with scipy.optimize.leastsq:
    lsq_fit = optimize.leastsq(pc.devfunc, bestparams, args=fitargs, full_output=True, xtol=xtol, ftol=1e-10)
    bestparams = lsq_fit[0]
    covar = lsq_fit[1]
    bestchisq = pc.errfunc(bestparams, *fitargs)

    finalmodel = transit.modeltransit_general(bestparams, *fitargs[1:-3])

    return lsq_fit, pb_fits, sampler, weights, finalmodel
