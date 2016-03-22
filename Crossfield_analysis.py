""" My module for various data analysis tasks.

:REQUIREMENTS: :doc:`numpy`, :doc:`tools` (for :func:`errxy`)


2008-07-25 16:20 IJC: Created.

2009-12-08 11:31 IJC: Updated transit flag in planet objects and 
                      :func:`rveph` function.

2010-02-18 14:06 IJC: Added :func:`medianfilter`

2010-08-03 15:38 IJC: Merged versions.

2010-10-28 11:53 IJMC: Updated documentation strings for Sphinx;
                      moved pylab import inside individual
                      function.

2011-04-13 14:29 IJMC: Added keyword functionality to :func:`fmin`
                      (taken from scipy.optimize).

2011-04-14 09:48 IJMC: Added a few fundamental constants.

2011-04-22 14:48 IJMC: Added :func:`trueanomaly` and
                       :func:`eccentricanomaly`.

2011-06-21 13:15 IJMC: Added :func:`get_t23` and :func:`get_t14` to
                       planet objects.
"""


import numpy as np
from numpy import ones, std, sum, mean, median, array, linalg, tile, concatenate, floor, Inf, arange, meshgrid, zeros, sin, cos, tan, arctan, sqrt, exp, nan, max
import pdb
from pylab import find
from scipy import optimize


c   = 299792458  # speed of light, m/s
h = 6.626068e-34 # SI units: Planck's constant
k = 1.3806503e-23 # SI units: Boltzmann constant, J/K
G = 6.67300e-11 # SI units: Gravitational constant
sigma = 5.670373e-8 # SI units: Stefan-Boltzmann constant

qe = 1.602176565e-19 # Electron charge, in Coulombs
me = 9.11e-31  # electron mass, in kg
ev = 1.602176565e-19 # electron Volt, in Joules
amu = 1.66053886e-27 # atomic mass unit, in kg
mh = 1.008 * amu # mass of hydrogen atom, in kg

pi  = 3.14159265358979
AU  = 149597870691.0  # AU in meters
day = 86400.0  # seconds in a Julian Day
rsun = 6.95508e8  # Solar mean radius, in m
msun = 1.9891e30  # solar mass in kg
rearth = 6.378136e6 # Earth's equatorial radius in m; [Allen's]
mearth = 5.9737e24 # in kg; [Allen's]
rjup = 7.1492e7 # Jupiter equatorial radius in m
mjup = 1898.7e24  # Jupiter mass in kg
pc = 3.08568025e16 # parsec in meters

class planet:
    """Very handy planet object.

    Best initialized using :func:`getobj`.

    :REQUIREMENTS: Database file `exoplanets.csv` from http://exoplanets.org/

    """
    # 2010-03-07 22:21 IJC: Created
    # 2011-01-22 17:15 IJMC: Updated format b/c CPS folks keep changing
    #                       their database file format

    # 2011-05-19 16:11 IJMC: Updated format b/c CPS folks keep
    #                       changing their database file format -- but
    #                       it's almost worth it to finally have
    #                       stellar radii.
    def __init__(self, *args):
        keys = ['name','comp','ncomp','mult','discmeth','firstref','firsturl','date','jsname','etdname','per','uper','t0','ut0','ecc','uecc','ueccd','om','uom','k','uk','msini','umsini','a','ua','orbref','orburl','transit','t14','ut14','tt','utt','ar','uar','uard','i','ui','uid','b','ub','ubd','depth','udepth','udepthd','r','ur','density','udensity','gravity','ugravity','transitref','transiturl','trend','dvdt','udvdt','freeze_ecc','rms','chi2','nobs','star','hd','hr','hipp','sao','gl','othername','sptype','binary','v','bmv','j','h','ks','ra','dec','ra_string','dec_string','rstar', 'urstar', 'urstard', 'rstarref', 'rstarurl','mstar','umstar','umstard','teff','uteff','vsini','uvsini','fe','ufe','logg','ulogg','shk','rhk','par','upar','distance','udistance','lambd', 'ulambd', 'massref','massurl','specref','specurl','distref','disturl','simbadname','nstedid','binaryref', 'binaryurl']

        if len(keys)<>len(args):
            print "Incorrect number of input arguments (%i, but should be %i)" % (len(args), len(keys))
            return None

        for key,arg in zip(keys, args):
            try:
                temp = float(arg)+1
                isnumber = True
            except:
                isnumber = False
                

            if isnumber:
                exec('self.%s=%s' % (key,arg) )
            else:
                exec('self.%s="%s"' % (key,arg) )

        return None
        
    def get_t23(self, *args):
        """Compute full transit duration (in days) for a transiting planet.

        Returns:
           nan if required fields are missing.

        Using Eq. 15 of J. Winn's chapter in S. Seager's book "Exoplanets." 

        :SEE ALSO:
          :func:`get_t14`
          """ 
        # 2011-06-21 12:53 IJMC: Created

        # Get necessary parameters:
        per, ra, k, b, inc, ecc, om = self.per, 1./self.ar, sqrt(self.depth), self.b, self.i, self.ecc, self.om

        inc *= np.pi/180.
        om *= np.pi/180.

        ret = 0.

        # Check that no parameters are empty:
        if per=='' or k=='':
            ret = nan

        if b=='':
            try:
                b = (cos(inc)/ra) * (1. - ecc**2) / (1. + ecc * sin(om))
            except:
                ret = nan
        elif ra=='':
            try:
                ra = (cos(inc)/b) * (1. - ecc**2) / (1. + ecc * sin(om))
            except:
                ret = nan
        elif inc=='':
            try:
                inc = np.arccos((b * ra) / ((1. - ecc**2) / (1. + ecc * sin(om))))
            except:
                ret = nan
        
        if np.isnan(ret):
            print "Could not compute t_23.  Parameters are:"
            print "period>>", per
            print "r_*/a>>", ra
            print "r_p/r_*>>", k
            print "impact parameter (b)>>", b
            print "inclination>>", inc*180./np.pi, " deg"
            print "eccentricity>>", ecc
            print "omega>>", om*180./np.pi, " deg"
        else:
            ret = (per/np.pi) * np.arcsin(ra * np.sqrt((1. - k)**2 - b**2) / np.sin(inc))

        return ret
        
    def get_t14(self, *args):
        """Compute total transit duration (in days) for a transiting planet.

        Returns:
           nan if required fields are missing.

        Using Eq. 14 of J. Winn's chapter in S. Seager's book "Exoplanets." 

        :SEE ALSO:
          :func:`get_t23`
          """ 
        # 2011-06-21 12:53 IJMC: Created

        # Get necessary parameters:
        per, ra, k, b, inc, ecc, om = self.per, 1./self.ar, sqrt(self.depth), self.b, self.i, self.ecc, self.om

        inc *= np.pi/180.
        om *= np.pi/180.

        ret = 0.

        # Check that no parameters are empty:
        if per=='' or k=='':
            ret = nan

        if b=='':
            try:
                b = (cos(inc)/ra) * (1. - ecc**2) / (1. + ecc * sin(om))
            except:
                ret = nan
        elif ra=='':
            try:
                ra = (cos(inc)/b) * (1. - ecc**2) / (1. + ecc * sin(om))
            except:
                ret = nan
        elif inc=='':
            try:
                inc = np.arccos((b * ra) / ((1. - ecc**2) / (1. + ecc * sin(om))))
            except:
                ret = nan
        
        if np.isnan(ret):
            print "Could not compute t_14.  Parameters are:"
            print "period>>", per
            print "r_*/a>>", ra
            print "r_p/r_*>>", k
            print "impact parameter (b)>>", b
            print "inclination>>", inc*180./np.pi, " deg"
            print "eccentricity>>", ecc
            print "omega>>", om*180./np.pi, " deg"
        else:
            ret = (per/np.pi) * np.arcsin(ra * np.sqrt((1. + k)**2 - b**2) / np.sin(inc))

        return ret
        
    def get_teq(self, ab, f, reterr=False):
        """Compute equilibrium temperature.

        :INPUTS:
          ab : scalar, 0 <= ab <= 1
            Bond albedo.

          f : scalar, 0.25 <= ab <= 2/3.
            Recirculation efficiency.  A value of 0.25 indicates full
            redistribution of incident heat, while 2/3 indicates zero
            redistribution.

        :EXAMPLE:
           ::

               import analysis
               planet = analysis.getobj('HD 189733 b')
               planet.get_teq(0.0, 0.25) # zero albedo, full recirculation

        :REFERENCE:
           S. Seager, "Exoplanets," 2010.  Eq. 3.9
        """
        # 2012-09-07 16:24 IJMC: Created

        return self.teff/np.sqrt(self.ar) * (f * (1. - ab))**0.25
        
    def rv(self, **kw):
        """Compute radial velocity on a planet object for given Julian Date.

        :EXAMPLE:
           ::

               import analysis
               p = analysis.getobj('HD 189733 b')
               jd = 2400000.
               print p.rv(jd)
        
        refer to function `analysis.rv` for full documentation.

        SEE ALSO: :func:`analysis.rv`, :func:`analysis.mjd`
        """
        return rv(self, **kw)
        
    def rveph(self, jd):
        """Compute the most recently elapsed RV emphemeris of a given
        planet at a given JD.  RV ephemeris is defined by the having
        radial velocity equal to zero.
        
        refer to  :func:`analysis.rv` for full documentation.
        
        SEE ALSO: :func:`analysis.getobj`, :func:`analysis.phase`
        """
        return rveph(self, jd)
        
    def phase(self, hjd):
        """Get phase of an orbiting planet.

        refer to function `analysis.getorbitalphase` for full documentation.

        SEE ALSO: :func:`analysis.getorbitalphase`, :func:`analysis.mjd`
        """
        # 2009-09-28 14:07 IJC: Implemented object-oriented version
        return getorbitalphase(self, hjd)
        
    def writetext(self, filename, **kw):
        """See :func:`analysis.planettext`"""
        
        return planettext(self, filename, **kw)  
        
    
def loaddata(filelist, path='', band=1):
    """Load a set of reduced spectra into a single data file.

   datalist  = ['file1.fits', 'file2.fits']
   datapath = '~/data/'

   data = loaddata(datalist, path=datapath, band=1)


   The input can also be the name of an IRAF-style file list.
   """
    # 2008-07-25 16:26 IJC: Created
    # 2010-01-20 12:58 IJC: Made function work for IRTF low-res data.
    #                       Replaced 'warn' command with 'print'.
    # 2011-04-08 11:42 IJC: Updated; moved inside analysis.py.
    # 2011-04-12 09:57 IJC: Fixed misnamed imports

    import pyfits

    data = array([])

    if filelist.__class__==str or isinstance(filelist,np.string_):
        filelist = ns.file2list(filelist)
    elif filelist.__class__<>list:
        print('Input to loaddata must be a python list or string')
        return data

    num = len(filelist)

    # Load one file just to get the dimensions right.
    irspec = pyfits.getdata(filelist[0])

    ii = 0
    for element in filelist:
        irspec = pyfits.getdata(element)

        if ii==0:
            irsh = irspec.shape
            data  = zeros((num,)+irsh[1::], float)
            
        if len(irsh)>2:
            for jj in range(irsh[1]):
                data[ii, jj, :] = irspec[band-1, jj, :]
        else:
            data[ii,:] = irspec[band-1,:]

        ii=ii+1

    return data

def getobj(*args, **kw):
    """Get data for a specified planet.  

       :INPUTS:  (str) -- planet name, e.g. "HD 189733 b"

       :OPTIONAL INPUTS:  
           datafile : str 
                datafile name

           verbose : bool
                verbosity flag                          

       :EXAMPLE:
           ::

                   p = getobj('55cnce')
                   p.period   ---> 2.81705

       The attributes of the returned object are many and varied, and
       can be listed using the 'dir' command on the returned object.

       This looks up data from the local datafile, which could be out
       of date.

       SEE ALSO: :func:`rv`"""
    # 2008-07-30 16:56 IJC: Created
    # 2010-03-07 22:24 IJC: Updated w/new exoplanets.org data table! 
    # 2010-03-11 10:01 IJC: If planet name not found, return list of
    #                       planet names.  Restructured input format.
    # 2010-11-01 13:30 IJC: Added "import os"
    # 2011-05-19 15:56 IJC: Modified documentation.

    import os

    if kw.has_key('datafile'):
        datafile=kw['datafile']
    else:
        datafile=os.path.expanduser('~/python/exoplanets.csv')
    if kw.has_key('verbose'):
        verbose = kw['verbose']
    else:
        verbose=False

    if len(args)==0:
        inp = 'noplanetname'
    else:
        inp = args[0]

    if verbose:  print "datafile>>" + datafile
    f = open(datafile, 'r')
    data = f.readlines()
    f.close()
    data = data[1::]  # remove header line
    names = array([line.split(',')[0] for line in data])

    foundobj = (names==inp)
    if (not foundobj.any()):
        if verbose: print "could not find desired planet; returning names of known planets"
        return names
    else:
        planetind = int(find(foundobj))
        pinfo  = data[planetind].strip().split(',')
        myplanet = planet(*pinfo)
        
    return myplanet

def getorbitalphase(planet, hjd, **kw):
    """Get phase of an orbiting planet.
    
    INPUT: 
       planet  -- a planet from getobj; e.g., getobj('55cnce')
       hjd

    OUTPUT:
       orbital phase -- from 0 to 1.

    NOTES: 
       If planet.transit==True, phase is based on the transit time ephemeris.  
       If planet.transit==False, phase is based on the RV ephemeris as
          computed by function rveph
    

    SEE ALSO:  :func:`getobj`, :func:`mjd`, :func:`rveph`
    """

    hjd = array(hjd).copy()
    if bool(planet.transit)==True:
        thiseph = planet.tt
    else:
        thiseph = planet.rveph(hjd.max())

    orbphase = ((hjd - thiseph) ) / planet.per
    orbphase -= int(orbphase.mean())

    return orbphase
    
def mjd(date):
    """Convert Julian Date to Modified Julian Date, or vice versa.

    if date>=2400000.5, add 2400000.5
    if date<2400000.5, subtract 2400000.5
    """
    # 2009-09-24 09:54 IJC: Created 

    date = array(date, dtype=float, copy=True, subok=True)
    offset = 2400000.5
    if (date<offset).all():
        date += offset
    elif (date>=offset).all():
        date -= offset
    else:
        print "Input contains date both below and above %f" % offset

    return date

def rveph(p, jd):
    """Compute the most recently elapsed RV emphemeris of a given
    planet at a given JD.  RV ephemeris is defined by the having
    radial velocity equal to zero.



        :EXAMPLE:
            ::

                from analysis import getobj, rveph
                jd = 2454693            # date: 2008/08/14
                p  = getobj('55cnce')   # planet: 55 Cnc e
                t = rveph(p, jd)
            
            returns t ~ 


        SEE ALSO: :func:`getobj`, :func:`phase`
        """
    # 2009-12-08 11:20 IJC: Created.  Ref: Beauge et al. 2008 in
    #    "Extrasolar Planets," R. Dvorak ed.
    # 2010-03-12 09:34 IJC: Updated for new planet-style object.

    from numpy import cos, arccos, arctan, sqrt, tan, pi, sin, int

    if p.__class__<>planet:
        raise Exception, "First input must be a 'planet' object."
    
    omega = p.om*pi/180 
    tau = p.t0
    ecc = p.ecc
    per = p.per
    f = arccos(-ecc * cos(omega)) - omega  # true anomaly
    u = 2.*arctan(sqrt((1-ecc)/(1+ecc))*tan(f/2.)) # eccentric anomaly
    n = 2*pi/per

    time0 = tau+ (u-ecc*sin(u))/n
    norb = int((time0-jd)/per)
    time = time0-norb*per
    return time
    
def rv(p, jd=None, e=None, reteanom=False, tol=1e-8):
    """ Compute unprojected astrocentric RV of a planet for a given JD in m/s.
    
        :INPUTS:
          p : planet object, or 5- or 6-sequence
            planet object: see :func:`get_obj`

           OR:

            sequence: [period, t_peri, ecc, a, long_peri, gamma]
                      (if gamma is omitted, it is set to zero)
                      (long_peri should be in radians!)

          jd : NumPy array
            Dates of observation (in same time system as t_peri).

          e : NumPy array
            Eccentric Anomaly of observations (can be precomputed to
            save time)

        :EXAMPLE:
            ::

                jd = 2454693            # date: 2008/08/14
                p  = getobj('55 Cnc e')   # planet: 55 Cnc e
                vp = rv(p, jd)
            
            returns vp ~ 1.47e5  [m/s]

        The result will need to be multiplied by the sine of the
        inclination angle (i.e., "sin i").  Positive radial velocities
        are directed _AWAY_ from the observer.

        To compute the barycentric radial velocity of the host star,
        scale the returned values by the mass ratio -m/(m+M).

        SEE ALSO: :func:`getobj`, :func:`rvstar`
    """
    # 2008-08-13 12:59 IJC: Created with help from Debra Fischer,
    #    Murray & Dermott, and Beauge et al. 2008 in "Extrasolar
    #    Planets," R. Dvorak ed.
    # 2008-09-25 12:55 IJC: Updated documentation to be more clear.
    # 2009-10-01 10:25 IJC: Moved "import" statement within func.
    # 2010-03-12 09:13 IJC: Updated w/new planet-type objects
    # 2012-10-15 17:20 IJMC: First input can now be a list of
    #                        elements.  Added option to pass in
    #                        eccentric anomaly.
    
    jd = array(jd, copy=True, subok=True)
    if jd.shape==():  
        singlevalueinput = True
        jd = array([jd])
    else:
        singlevalueinput = False
    if ((jd-2454692) > 5000).any():
        raise Exception, "Implausible Julian Date entered."

    if hasattr(p, '__iter__'):
        if len(p)==5:
            per, tau, ecc, a, omega = p
            gamma = 0.
        else:
            per, tau, ecc, a, omega, gamma = p[0:6]
        if ecc > 1:  # Reset eccentricity directly
            ecc = 1. - tol
            p[2] = ecc
        elif ecc < 0:
            ecc = np.abs(ecc)
            p[2] = ecc
    else:
        if p.__class__<>planet:
            raise Exception, "First input must be a 'planet' object."
        try:
            per = p.per
            tau = p.t0
            ecc = p.ecc
            a   = p.a
            omega = p.om * pi/180.0
        except:
            raise Exception, "Could not load all desired planet parameters."

    if ecc < 0:
        ecc = np.abs(ecc)
    if ecc > 1:
        ecc = 1. - tol

    if e is None:
        n = 2.*pi/per    # mean motion
        m = n*(jd - tau) # mean anomaly
        e = []
        for element in m: # compute eccentric anomaly
            def kep(e): return element - e + ecc*sin(e)
            e.append(optimize.brentq(kep, element-1, element+1,  xtol=tol, disp=False))
            #e.append(optimize.newton(kep, 0, tol=tol))
    else:
        pass

    e = array(e, copy=False)
    f = 2. * arctan(  sqrt((1+ecc)/(1.-ecc)) * tan(e/2.)  )
    #r = a*(1-ecc*cos(e))
    #x = r*cos(f)
    #y = r*sin(f)

    K = n * a / sqrt(1-ecc**2)

    vz = -K * ( cos(f+omega) + ecc*cos(omega) )
    vzms = vz * AU/day  # convert to m/s

    if singlevalueinput:
        vzms = vzms[0]

    if reteanom:
        ret = vzms, e
    else:
        ret = vzms
    return ret
    
def rvstar(p, jd=None, e=None, reteanom=False, tol=1e-8):
    """ Compute radial velocity of a star which has an orbiting planet.

        :INPUTS:
          p : planet object, or 5- or 6-sequence
            planet object: see :func:`get_obj`

           OR:

            sequence: [period, t_peri, ecc, K, long_peri, gamma]
                      (if gamma is omitted, it is set to zero)

          jd : NumPy array
            Dates of observation (in same time system as t_peri).

          e : NumPy array
            Eccentric Anomaly of observations (can be precomputed to
            save time)

        :EXAMPLE:
            ::

                jd = 2454693            # date: 2008/08/14
                p  = getobj('55 Cnc e')   # planet: 55 Cnc e
                vp = rv(p, jd)

          Positive radial velocities are directed _AWAY_ from the
          observer.

        :SEE_ALSO: :func:`rv`, :func:`getobj`
    """
    # 2012-10-15 22:34 IJMC: Created from function 'rv'

    jd = array(jd, copy=True, subok=True)
    if jd.shape==():  
        singlevalueinput = True
        jd = array([jd])
    else:
        singlevalueinput = False
    if ((jd-2454692) > 5000).any():
        raise Exception, "Implausible Julian Date entered."

    if hasattr(p, '__iter__'):
        if len(p)==5:
            per, tau, ecc, k, omega = p
            gamma = 0.
        else:
            per, tau, ecc, k, omega, gamma = p[0:6]
        omega *= np.pi/180.
        if ecc < 0:
            ecc = np.abs(ecc)
            p[2] = ecc
        elif ecc > 1:
            ecc = 1. - tol
            p[2] = ecc

    else:
        if p.__class__<>planet:
            raise Exception, "First input must be a 'planet' object."
        try:
            per = p.per
            tau = p.t0
            ecc = p.ecc
            k   = p.k
            omega = p.om * pi/180.0
            gamma = 0.
        except:
            raise Exception, "Could not load all desired planet parameters."

    if ecc < 0:
        ecc = np.abs(ecc)
    if ecc > 1:
        ecc = 1. - tol

    if e is None:
        n = 2.*pi/per    # mean motion
        m = n*(jd - tau) # mean anomaly
        e = np.zeros(m.shape)
        for ii,element in enumerate(m): # compute eccentric anomaly
            def kep(e): return element - e + ecc*sin(e)
            e[ii] = optimize.brentq(kep, element-1, element+1,  xtol=tol, disp=False)
            #e.append(optimize.newton(kep, 0, tol=tol))
    else:
        pass

    # Compute true anomaly:
    f = 2. * arctan(  sqrt((1+ecc)/(1.-ecc)) * tan(e/2.)  )
    
    vrstar = k * (np.cos(f + omega) + ecc*np.cos(omega)) + gamma

    if singlevalueinput:
        vrstar = vrstar[0]

    if reteanom:
        ret = vrstar, e
    else:
        ret = vrstar

    return ret

def dopspec(starspec, planetspec, starrv, planetrv, disp, starphase=[], planetphase=[], wlscale=True):
    """ Generate combined time series spectra using planet and star
    models, planet and star RV profiles.

    D = dopspec(sspec, pspec, sRV, pRV, disp, sphase=[], pphase=[])

    :INPUTS:
       sspec, pspec:  star, planet spectra; must be on a common
                         logarithmic wavelength grid 
       sRV, pRV:      star, planet radial velocities in m/s
       disp:          constant logarithmic dispersion of the wavelength 
                         grid: LAMBDA_i/LAMBDA_(i-1)

    :OPTIONAL INPUTS:
       sphase, pphase:  normalized phase functions of star and planet.  
                           The inputs sspec and pspec will be scaled
                           by these values for each observation.
       wlscale:         return relative wavelength scale for new data

    NOTE: Input spectra must be linearly spaced in log wavelength and increasing:
            that is, they must have [lambda_i / lambda_(i-1)] = disp =
            constant > 1
          Positive velocities are directed AWAY from the observer."""

#2008-08-19 16:30 IJC: Created

# Options: 1. chop off ends or not?  2. phase function. 
#   4. Noise level 5. telluric? 6. hold star RV constant


# Initialize:
    starspec   = array(starspec  ).ravel()
    planetspec = array(planetspec).ravel()
    starrv     = array(starrv    ).ravel()
    planetrv   = array(planetrv  ).ravel()

    ns = len(starspec)
    nr = len(starrv)

    starphase   = array(starphase  ).ravel()
    planetphase = array(planetphase).ravel()
    if len(starphase)==0:
        starphase = ones(nr, float)
    if len(planetphase)==0:
        planetphase = ones(nr, float)
    

# Check inputs: 
    if ns<>len(planetspec):
        raise Exception, "Star and planet spectra must be same length."
    if nr<>len(planetrv):
        raise Exception, "Star and planet RV profiles must be same length."

    logdisp = log(disp)

# Calculate wavelength shift limits for each RV point
    sshift = ( log(1.0+starrv  /c) / logdisp ).round() 
    pshift = ( log(1.0+planetrv/c) / logdisp ).round() 

    limlo =  int( concatenate((sshift, pshift)).min() )
    limhi =  int( concatenate((sshift, pshift)).max() )

    ns2 = ns + (limhi - limlo)

    data = zeros((nr, ns2), float)

# Iterate over RV values, constructing spectra
    for ii in range(nr):
        data[ii, (sshift[ii]-limlo):(ns+sshift[ii]-limlo)] = \
            starphase[ii] * starspec
        data[ii, (pshift[ii]-limlo):(ns+pshift[ii]-limlo)] = \
            data[ii, (pshift[ii]-limlo):(ns+pshift[ii]-limlo)] + \
            planetphase[ii] * planetspec

    if wlscale:
        data = (data, disp**(arange(ns2) + limlo))
    return data

def loadatran(filename, wl=True, verbose=False):
    """ Load ATRAN atmospheric transmission data file.

        t = loadatran(filename, wl=True)

        INPUT: 
           filename -- filename of the ATRAN file.  This should be an
                       ASCII array where the second column is
                       wavelength and the third is the atmospheric
                       transmission.
                     (This can also be a list of filenames!)  
           
        :OPTIONAL INPUT: 
           wl -- if True (DEFAULT) also return the wavelength scale.
                 This can take up to twice as long for large files.

        RETURNS:
           if wl==True:   returns a 2D array, with columns [lambda, transmission]
           if wl==False:  returns a 1D Numpy array of the transmission
           
           NOTE: A list of these return values is created if
                 'filename' is actually an input list."""
    # 2008-08-21 09:42 IJC: Created to save myself a bit of time
    # 2008-08-25 10:08 IJC: Read in all lines at once; should go
    #                        faster with sufficient memory
    # 2008-09-09 13:56 IJC: Only convert the wavelength and flux
    #                        columns(#1 & #2) -- speed up slightly.
    if filename.__class__==list:
        returnlist = []
        for element in filename:
            returnlist.append(loadatran(element, wl=wl))
        return returnlist
    
    f = open(filename, 'r')
    dat = f.readlines()
    f.close()
    if verbose:
        print dat[0]
        print dat[0].split()
        print dat[0].split()[1:3]
        print dat[0].split()[2]
    
    if wl:
        data = array([map(float, line.split()[1:3]) for line in dat])
    else:
        data = array([float(line.split()[2]) for line in dat])

def poly2cheby(cin):
    """Convert straight monomial coefficients to chebychev coefficients.

    INPUT: poly coefficients (e.g., for use w/polyval)
    OUTPUT: chebyt coefficients

    SEE ALSO: :func:`gpolyval`; scipy.special.chebyt
    """
    # 2009-07-07 09:41 IJC: Created
    from scipy.special import poly1d, chebyt
    
    cin = poly1d(cin)
    cout = []

    ord = cin.order
    for ii in range(ord+1):
        chebyii = chebyt(ord-ii)
        cout.append(cin.coeffs[0]/chebyii.coeffs[0])
        cin -= chebyii*cout[ii]

    return cout
    
def cheby2poly(cin):
    """Convert  chebychev coefficients to 'normal' polyval coefficients .

    :INPUT: chebyt coefficients 

    :OUTPUT: poly coefficients (e.g., for use w/polyval)

    :SEE ALSO: :func:`poly2cheby`, :func:`gpolyval`; scipy.special.chebyt
    """
    # 2009-10-22 22:19 IJC: Created
    from scipy.special import poly1d, chebyt
    
    cin = poly1d(cin)
    cout = poly1d(0)

    ord = cin.order
    for ii in range(ord+1):
        cout += chebyt(ii)*cin[ii]

    return cout

def gpolyval(c,x, mode='poly', retp=False):
    """Generic polynomial evaluator.

    INPUT:
            c (1D array) -- coefficients of polynomial to evaluate,
                          from highest-order to lowest.
            x (1D array) -- pixel values at which to evaluate C

    OPTINAL INPUTS:
            MODE='poly' -- 'poly' uses standard monomial coefficients
                            as accepted by, e.g., polyval.  Other
                            modes -- 'cheby' (1st kind) and 'legendre'
                            (P_n) -- convert input 'x' to a normalized
                            [-1,+1] domain
            RETP=False  -- Return coefficients as well as evaluated poly.

    OUTPUT:
            y -- array of shape X; evaluated polynomial.  
            (y, p)  (if retp=True)

    SEE ALSO:  :func:`poly2cheby`

     """
    # 2009-06-17 15:42 IJC: Created 
    # 2011-12-29 23:11 IJMC: Fixed normalization of the input 'x' array
    from scipy import special
    from numpy import zeros, polyval, concatenate


    c = array(c).copy()
    nc = len(c)

    if mode=='poly':
        totalcoef = c
        ret = polyval(totalcoef, x)
    elif mode=='cheby':
        x = 2. * (x - 0.5*(x.max() + x.min())) / (x.max() - x.min())
        totalcoef = special.poly1d([0])
        for ii in range(nc):
            totalcoef += c[ii]*special.chebyt(nc-ii-1)
        ret = polyval(totalcoef, x)
    elif mode=='legendre':
        x = 2. * (x - 0.5*(x.max() + x.min())) / (x.max() - x.min())
        totalcoef = special.poly1d([0])
        for ii in range(nc):
            totalcoef += c[ii]*special.legendre(nc-ii-1)
        ret = polyval(totalcoef, x)

    if retp:
        return (ret, totalcoef)
    else:
        return ret
    return -1

def stdr(x, nsigma=3, niter=Inf, finite=True, verbose=False, axis=None):
    """Return the standard deviation of an array after removing outliers.
    
    :INPUTS:
      x -- (array) data set to find std of

    :OPTIONAL INPUT:
      nsigma -- (float) number of standard deviations for clipping
      niter -- number of iterations.
      finite -- if True, remove all non-finite elements (e.g. Inf, NaN)
      axis -- (int) axis along which to compute the mean.

    :EXAMPLE:
      ::

          from numpy import *
          from analysis import stdr
          x = concatenate((randn(200),[1000]))
          print std(x), stdr(x, nsigma=3)
          x = concatenate((x,[nan,inf]))
          print std(x), stdr(x, nsigma=3)

    SEE ALSO: :func:`meanr`, :func:`medianr`, :func:`removeoutliers`, 
              :func:`numpy.isfinite`
    """
    # 2010-02-16 14:57 IJC: Created from mear
    # 2010-07-01 14:06 IJC: ADded support for higher dimensions
    from numpy import array, isfinite, zeros, swapaxes

    x = array(x, copy=True)
    xshape = x.shape
    ndim =  len(xshape)
    if ndim==0:
        return x

    if  ndim==1 or axis is None:
        # "1D" array
        x = x.ravel()
        if finite:
            x = x[isfinite(x)]
        x = removeoutliers(x, nsigma, niter=Inf, verbose=verbose)
        return x.std()
    else:
        newshape = list(xshape)
        oldDimension = newshape.pop(axis) 
        ret = zeros(newshape, float)

        # Prevent us from taking the action along the axis of primary incidices:
        if axis==0: 
            x=swapaxes(x,0,1)

        if axis>1:
            nextaxis = axis-1
        else:
            nextaxis = 0

        for ii in range(newshape[0]):
            #print 'x.shape>>',x.shape, 'newshape>>',newshape, 'x[ii].shape>>',x[ii].shape, 'ret[ii].shape>>',ret[ii].shape,'ii>>',ii
            ret[ii] = stdr(x[ii], nsigma=nsigma,niter=niter,finite=finite,\
                                 verbose=verbose, axis=nextaxis)
        return ret

def removeoutliers(data, nsigma, remove='both', center='mean', niter=Inf, retind=False, verbose=False):
    """Strip outliers from a dataset, iterating until converged.

    :INPUT:
      data -- 1D numpy array.  data from which to remove outliers.

      nsigma -- positive number.  limit defining outliers: number of
                standard deviations from center of data.

    :OPTIONAL INPUTS:               
      remove -- ('min'|'max'|'both') respectively removes outliers
                 below, above, or on both sides of the limits set by
                 nsigma.

      center -- ('mean'|'median'|value) -- set central value, or
                 method to compute it.

      niter -- number of iterations before exit; defaults to Inf,
               which can occasionally result in empty arrays returned

      retind -- (bool) whether to return index of good values as
                second part of a 2-tuple.

    :EXAMPLE: 
       ::

           from numpy import hist, linspace, randn
           from analysis import removeoutliers
           data = randn(1000)
           hbins = linspace(-5,5,50)
           d2 = removeoutliers(data, 1.5, niter=1)
           hist(data, hbins)
           hist(d2, hbins)

       """
    # 2009-09-04 13:24 IJC: Created
    # 2009-09-24 17:34 IJC: Added 'retind' feature.  Tricky, but nice!
    # 2009-10-01 10:40 IJC: Added check for stdev==0
    # 2009-12-08 15:42 IJC: Added check for isfinite

    from numpy import median, ones, isfinite

    def getcen(data, method):
        "Get central value of a 1D array (helper function)"
        if method.__class__==str:
            if method=='median':
                cen = median(data)
            else:
                cen = data.mean()
        else:
            cen = method
        return cen

    def getgoodindex(data, nsigma, center, stdev, remove):
        "Get number of outliers (helper function!)"
        if stdev==0:
            distance = data*0.0
        else:
            distance = (data-center)/stdev
        if remove=='min':
            goodind = distance>-nsigma
        elif remove=='max':
            goodind = distance<nsigma
        else:
            goodind = abs(distance)<=nsigma
        return goodind

    data = data.ravel().copy()

    ndat0 = len(data)
    ndat = len(data)
    iter=0
    goodind = ones(data.shape,bool)
    goodind *= isfinite(data)
    while ((ndat0<>ndat) or (iter==0)) and (iter<niter) and (ndat>0) :
        ndat0 = len(data[goodind])
        cen = getcen(data[goodind], center)
        stdev = data[goodind].std()
        thisgoodind = getgoodindex(data[goodind], nsigma, cen, stdev, remove)
        goodind[find(goodind)] = thisgoodind
        if verbose:
            print "cen>>",cen
            print "std>>",stdev
        ndat = len(data[goodind])
        iter +=1
        if verbose:
            print ndat0, ndat
    if retind:
        ret = data[goodind], goodind
    else:
        ret = data[goodind]
    return ret

def prayerbead(*arg, **kw):
    """Generic function to perform Prayer-Bead (residual permutation) analysis.

    :INPUTS:
       (fitparams, modelfunction, arg1, arg2, ... , data, weights)

      OR:
       
       (allparams, (args1, args2, ..), npars=(npar1, npar2, ...))

       where allparams is an array concatenation of each functions
       input parameters.


    :OPTIONAL INPUTS:
      jointpars -- list of 2-tuples.  
                   For use with multi-function calling (w/npars
                   keyword).  Setting jointpars=[(0,10), (0,20)] will
                   always set params[10]=params[0] and
                   params[20]=params[0].

      parinfo -- None, or list of dicts
                   'parinfo' to pass to the kapteyn.py kpmfit routine.

      gaussprior -- list of 2-tuples, same length as "allparams."
                   The i^th tuple (x_i, s_i) imposes a Gaussian prior
                   on the i^th parameter p_i by adding ((p_i -
                   x_i)/s_i)^2 to the total chi-squared.  

      axis -- int or None
                   If input is 2D, which axis to permute over.
                   (NOT YET IMPLEMENTED!)

      step -- int > 0
                   Stepsize for permutation steps.  1 by default.
                   (NOT YET IMPLEMENTED!)

      verbose -- bool
                   Print various status lines to console.

      maxiter -- int
                   Maximum number of iterations for _each_ fitting step.

      maxfun -- int
                   Maximum number of function evaluations for _each_ fitting step.

    :EXAMPLE: 
      ::

        TBW

    :REQUIREMENTS: 
      :doc:`kapteyn`, :doc:`phasecurves`, :doc:`numpy`
    """
    # 2012-04-30 07:29 IJMC: Created
    # 2012-05-03 16:35 IJMC: Now can impose gaussian priors
    # 2012-09-17 14:08 IJMC: Fixed bug when shifting weights (thanks
    #                        to P. Cubillos)
    
    #from kapteyn import kmpfit
    import phasecurves as pc

    if kw.has_key('axis'):
        axis = kw['axis']
    else:
        axis = None

    if kw.has_key('parinfo'):
        parinfo = kw.pop('parinfo')
    else:
        parinfo = None

    if kw.has_key('verbose'):
        verbose = kw.pop('verbose')
    else:
        verbose = None

    if kw.has_key('step'):
        step = kw.pop('step')
    else:
        step = None

    if kw.has_key('maxiter'):
        maxiter = kw.pop('maxiter')
    else:
        maxiter = 3000

    if kw.has_key('maxfun'):
        maxfun = kw.pop('maxfun')
    else:
        maxfun = 6000

    if kw.has_key('xtol'):
        xtol = kw.pop('xtol')
    else:
        xtol = 1e-12

    if kw.has_key('ftol'):
        ftol = kw.pop('ftol')
    else:
        ftol = 1e-12

    guessparams = arg[0]
    modelfunction = arg[1]
    nparam = len(guessparams)

    if isinstance(arg[-1], dict): 
        # Surreptiously setting keyword arguments:
        kw2 = arg[-1]
        kw.update(kw2)
        arg = arg[0:-1]
    else:
        pass

    narg = len(arg)
    helperargs = arg[2:narg-2]
    data = np.array(arg[-2], copy=False)
    weights = arg[-1]

    if data.ndim > 1:
        print "I haven't implemented 2D multi-dimensional data handling yet!"
    else:
        ndata = data.size

    
    if kw.has_key('npars'):
        print "I haven't yet dealt with this for prayerbead analyses!"
        npars = kw['npars']
        ret = []
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
            ret.append(resfunc(these_params, *arg[1][ii], **lower_kw))

        return ret


    

    fitter_args = (modelfunction,) + helperargs + (data, weights, kw)
    #fitter = kmpfit.Fitter(residuals=pc.devfunc, data=fitter_args)
    #fitter.parinfo = parinfo
    #fitter.fit(params0=guessparams)
    fmin_fit = fmin(pc.errfunc, guessparams, args=fitter_args, full_output=True, disp=False, maxiter=maxiter, maxfun=maxfun)
    bestparams = np.array(fmin_fit[0], copy=True)
    bestmodel = modelfunction(*((guessparams,) + helperargs))
    residuals = data - bestmodel
    allfits = np.zeros((ndata, nparam), dtype=float)
    allfits[0] = bestparams
    if verbose: print "Finished prayer bead step ",
    for ii in range(1, ndata):
        shifteddata = bestmodel + np.concatenate((residuals[ii::], residuals[0:ii]))
        shiftedweights = np.concatenate((weights[ii::], weights[0:ii]))
        shifted_args = (modelfunction,) + helperargs + (shifteddata, shiftedweights, kw)

        fmin_fit = fmin(pc.errfunc, bestparams, args=shifted_args, full_output=True, disp=False, maxiter=maxiter, maxfun=maxfun, xtol=xtol, ftol=ftol)
        theseparams = fmin_fit[0]

        #lsq_fit = optimize.leastsq(pc.devfunc, bestparams, args=shifted_args, full_output=True)
        #theseparams = lsq_fit[0]
        #lsq_fit = lsq_fit[1]
        #bestchisq = pc.errfunc(bestparams, *fitargs)



        #newfitter = kmpfit.Fitter(residuals=pc.devfunc, data=shifted_args)
        #newfitter.parinfo = parinfo
        #newfitter.ftol = 1e-12
        #newfitter.xtol = 1e-12


        #try:
        #    newfitter.fit(params0=bestparams)
        #except:
        #    print "Fitter crashed -- entering debug.  Enter 'q' to quit."
        #theseparams = newfitter.params
        #del newfitter

        #bestmodel = modelfunction(*((guessparams,) + helperargs))
        #residuals = data - bestmodel
        #chisq = pc.errfunc(newfitter.params, *shifted_args)
        allfits[ii] = theseparams
        if verbose: print ("%i of %1." % (ii+1, ndata)),
        #pdb.set_trace()

    return allfits

def meanr(x, nsigma=3, niter=Inf, finite=True, verbose=False,axis=None):
    """Return the mean of an array after removing outliers.
    
    :INPUTS:
      x -- (array) data set to find mean of

    :OPTIONAL INPUT:
      nsigma -- (float) number of standard deviations for clipping
      niter -- number of iterations.
      finite -- if True, remove all non-finite elements (e.g. Inf, NaN)
      axis -- (int) axis along which to compute the mean.

    :EXAMPLE:
      ::

          from numpy import *
          from analysis import meanr
          x = concatenate((randn(200),[1000]))
          print mean(x), meanr(x, nsigma=3)
          x = concatenate((x,[nan,inf]))
          print mean(x), meanr(x, nsigma=3)

    SEE ALSO: :func:`medianr`, :func:`stdr`, :func:`removeoutliers`, 
              :func:`numpy.isfinite`
    """
    # 2009-10-01 10:44 IJC: Created
    # 2010-07-01 13:52 IJC: Now handles higher dimensions.
    from numpy import array, isfinite, zeros, swapaxes

    x = array(x, copy=True)
    xshape = x.shape
    ndim =  len(xshape)
    if ndim==0:
        return x

    if  ndim==1 or axis is None:
        # "1D" array
        x = x.ravel()
        if finite:
            x = x[isfinite(x)]
        x = removeoutliers(x, nsigma, niter=Inf, verbose=verbose)
        return x.mean()
    else:
        newshape = list(xshape)
        oldDimension = newshape.pop(axis) 
        ret = zeros(newshape, float)

        # Prevent us from taking the mean along the axis of primary incidices:
        if axis==0: 
            x=swapaxes(x,0,1)

        if axis>1:
            nextaxis = axis-1
        else:
            nextaxis = 0

        for ii in range(newshape[0]):
            #print 'x.shape>>',x.shape, 'newshape>>',newshape, 'x[ii].shape>>',x[ii].shape, 'ret[ii].shape>>',ret[ii].shape,'ii>>',ii
            ret[ii] = meanr(x[ii], nsigma=nsigma,niter=niter,finite=finite,\
                                 verbose=verbose, axis=nextaxis)
        return ret

def medianr(x, nsigma=3, niter=Inf, finite=True, verbose=False,axis=None):
    """Return the median of an array after removing outliers.
    
    :INPUTS:
      x -- (array) data set to find median of

    :OPTIONAL INPUT:
      nsigma -- (float) number of standard deviations for clipping
      niter -- number of iterations.
      finite -- if True, remove all non-finite elements (e.g. Inf, NaN)
      axis -- (int) axis along which to compute the mean.

    :EXAMPLE:
      ::

          from numpy import *
          from analysis import medianr
          x = concatenate((randn(200),[1000]))
          print median(x), medianr(x, nsigma=3)
          x = concatenate((x,[nan,inf]))
          print median(x), medianr(x, nsigma=3)

    SEE ALSO: :func:`meanr`, :func:`stdr`, :func:`removeoutliers`, 
              :func:`numpy.isfinite`
    """
    # 2009-10-01 10:44 IJC: Created
    #2010-07-01 14:04 IJC: Added support for higher dimensions
    from numpy import array, isfinite, zeros, median, swapaxes

    x = array(x, copy=True)
    xshape = x.shape
    ndim =  len(xshape)
    if ndim==0:
        return x

    if  ndim==1 or axis is None:
        # "1D" array
        x = x.ravel()
        if finite:
            x = x[isfinite(x)]
        x = removeoutliers(x, nsigma, niter=Inf, verbose=verbose)
        return median(x)
    else:
        newshape = list(xshape)
        oldDimension = newshape.pop(axis) 
        ret = zeros(newshape, float)

        # Prevent us from taking the action along the axis of primary incidices:
        if axis==0: 
            x=swapaxes(x,0,1)

        if axis>1:
            nextaxis = axis-1
        else:
            nextaxis = 0

        for ii in range(newshape[0]):
            #print 'x.shape>>',x.shape, 'newshape>>',newshape, 'x[ii].shape>>',x[ii].shape, 'ret[ii].shape>>',ret[ii].shape,'ii>>',ii
            ret[ii] = medianr(x[ii], nsigma=nsigma,niter=niter,finite=finite,\
                                 verbose=verbose, axis=nextaxis)
        return ret

def amedian(a, axis=None):
    """amedian(a, axis=None)

    Median the array over the given axis.  If the axis is None,
    median over all dimensions of the array.  

    Think of this as normal Numpy median, but preserving dimensionality.

    """
#    2008-07-24 14:12 IJC: Copied from
#         http://projects.scipy.org/scipy/numpy/ticket/558#comment:3,
#         with a few additions of my own (i.e., keeping the same
#         dimensionality).
#  2011-04-08 11:49 IJC: Moved to "analysis.py"

    sorted = array(a, subok=True, copy=True)

    if axis==None:
        return median(sorted.ravel())

    ash  = list(sorted.shape)
    ash[axis] = 1

    sorted = np.rollaxis(sorted, axis)
    
    sorted.sort(axis=0)
    
    index = int(sorted.shape[0]/2)
    
    if sorted.shape[0] % 2 == 1:
        returnval = sorted[index]
    else:
        returnval = (sorted[index-1]+sorted[index])/2.0
     
    return returnval.reshape(ash)

def polyfitr(x, y, N, s, fev=100, w=None, diag=False, clip='both', \
                 verbose=False, plotfit=False, plotall=False, eps=1e-13):
    """Matplotlib's polyfit with weights and sigma-clipping rejection.

    :DESCRIPTION:
      Do a best fit polynomial of order N of y to x.  Points whose fit
      residuals exeed s standard deviations are rejected and the fit is
      recalculated.  Return value is a vector of polynomial
      coefficients [pk ... p1 p0].

    :OPTIONS:
        w:   a set of weights for the data; uses CARSMath's weighted polynomial 
             fitting routine instead of numpy's standard polyfit.

        fev:  number of function evaluations to call before stopping

        'diag'nostic flag:  Return the tuple (p, chisq, n_iter)

        clip: 'both' -- remove outliers +/- 's' sigma from fit
              'above' -- remove outliers 's' sigma above fit
              'below' -- remove outliers 's' sigma below fit

    :REQUIREMENTS:
       :doc:`CARSMath`

    :NOTES:
       Iterates so long as n_newrejections>0 AND n_iter<fev. 


     """
    # 2008-10-01 13:01 IJC: Created & completed
    # 2009-10-01 10:23 IJC: 1 year later! Moved "import" statements within func.
    # 2009-10-22 14:01 IJC: Added 'clip' options for continuum fitting
    # 2009-12-08 15:35 IJC: Automatically clip all non-finite points
    # 2010-10-29 09:09 IJC: Moved pylab imports inside this function
    # 2012-08-20 16:47 IJMC: Major change: now only reject one point per iteration!
    # 2012-08-27 10:44 IJMC: Verbose < 0 now resets to 0

    from CARSMath import polyfitw
    from numpy import polyfit, polyval, isfinite, ones
    from pylab import plot, legend, title

    if verbose < 0:
        verbose = 0

    xx = array(x, copy=False)
    yy = array(y, copy=False)
    noweights = (w==None)
    if noweights:
        ww = ones(xx.shape, float)
    else:
        ww = array(w, copy=False)

    ii = 0
    nrej = 1

    if noweights:
        goodind = isfinite(xx)*isfinite(yy)
    else:
        goodind = isfinite(xx)*isfinite(yy)*isfinite(ww)
    
    xx2 = xx[goodind]
    yy2 = yy[goodind]
    ww2 = ww[goodind]

    while (ii<fev and (nrej<>0)):
        if noweights:
            p = polyfit(xx2,yy2,N)
            residual = yy2 - polyval(p,xx2)
            stdResidual = std(residual)
            clipmetric = s * stdResidual
        else:
            p = polyfitw(xx2,yy2, ww2, N)
            p = p[::-1]  # polyfitw uses reverse coefficient ordering
            residual = (yy2 - polyval(p,xx2)) * np.sqrt(ww2)
            clipmetric = s

        if clip=='both':
            worstOffender = abs(residual).max()
            if worstOffender <= clipmetric or worstOffender < eps:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = abs(residual) <= worstOffender
        elif clip=='above':
            worstOffender = residual.max()
            if worstOffender <= clipmetric:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = residual < worstOffender
        elif clip=='below':
            worstOffender = residual.min()
            if worstOffender >= -clipmetric:
                ind = ones(residual.shape, dtype=bool)
            else:
                ind = residual > worstOffender
        else:
            ind = ones(residual.shape, dtype=bool)
    
        xx2 = xx2[ind]
        yy2 = yy2[ind]
        if (not noweights):
            ww2 = ww2[ind]
        ii = ii + 1
        nrej = len(residual) - len(xx2)
        if plotall:
            plot(x,y, '.', xx2,yy2, 'x', x, polyval(p, x), '--')
            legend(['data', 'fit data', 'fit'])
            title('Iter. #' + str(ii) + ' -- Close all windows to continue....')

        if verbose:
            print str(len(x)-len(xx2)) + ' points rejected on iteration #' + str(ii)

    if (plotfit or plotall):
        plot(x,y, '.', xx2,yy2, 'x', x, polyval(p, x), '--')
        legend(['data', 'fit data', 'fit'])
        title('Close window to continue....')

    if diag:
        chisq = ( (residual)**2 / yy2 ).sum()
        p = (p, chisq, ii)

    return p

def spliner(x, y, k=3, sig=5, s=None, fev=100, w=None, clip='both', \
                 verbose=False, plotfit=False, plotall=False, diag=False):
    """Matplotlib's polyfit with sigma-clipping rejection.

    Do a scipy.interpolate.UnivariateSpline of order k of y to x.
     Points whose fit residuals exeed s standard deviations are
     rejected and the fit is recalculated.  Returned is a spline object.

     Iterates so long as n_newrejections>0 AND n_iter<fev. 

     :OPTIONAL INPUTS:
        err:   a set of errors for the data
        fev:  number of function evaluations to call before stopping
        'diag'nostic flag:  Return the tuple (p, chisq, n_iter)
        clip: 'both' -- remove outliers +/- 's' sigma from fit
              'above' -- remove outliers 's' sigma above fit
              'below' -- remove outliers 's' sigma below fit
     """
    # 2010-07-05 13:51 IJC: Adapted from polyfitr
    from numpy import polyfit, polyval, isfinite, ones
    from scipy import interpolate
    from pylab import plot, legend, title

    xx = array(x, copy=True)
    yy = array(y, copy=True)
    noweights = (w==None)
    if noweights:
        ww = ones(xx.shape, float)
    else:
        ww = array(w, copy=True)

    #ww = 1./err**2

    ii = 0
    nrej = 1

    goodind = isfinite(xx)*isfinite(yy)*isfinite(ww)
    
    #xx = xx[goodind]
    #yy = yy[goodind]
    #ww = ww[goodind]

    while (ii<fev and (nrej<>0)):
        spline = interpolate.UnivariateSpline(xx[goodind],yy[goodind],w=ww[goodind],s=s,k=k)
        residual = yy[goodind] - spline(xx[goodind])
        stdResidual = std(residual)
        #if verbose: print stdResidual
        if clip=='both':
            ind =  abs(residual) <= (sig*stdResidual) 
        elif clip=='above':
            ind = residual < sig*stdResidual
        elif clip=='below':
            ind = residual > -sig*stdResidual
        else:
            ind = ones(residual.shape, bool)

        goodind *= ind
        #xx = xx[ind]
        #yy = yy[ind]
        #ww = ww[ind]

        ii += 1
        nrej = len(residual) - len(xx)
        if plotall:
            plot(x,y, '.', xx[goodind],yy[goodind], 'x', x, spline(x), '--')
            legend(['data', 'fit data', 'fit'])
            title('Iter. #' + str(ii) + ' -- Close all windows to continue....')

        if verbose:
            print str(len(x)-len(xx[goodind])) + ' points rejected on iteration #' + str(ii)

    if (plotfit or plotall):
        plot(x,y, '.', xx[goodind],yy[goodind], 'x', x, spline(x), '--')
        legend(['data', 'fit data', 'fit'])
        title('Close window to continue....')

    if diag:
        chisq = ( (residual)**2 / yy )[goodind].sum()
        spline = (spline, chisq, ii, goodind)

    return spline

def neworder(N):
    """Generate a random order the integers [0, N-1] inclusive.

    """
    #2009-03-03 15:15 IJC: Created
    from numpy import random, arange, int
    from pylab import find

    neworder = arange(N)
    random.shuffle(neworder)

    #print N

    return  neworder


def im2(data1, data2, xlab='', ylab='', tit='', bar=False, newfig=True, \
            cl=None, x=[], y=[], fontsize=16):
    """Show two images; title will only be drawn for the top one."""
    from pylab import figure, subplot, colorbar, xlabel, ylabel, title, clim
    from nsdata import imshow

    if newfig:  figure()
    subplot(211)
    imshow(data1,x=x,y=y); 
    if clim<>None:   clim(cl)
    if bar:  colorbar()
    xlabel(xlab, fontsize=fontsize); ylabel(ylab, fontsize=fontsize)
    title(tit)

    subplot(212)
    imshow(data2,x=x,y=y); 
    if clim<>None:   clim(cl)
    if bar:  colorbar()
    xlabel(xlab, fontsize=fontsize); ylabel(ylab, fontsize=fontsize)

    return

def wmean(a, w, axis=None, reterr=False):
    """wmean(a, w, axis=None)

    Perform a weighted mean along the specified axis.

    :INPUTS:
      a : sequence or Numpy array
        data for which weighted mean is computed

      w : sequence or Numpy array
        weights of data -- e.g., 1./sigma^2

      reterr : bool
        If True, return the tuple (mean, err_on_mean), where
        err_on_mean is the unbiased estimator of the sample standard
        deviation.

    :SEE ALSO:  :func:`wstd`
    """
    # 2008-07-30 12:44 IJC: Created this from ...
    # 2012-02-28 20:31 IJMC: Added a bit of documentation
    # 2012-03-07 10:58 IJMC: Added reterr option

    newdata    = array(a, subok=True, copy=True)
    newweights = array(w, subok=True, copy=True)

    if axis==None:
        newdata    = newdata.ravel()
        newweights = newweights.ravel()
        axis = 0

    ash  = list(newdata.shape)
    wsh  = list(newweights.shape)

    nsh = list(ash)
    nsh[axis] = 1

    if ash<>wsh:
        warn('Data and weight must be arrays of same shape.')
        return []
    
    wsum = newweights.sum(axis=axis).reshape(nsh) 
    
    weightedmean = (a * newweights).sum(axis=axis).reshape(nsh) / wsum
    if reterr:
        # Biased estimator:
        #e_weightedmean = sqrt((newweights * (a - weightedmean)**2).sum(axis=axis) / wsum)

        # Unbiased estimator:
        #e_weightedmean = sqrt((wsum / (wsum**2 - (newweights**2).sum(axis=axis))) * (newweights * (a - weightedmean)**2).sum(axis=axis))
        
        # Standard estimator:
        e_weightedmean = np.sqrt(1./newweights.sum(axis=axis))

        ret = weightedmean, e_weightedmean
    else:
        ret = weightedmean

    return ret
    
#def im2(data1, data2, xlab='', ylab='', tit='', bar=False, newfig=True, \ cl=None, x=[], y=[], fontsize=16): """Show two images; title will only be drawn for the top one.""" from pylab import figure, subplot, colorbar, xlabel, ylabel, title, clim from nsdata import imshow if newfig: figure() subplot(211) imshow(data1,x=x,y=y); if clim<>None: clim(cl) if bar: colorbar() xlabel(xlab, fontsize=fontsize); ylabel(ylab, fontsize=fontsize) title(tit) subplot(212) imshow(data2,x=x,y=y); if clim<>None: clim(cl) if bar: colorbar() xlabel(xlab, fontsize=fontsize); ylabel(ylab, fontsize=fontsize) return 
    
def trueanomaly(ecc, eanom=None, manom=None):
    """Calculate (Keplerian, orbital) true anomaly.

    One optional input must be given.

    :INPUT:
       ecc -- scalar.  orbital eccentricity.

    :OPTIONAL_INPUTS:
       eanom -- scalar or Numpy array.  Eccentric anomaly.  See
               :func:`eccentricanomaly`

       manom -- scalar or sequence.  Mean anomaly, equal to 
                2*pi*(t - t0)/period
    """
    # 2011-04-22 14:35 IJC: Created

    if manom is not None:
        eanom = eccentricanomaly(ecc, manom=manom)

    if eanom is not None:
        ret = 2. * np.arctan(  np.sqrt((1+ecc)/(1.-ecc)) * np.tan(eanom/2.)  )
    else:
        ret = None

    return ret
    
def eccentricanomaly(ecc, manom=None, tanom=None, tol=1e-8):
    """Calculate (Keplerian, orbital) eccentric anomaly.

    One optional input must be given.

    :INPUT:
       ecc -- scalar.  orbital eccentricity.

    :OPTIONAL_INPUTS:

       manom -- scalar or sequence.  Mean anomaly, equal to 
                2*pi*(t - t0)/period

       tanom -- scalar or Numpy array.  True anomaly.  See
               :func:`trueanomaly`.
    """
    # 2011-04-22 14:35 IJC: Created

    

    ret = None
    if manom is not None:
        if not hasattr(manom, '__iter__'):
            mwasscalar = True
            manom = [manom]
        else:
            mwasscalar = False
        # Solve Kepler's equation for each element of mean andef maly:
        e = np.zeros(len(manom)) # Initialize eccentric anomaly
        for ii,element in enumerate(manom):
            def kep(e): return element - e + ecc*sin(e)
            e[ii] = optimize.newton(kep, 0., tol=tol)

        if mwasscalar:
            e = e[0]
        ret = e
    
    elif tanom is not None:
        ret = 2. * np.arctan(np.tan(0.5 * tanom) / \
                                 np.sqrt((1. + ecc) / (1. - ecc)))
        
    else:
        ret = None

    return ret