import numpy as np
import glob
import os
from os import path
import pylab as p
import pyfits

'''
Assorted scripts used by Namaste
'''
Namwd = path.dirname(path.realpath(__file__))

def FindTransitManual(kic, lc, cut=10):
    '''
    This uses the KIC/EPIC to
    1) Get matching data
    2) Display in pyplot window
    3) Allow the user to find and type the single transit parameters

    returns
    # lightcurve  (cut to 20Tdur long)
    # 'guess' parameters''
    '''
    if lc==[]:
        lc=getKeplerLC(kic)
    p.ion()
    #PlotLC(lc, 0, 0, 0.1)
    p.figure(1)
    p.plot(lc[:,0],lc[:,1],'.')
    print lc
    print "You have 10 seconds to manouever the window to the transit (before it freezes)."
    print "(Sorry; matplotlib and raw_input dont work too well)"
    p.pause(25)
    Tcen=float(raw_input('Type centre of transit:'))
    Tdur=float(raw_input('Type duration of transit:'))
    depth=float(raw_input('Type depth of transit:'))
    #Tcen,Tdur,depth=2389.7, 0.35, 0.0008
    return np.array((Tcen, Tdur, depth))

def getKeplerLC(kic, getlc=True, dirout=Namwd+'/KeplerFits/'):
    '''
    This module uses the KIC of a planet candidate to download lightcurves to a folder (if it doesnt already exist)
    Default here is 'Namaste/KeplerFits/KIC'
    Input: EPIC or KIC
    Output: Lightcurve (if getlc=true, as deault) or string for file location

    Args:
        kic: EPIC (K2) or KIC (Kepler) id number
        getlc: return directly the lightcurve or simply the file location
        dirout: directory to save new fits files to

    Returns:
        lightcurve or lightcurve file location
    '''
    if str(int(kic)).zfill(9)[0]=='2':
        #K2 File
        fout=K2Lookup(kic)
        if fout=='':
            for camp in range(6):
                # https://archive.stsci.edu/pub/k2/lightcurves/cN/XXXX00000/YY000
                fname='ktwo'+str(int(kic))+'-c'+str(camp)+'_llc.fits'
                #loc=pl.K2Lookup(kic)
                fout=dirout+fname
                if not os.path.exists(fout):
                    print 'Downloading K2 file from MAST...'
                    comm='wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -R \'index*\' -erobots=off https://archive.stsci.edu/pub/k2/lightcurves/c'+str(camp)+'/'+str(np.round(int(kic),-5))+'/'+str(np.round(kic-np.round(int(kic),-5),-3))+'/'+fname
                    os.system(comm)
                    os.system('mv '+fname+' '+fout)
            print 'Downloaded lightcurve to '+fout
        if getlc:
            lc,h=ReadLC(fout)
            return lc,h
        else:
            #returns directory
            return fout
    else:
        #Kepler File
        dout=Namwd+'/KeplerFits/'+str(kic)+'/'
        if not os.path.exists(dout):
            #Downloading Kepler data if not already a folder
            os.system('mkdir '+dout)
            print 'Downloading Kepler files from MAST...'
            comm='wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -R \'index*\' -erobots=off http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(kic)).zfill(9)[0:4]+'/'+str(int(kic)).zfill(9)+'/'
            os.system(comm)
            os.system('mv kplr'+str(int(kic)).zfill(9)+'* '+dout)
            print "Moving files to new directory"
            print 'Done downloading.'
            #Removing short-cadence data:
            for file in glob.glob(dout+'*slc.fits'):
                os.system('rm '+file)
                print 'removing short cadence file '+str(file)
            if getlc:
                lc=np.zeros((3, 3))
                #Stitching lightcurves together
                for file in glob.glob(dout+'*llc.fits'):
                    lc=np.vstack((lc, ReadLC(file)[0]))
                h=ReadLC(file)[1]
                return lc[3:, :], h
            else:
                #Using final file from loop
                return file
        else:
            print "Path exists for "+str(kic)
            if getlc:
                lc=np.zeros((3, 3))
                for file in glob.glob(dout+'*llc.fits'):
                    lc=np.vstack((lc, ReadLC(file)[0]))
                h=ReadLC(file)[1]
                return lc[3:, :], h
            else:
                for file in glob.glob(dout+'*llc.fits'):
                    pass
                return file

def K2Lookup(EPIC, fname=Namwd+'/EPIC_FileLocationTable.txt'):
    '''
    Finds the filename and location for any K2 EPIC (ID)

    Args:
        EPIC: K2 epic number
        fname: location of table of filelocations. HPO-specific currently.

    Returns:
        filename
    '''
    import glob
    if not path.exists('EPIC_FileLocationTable.txt'):
        direct=raw_input('Please enter the directory location of K2 lightcurves:')
        if direct!='':
            f=open(Namwd+'/EPIC_FileLocationTable.txt', 'wa')
            #printing location and EPIC to file in Namaste directory
            for file in glob.glob(direct+'/ktwo?????????-c??_?lc.fits'):
                f.write(file.split('/')[-1][4:13]+','+file+'\n')
        else:
            return ''
    if path.exists(fname):
        EPIC=str(EPIC).split('.')[0]
        tab=np.genfromtxt(fname, dtype=str)
        if EPIC in tab[:, 0]:
            file=tab[np.where(tab[:,0]==EPIC)[0][0]][1]
        else:
            print 'No K2 EPIC in archive.'
            file=''
        return file
    else:
        return ''


def getKeplerLDs(Ts,logFeH=0.0,logg=4.43812,VT=2.0, type='2'):
    '''
    # estimate Quadratic limb darkening parameters u1 and u2 for the Kepler bandpass

    Args:
        Ts: Stellar temp
        logFeH: log(Fe/H)
        logg: log of surface gravity in cgs
        VT:
        type: Type of limb darkening parameter requested.
            1: linear, 2: quadratic, 3: cubic, 4: quintic

    Returns:
        Quadratic limb darkening parameters

    '''
    import scipy.interpolate
    types={'1':[3],'2':[4, 5],'3':[6, 7, 8],'4':[9, 10, 11, 12]}
    #print(label)
    us=[0.0,0.0]
    if types.has_key(type):
        checkint = types[type]
        #print(checkint)
    else:
        print("no key...")
    arr = np.genfromtxt(Namwd+'/KeplerLDLaws.txt',skip_header=2)

    #Rounding Logg and LogFe/H to nearest useable values
    arr=np.hstack((arr[:, 0:3], arr[:, checkint]))
    arr=arr[arr[:, 1]==float(int(logg*2+0.5))/2]
    FeHarr=np.unique(arr[:, 2])
    logFeH=FeHarr[find_nearest(FeHarr,logFeH)]
    arr=arr[arr[:, 2]==logFeH]
    if 3500.<Ts<35000.:
        #Interpolating arrays  for u1 & u2 against T.
        OutArr=np.zeros(len(checkint))
        for i in range(3, len(arr[0, :])):
            us=scipy.interpolate.interp1d(arr[:, 0], arr[:, i])
            OutArr[i-3]=us(Ts)
        return OutArr
    else:
        print 'Temperature outside limits'
        return 0.0, 0.0

def k2f(lc, winsize=7.5, stepsize=0.3, niter=10):
    '''
    #Flattens any K2 lightcurve while maintaining in-transit depth.

    Args:
        lc: 3-column lightcurve
        winsize: size of window over which to polyfit
        stepsize: size of step each iteration
        niter: number of iterations over which anomalous values are removed to improve Chi-Sq

    Returns:

    '''
    import k2flatten as k2f
    return k2f.ReduceNoise(lc,winsize=winsize,stepsize=stepsize,niter=niter)#,ReduceNoise(lc, winsize=winsize, stepsize=stepsize, niter=niter)

def find_nearest(arr,value):
    '''
    #Finds index of nearest matching value in an array

    Args:
        arr: Array to search for value in
        value: Value

    Returns:
        index of the nearest value to that searched

    '''
    arr = np.array(arr)
    #find nearest value in array
    idx=(np.abs(arr-value)).argmin()
    return idx

def ReadLC(fname):
    '''

    # Reads a Kepler fits file to outputt a 3-column lightcurve and header.
    # It recognises the input of K2 PDCSAP, K2 Dave Armstrong's detrending and Kepler lightcurves

    INPUTS:
    # filename

    OUTPUT:
    # Lightcurve in columns of [time, relative flux, relative flux error];
    # Header with fields [KEPLERID,RA_OB,DEC_OBJ,KEPMAG,SAP_FLUX]

    '''
    import sys
    fnametrue=str(np.copy(fname))
    print fnametrue
    fname=fname.split('/')[-1]
    if str(fname).find('kplr')!=-1 or str(fname).find('Kepler')!=-1 or str(fname).find('KOI')!=-1:
        #Looks for kepler, kplr or KOI in name
        if fnametrue.split('/')[-3]=='KeplerFits':
            #If a load of lightcurves have been downloaded to the default folder, this sticks them all together:
            lc=np.zeros((3, 3))
            #Stitching lightcurves together
            for file in glob.glob(str(fnametrue.rsplit('/',1)[0])+'/*llc.fits'):
                newlc,h=OpenKepler(file)
                lc=np.vstack((lc, newlc))
            return lc[3:, :], h
        else:
            #Else we just return the opened filename
            return OpenKepler(fnametrue)
    elif str(fname).find('ktwo')!=-1 or str(fname).find('epic')!=-1:
        #Looks for 'ktwo' or 'epic' in name
        if str(fname).find('lc.fits')!=-1:
            OpenK2_PDC(fnametrue)
        elif str(fname).find('D.fits')!=-1:
            #Assumes file is detrended with Armstrong technique
            return OpenK2_DJA(fnametrue)
        else:
            print 'Not detrended'
            return 0, 0
    else:
        print 'File not found'

def OpenK2_DJA(fnametrue):
    a = pyfits.open(fnametrue)
    Nonans=[np.isnan(a[1].data['TIME'])==0]
    time=(a[1].data['TIME'])[Nonans]
    flux=(a[1].data['DETFLUX'])[Nonans]
    flux_err=(a[1].data['DETFLUX_ERR'])[Nonans]
    #((a[1].data['APTFLUX'][Nonans])/np.median(a[1].data['APTFLUX'][Nonans]))*(a[1].data['APTFLUX_ERR'][Nonans]/a[1].data['APTFLUX'][Nonans])
    if np.isnan(flux_err).sum()==len(flux_err):
        flux_err=np.sqrt(a[1].data['APTFLUX'][Nonans])/a[1].data['APTFLUX'][Nonans]*flux
    Lc=np.column_stack(((a[1].data['TIME']), flux, flux_err/flux))
    Lc=Lc[np.isnan(Lc.sum(axis=1))==0, :]
    #TCEN=14
    #FORMAT: KEPID, RA, DEC, KEPMAG, MEDFLUX
    header = np.array([a[0].header['KEPLERID'],a[0].header['RA_OBJ'],a[0].header['DEC_OBJ'],a[0].header['KEPMAG'],np.median(a[1].data['APTFLUX'][Nonans])])

    #DOING POLY FIT TO BOTH HALFS OF CAMPAIGN 1 DATA
    return Lc,  np.array(header.split(','))

def OpenK2_PDC(fnametrue):
    a = pyfits.open(fnametrue)
    time=(a[1].data['TIME'])
    flux=(a[1].data['PDCSAP_FLUX'])
    flux_err=(a[1].data['PDCSAP_FLUX_ERR'])
    Lc=np.column_stack(((a[1].data['TIME']), flux, flux_err))
    Nonans=[np.isnan(Lc.sum(axis=1))==0]
    Lc=Lc[Nonans]
    med=np.median(Lc[:, 1])
    Lc[:, 1]/=med
    Lc[:, 2]/=med
    header = np.array([a[0].header['KEPLERID'],a[0].header['RA_OBJ'],a[0].header['DEC_OBJ'],a[0].header['KEPMAG'],np.median(a[1].data['PDCSAP_FLUX'][Nonans])])
    return Lc,  header

def OpenKepler(fnametrue):
    a = pyfits.open(fnametrue)
    time=(a[1].data['TIME'])
    flux=(a[1].data['SAP_FLUX'])
    flux_err=a[1].data['SAP_FLUX_ERR']
    Lc=np.column_stack((time, flux, flux_err))
    Nonans=np.logical_not(np.isnan(Lc.sum(axis=1)))
    Lc=Lc[Nonans, :]
    Lc[:, 2]=((Lc[:, 2])/np.median(Lc[:, 1]))
    Lc[:, 1]/=np.median(Lc[:, 1])
    Lc=Lc[np.prod(Lc, axis=1)!=0, :]
    header = np.array([a[0].header['KEPLERID'],a[0].header['RA_OBJ'],a[0].header['DEC_OBJ'],a[0].header['KEPMAG'],np.median(a[1].data['SAP_FLUX'][Nonans])])
    return Lc,  header

def TypefromT(T):
    '''
    # Estimates spectral type from Temperature
    # Types A0 to G6 taken from Boyajian 2012. Other values taken from Unreferenced MS approximations

    INPUTS:
    Temp

    OUTPUTS:
    Spectral Type (string)

    '''

    defs=np.genfromtxt(Namwd+"specTypes.txt", dtype=[('N', 'f8'),('Spec', 'S4'), ('Temp', 'f8'), ('Tmax','f8')])
    if T>35000 or T<3000:
        print "Stellar temperature outside spectral type limits"
        return None
    else:
        Type=defs['Spec'][(find_nearest(defs['Temp'],T))]
        return Type

def RfromT(Teff,Tefferr, lenn=10000, logg=4.43812,FeH=0.0):
    '''
    # Estimates radius from Temperature.
    # From Boyajian 2012, and only valid below G0 stars
    # For hotter stars, Torres (2010) is used

    INPUTS:
    Temp, Temp Error, (Nint, (logg, Fe/H)<-only for Torres calculation)

    OUTPUTS:
    Radius, Radius Error

    '''

    #Scaling Temp error up if extremely low
    Tefferr=150 if Tefferr<150 else Tefferr

    #Boyajian (2012) dwarf star relation (<5800K). Generates 10000 Teffs and puts through equation
    if 3200<=Teff<=5600:
        Teff=np.random.normal(Teff, Tefferr, lenn)
        d, c, b, a=np.random.normal(-8.133, 0.226, lenn), np.random.normal(5.09342, 0.16745, lenn)*10**(-3), 10**(-7)*np.random.normal(-9.86602, 0.40672, lenn), np.random.normal(6.479643, 0.32429, lenn)*10**(-11)
        #d, c, b, a=np.random.normal(-10.8828, 0.1355, lenn), np.random.normal(7.18727, 0.09468, lenn)*10**(-3), 10**(-6)*np.random.normal(-1.50957, 0.02155, lenn), np.random.normal(1.07572, 0.01599, lenn)*10**(-10)
        R = d + c*Teff + b*Teff**2 + a*Teff**3
        Rout,  Routerr=np.median(R), np.average(abs(np.tile(np.median(R),2)-np.percentile(R, [18.39397206,81.60602794])))/np.sqrt(len(R))
        return Rout, Routerr

    #Old method from Torres. Now used for 5800K+
    elif Teff>5600:
        Teff=np.random.normal(Teff, Tefferr, lenn)
        X= np.log10(Teff)-4.1
        b=np.random.normal([2.4427, 0.6679, 0.1771, 0.705, -0.21415, 0.02306, 0.04173], [0.038, 0.016, 0.027, 0.13, 0.0075, 0.0013, 0.0082], (lenn,7))
        logR = b[:, 0] + b[:, 1]*X + b[:, 2]*X**2 + b[:, 3]*X**3 + b[:, 4]*(logg)**2 + b[:, 5]*(logg)**3 + b[:,6]*FeH
        R=10**logR
        Rmed, Rerr=np.median(R), np.average(abs(np.median(R)-np.percentile(R, [18.39397206,81.60602794])))
        return Rmed, Rerr
    else:
        print "Temp too low"

def MfromR(R, Rerr, lenn=10000):
    '''
    # Estimates radius from Mass.
    # From Boyajian 2012, and only technically valid below G0 stars, although most Kepler lcs are dwarfs.

    INPUTS:
    Radius, Radius Error, (Nint)

    OUTPUTS:
    Mass, Mass Error

    '''

    #Converting to solar radii if not alread
    R=R/695500000 if R>1e4 else R
    R=np.random.normal(R, Rerr, lenn)

    #Coefficients from Boyajian
    c, b, a=np.random.normal(0.0906, 0.0027, lenn), np.random.normal(0.6063, 0.0153,lenn), np.random.normal(0.3200, 0.0165, lenn)

    #Calculating
    M= (-b+np.sqrt(b**2-4*a*(c-R)))/(2*a)
    return np.median(M), np.average(abs(np.median(M)-np.percentile(M, [18.39397206,81.60602794])))


def getStarDat(kic,file=''):
    '''
    # Uses photometric information in the Kepler fits file of each lightcurve to estimate Temp, SpecType, Radius & Mass
    #Colours include B-V,V-J,V-H,V-K.
    # V is estimated from g and r mag if available
    # otherwise assumes KepMag=V (which is true for G-type stars, but diverges fot hotter/cooler stars)
    # JHK magnitudes also calculated independant of Vmags (which are often inaccurate).
    # Any disagreement is therefore obvious in Terr

    INPUTS:
    kplr filename

    OUTPUTS:
    StarData Object [T,Terr,R,Rerr,M,Merr]

    '''
    #Looking in datafile for K2 stars:
    if str(int(kic)).zfill(9)[0]=='2':
        Ts = K2Guess(int(kic))
        if Ts==None and file!='':
            #Using the fits file
            Ts=FitsGuess(file)
        if Ts!=None:
            R,Rerr=RfromT(Ts[0],Ts[1])
            M,Merr=MfromR(R,Rerr)
            return [Ts[0],Ts[1],R,Rerr,M,Merr]
    #If a value from the above arrays is 2sigma out in temperature, cut...
    else:
        try:
            import pandas as pd
            #One problem is that this table is not particularly up to date and has huge Radii errors...
            koitable=pd.read_csv(Namwd+'/KeplerKOItable.csv',skiprows=155)
            match=koitable[koitable['kepid']==int(kic)]
            return list(abs(np.array([match['koi_steff'].values[0], match['koi_steff_err1'].values[0],  match['koi_steff_err2'].values[0], match['koi_srad'].values[0], match['koi_srad_err1'].values[0],  match['koi_srad_err2'].values[0], match['koi_smass'].values[0], match['koi_smass_err1'].values[0],  match['koi_smass_err2'].values[0]])))
        except KeyError:
            if file!='':
                #Using the fits file
                Ts=FitsGuess(file)
                if Ts!=None:
                    R,Rerr=RfromT(Ts[0],Ts[1])
                    M,Merr=MfromR(R,Rerr)
                    return [Ts[0],Ts[1],R,Rerr,M,Merr]
    print "Unable to get Star Data from catalogues or colours."
    #Does this kill the script here?
    pass

def FitsGuess(file):
    #Otherwise using fits colours
    import pyfits
    try:
        lcfits=pyfits.open(file)
    except ValueError:
        print "Lightcurve is not a fits file and data not in EPIC/KIC catalogues."
        return None
    Kepmag, gmag, rmag, imag, zmag,  Jmag, Hmag, Kmag= lcfits[0].header['KEPMAG'], \
            lcfits[0].header['GMAG'], lcfits[0].header['RMAG'], lcfits[0].header['IMAG'],\
            lcfits[0].header['ZMAG'], lcfits[0].header['JMAG'], lcfits[0].header['HMAG'],  lcfits[0].header['KMAG']
    if type(gmag)==float and type(rmag)==float:
        Vmag=float(gmag) - 0.58*(float(gmag)-float(rmag)) - 0.01
        #Verr=((0.42*float(catdat[24]))**2+(0.58*float(catdat[26]))**2)**0.5
    else:
        #Assuming kepler mag is the same as V (correct for ~Gtype)
        Vmag=Kepmag
    Temp,Terr= coloursCheck(np.array([0,Vmag, Jmag, Hmag, Kmag]))
    return Temp,Terr

def coloursCheck(mags):
    '''
    #Estimate stellar Temperature from b,v,j,h,k magnitudes

    Args:
        mags: array of b,v,j,h,k magnitudes

    Returns:
        T : Stellar Temperature
        Terr: Stellar Temperature error from the STD of the distribution of each temp estimate \
            (to a minimum of 150K precision)

    '''
    mags=np.where(mags==0,np.nan,mags)
    b,v,j,h,k=mags
    cols=np.array([b-v,v-j,v-h,v-k,j-h,j-k])
    nonan=np.arange(0,len(cols),1)[np.logical_not(np.isnan(cols))]
    MScols=np.genfromtxt(Namwd+"/MainSeqColours.csv", delimiter=',')
    coltab=np.column_stack(((MScols[:, 3]),(MScols[:, 6]),(MScols[:, 7]), \
        (MScols[:, 8]), (MScols[:, 7])-(MScols[:, 6]), (MScols[:, 8])-(MScols[:, 6])))
    T=np.zeros((len(cols)))
    import scipy.interpolate
    for c in nonan:
        func=scipy.interpolate.interp1d(coltab[:,c],MScols[:, 14])
        T[c]=func(cols[c])
    T=T[T!=0]
    #Returning median of temp and std (if std>150)
    return np.median(T), np.std(T) if np.std(T)>150.0 else 150.0

def PlanetRtoM(Rad):
    '''
    # Estimate Planetary Mass from Radius.
    # Adapted from Weiss & Marcy and extrapolated below/above distribution
    # Could use the Kipping paper

    Args:
        Rad: Planet radius (in Rearths)

    Returns:
        Planet Mass in MEarth

    '''
    Rearth=6.137e6
    Mearth=5.96e24

    if Rad<=1.5:
        M=((2430+3390*Rad)*(1.3333*np.pi*(Rad*Rearth)**3))/Mearth
    elif Rad>1.5 and Rad<=4.0:
        M= 2.69*(Rad**0.93)
    elif Rad>4. and Rad<=16.0:
        import scipy.interpolate
        o=np.array([[3.99, 9.0,10.5,12.0,13.0,14.0,14.5, 15.5, 16.0],[10.0, 100.0,300.0,1000.0,1100.0,1200.0, 1350.0, 5000.0,  10000.0]])
        f= scipy.interpolate.interp1d(o[0, :],o[1, :])
        M= f(Rad)
    elif Rad>16.0:
        #Above ~1.5Rjup, estimated to be M~30Mjup
        M=10000.0
    return M

def PlotLC(lc, offset=0,  col=0,  fig=1, titl='Lightcurve', hoff=0, epic=0):
    '''
    This function plots a lightcurve (lc) in format: times, flux, flux_errs.
    Offset allows multiple lightcurves to be displayed above each other. Col allows colour to be vaired (otherwise, grey)

    Args:
        lc: 3-column lightcurve (t,y,yerr)
        offset: y-axis offset
        col: color (matplotlib single-digit code or hexcolor)
        fig: Figure number
        titl: Title to plot
        hoff: horizontal offset
        epic: Lightcurve ID

    Returns:
        You return nothing, Jon Snow.
    '''
    import pylab as p
    if epic==1:
        #Doing lookup and get lc intrisincally
        lc=ReadLC(K2Lookup(lc))[0]
    print "Plotting Lightcurve"
    if col==0:
        col='#DDDDDD'#'#28596C'
        #col= '#'+"".join([random.choice('0123456789abcdef') for n in xrange(6)])
    p.ion(); p.figure(fig); p.title(titl);p.xlabel('time');p.ylabel('Relative flux');
    p.errorbar(lc[:, 0]+hoff, lc[:, 1]+offset, yerr=lc[:, 2], fmt='.',color=col)

def K2Guess(EPIC, colours=0, pm=0, errors=0):
    '''#This module takes the K2 EPIC and finds the magnitudes from the EPIC catalogue. It then uses these to make a best-fit guess for spectral type, mass and radius
    #It can also take just the B-V, V-J, V-H and V-K colours.
    #Fitzgerald (1970 - A&A 4, 234) http://www.stsci.edu/~inr/intrins.html
    #[N Sp. Type 	U-B 	(B-V) 	(V-R)C 	(V-I)C 	(V-J) 	(V-H) 	(V-K) 	(V-L) 	(V-M) 	(V-N) 	(V-R)J 	(V-I)J  Temp    Max Temp]
    Returns:
    #   Spec Types, Temps, Radii, Masses.
    '''
    import csv
    #ROWS: EPIC,RA,Dec,Data Availability,KepMag,HIP,TYC,UCAC,2MASS,SDSS,Object type,Kepflag,pmra,e_pmra,pmdec,e_pmdec,plx,e_plx,Bmag,e_Bmag,Vmag,e_Vmag,umag,e_umag,gmag,e_gmag,rmag,e_rmag,imag,e_imag,zmag,e_zmag,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag
    reader=csv.reader(open(Namwd+'/EPICcatalogue.csv'))
    a=0
    result={}
    for row in reader:
        if a>1:
            key=row[0]
            if key in result:
                pass
            result[key] = row[1:]
        a+=1
    try:
        catdat=np.array(result[str(EPIC)])
    except KeyError:
        return None

    catdat[np.where(catdat=='')]='0'
    V=float(catdat[19])
    Verr=float(catdat[20])
    if V=='0' and 0 not in catdat[23:25]:
        V=float(catdat[23]) - 0.58*(float(catdat[23])-float(catdat[25])) - 0.01
        Verr=((0.42*float(catdat[24]))**2+(0.58*float(catdat[26]))**2)**0.5
    #b,v,j,h,k COLOURS
    mags=np.array((float(catdat[17]),V,float(catdat[31]),float(catdat[33]),float(catdat[35])))
    return coloursCheck(mags)