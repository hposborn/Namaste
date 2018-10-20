import numpy as np
import glob
import os
from os import path
import matplotlib
#matplotlib.use('Agg')
import pylab as plt
import logging
import astropy.io.fits as fits
import astropy.units as u
import astropy.coordinates as co
import pandas as pd

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
    plt.ion()
    #PlotLC(lc, 0, 0, 0.1)
    plt.figure(1)
    plt.plot(lc[:,0],lc[:,1],'.')
    print(lc)
    print("You have 10 seconds to manouever the window to the transit (before it freezes).")
    print("Sorry; matplotlib and raw_input dont work too well")
    plt.pause(25)
    Tcen=float(raw_input('Type centre of transit:'))
    Tdur=float(raw_input('Type duration of transit:'))
    depth=float(raw_input('Type depth of transit:'))
    #Tcen,Tdur,depth=2389.7, 0.35, 0.0008
    plt.clf()
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
                    print('Downloading K2 file from MAST...')
                    comm='wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -R \'index*\' -erobots=off https://archive.stsci.edu/pub/k2/lightcurves/c'+str(camp)+'/'+str(np.round(int(kic),-5))+'/'+str(np.round(kic-np.round(int(kic),-5),-3))+'/'+fout
                    os.system(comm)
                    #os.system('mv '+fname+' '+fout)
            print('Downloaded lightcurve to '+fout)
        if getlc:
            lc,mag=ReadLC(fout)
            return lc,mag
        else:
            #returns directory
            return fout
    else:
        #Kepler File
        dout=dirout+str(kic)+'/'
        if not os.path.exists(dout):
            #Downloading Kepler data if not already a folder
            os.system('mkdir '+dout)
            print('Downloading Kepler files from MAST...')
            comm='wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -R \'index*\' -erobots=off http://archive.stsci.edu/pub/kepler/lightcurves/'+str(int(kic)).zfill(9)[0:4]+'/'+str(int(kic)).zfill(9)+'/'
            os.system(comm)
            os.system('mv kplr'+str(int(kic)).zfill(9)+'* '+dout)
            #print "Moving files to new directory"
            print('Done downloading.')
            #Removing short-cadence data:
            for file in glob.glob(dout+'*slc.fits'):
                os.system('rm '+file)
                print('removing short cadence file '+str(file))
        if getlc:
            print(dout)
            #Stitching lightcurves together
            return ReadLC(dout)
        else:
            #Using final file from loop
            return file

def VandDL(camp,epic,fitsfolder,v=1,returnradecmag=False):
    from urllib.request import urlopen
    urlfitsname='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(camp)+'/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(camp)+'_kepler_v'+str(int(v))+'_llc.fits'.replace(' ','')
    #http://archive.stsci.edu/missions/hlsp/k2sff/c03/205800000/91876/hlsp_k2sff_k2_lightcurve_205891876-c03_kepler_v1_llc.fits
    #http://archive.stsci.edu/missions/hlsp/k2sff/c00/202000000/89984/hlsp_k2sff_k2_lightcurve_202089984-c00_kepler_v1_llc-default-aper.txt
    print(urlfitsname)
    if not path.exists(fitsfolder+urlfitsname.split('/')[-1]):
        ret = urlopen(urlfitsname)
        print(ret.code)
        if ret.code == 200:
            f=fits.open(urlfitsname)
            print("writing to "+fitsfolder+urlfitsname.split('/')[-1])
            f.writeto(fitsfolder+urlfitsname.split('/')[-1])
    else:
        f=fits.open(fitsfolder+urlfitsname.split('/')[-1])
    if returnradecmag:
        return [f[0].header['RA_OBJ'],f[0].header['DEC_OBJ'],f[0].header['KEPMAG']]
    return OpenFits(f,fitsfolder+urlfitsname.split('/')[-1])

def noneg_GetAssymDist(inval,uperr,derr,nd=5000,Nnegthresh=0.001):
    neg=np.tile(True,nd)
    retdist=np.zeros(nd)
    Nnegs=nd
    while Nnegs>Nnegthresh*nd:
        retdist[neg]=GetAssymDist(inval,uperr,derr,Nnegs,returndist=True)
        neg=retdist<0
        Nnegs=np.sum(neg)
    return abs(retdist)

def GetAssymDist(inval,uperr,derr,nd=5000,returndist=False):
    derr=abs(derr)

    #Up and Down errors indistinguishable, returning numpy array:
    if abs(np.log10(uperr/derr)-1.0)<0.05:
        if returndist:
            return np.random.normal(inval,0.5*(uperr+derr))
        else:
            return inval, uperr,derr
    else:
        #Fitting some skew to the distribution to match up/down errors.
        if derr>uperr:
            minusitall=True
            derr,uperr=uperr,derr
        else:
            minusitall=False
        mult= -1.0 if inval<0.0 else 1.0
        inval=mult*inval
        val = derr*3 if inval>derr*3 else inval
        plushift= 2.5*derr if val<1.2*derr else 0
        val+=plushift
        #finding n exponent for which power to this produces equidistant errors
        findn = lambda n : abs(1.0-(((val+uperr)**(n)-(val)**(n))/((val)**(n)-(val-derr)**(n))))
        import scipy.optimize as op
        res=op.minimize(findn,0.5)
        try:
            dist=np.random.normal(val**(res.x),abs((val**(res.x))-(val-derr)**(res.x)),nd)**(1.0/res.x)
            n=0
            if np.sum(np.isnan(dist))!=nd:
                while np.sum(np.isnan(dist))>0 or n>10:
                    dist[np.isnan(dist)]=np.random.normal(val**(res.x),abs((val**(res.x))-(val-derr)**(res.x)),np.sum(np.isnan(dist)))**(1.0/res.x)
                    n+=1
                #print np.sum(np.isnan(dist))
                #print str(derr)+' vs '+str(np.median(dist)-np.percentile(dist,16))+"  |  "+str(uperr)+' vs '+str(np.percentile(dist,84)-np.median(dist))
                if plushift!=0:
                    dist-=plushift
                if minusitall:
                    dist=-1*(dist-val)+val
                dist=mult*(dist+inval-derr*3) if inval>derr*3 else mult*(dist)
                if returndist:
                    return dist
                else:
                    if mult==-1:
                        return [np.median(dist),np.percentile(dist,84)-np.median(dist),np.median(dist)-np.percentile(dist,16)]
                    else:
                        return [np.median(dist),np.median(dist)-np.percentile(dist,16),np.percentile(dist,84)-np.median(dist)]
            else:
                raise ValueError()
        except:
            return np.zeros(3)

def CalcDens(Star,nd=6000,returnpost=False,overwrite=True):
    #Returns density from StarDat (in form T,Te1,Te2,R,Re1,Re2,M,Me1,Me2,opt:[logg,logge1,logge2,dens,dense1,dense2])
    dens1 = lambda logg,rad: np.power(10,logg-2.0)*(4/3.0*np.pi*6.67e-11*(rad*695500000))**-1.0
    dens2 = lambda mass,rad: (1.96e30*mass)/(4/3.0*np.pi*(695500000*rad)**3)

    if not hasattr(Star,'srads'):
        srads= GetAssymDist(Star.srad,Star.sraduerr,Star.sradderr,nd=nd,returndist=True)
    else:
        srads=Star.srads
    print(srads)
    srads[srads<0]=abs(np.random.normal(Star.srad,0.25*(Star.sraduerr+Star.sradderr),np.sum(srads<0)))

    if hasattr(Star,'sdens') and not overwrite:
        if returnpost and not hasattr(Star,'sdenss'):
            densdist=GetAssymDist(Star.dens,Star.sdensuerr,Star.sdensderr,nd=nd,returndist=True)
            return densdist
        elif returnpost and hasattr(Star,'sdenss'):
            return getattr(Star,'sdenss')
        else:
            return Star.sdens,Star.sdensuerr,Star.sdensderr
    elif hasattr(Star,'slogg')==12:
        #Returns a density as the weighted average of that from logg and mass
        if not hasattr(Star,'sloggs'):
            sloggs= GetAssymDist(Star.slogg,Star.slogguerr,Star.sloggderr,nd=nd,returndist=True)
        else:
            sloggs=Star.sloggs
        d1=np.array([dens1(sloggs[l],srads[l]) for l in range(nd)])
        stdd1=np.std(d1)
    elif hasattr(Star,'smass'):
        stdd1=np.inf
        if not hasattr(Star,'sloggs'):
            smasss= GetAssymDist(Star.smass,Star.smassuerr,Star.smassderr,nd=nd,returndist=True)
        else:
            smasss=Star.smasss
        smasss[smasss<0]=abs(np.random.normal(Star.smass,0.25*(Star.smassuerr+Star.smassderr),np.sum(smasss<0)))
        d2=np.array([dens2(smasss[m],srads[m]) for m in range(nd)])
        #Using distribution with smallest std...
    post=d1 if stdd1<np.std(d2) else d2
    post/=1409 #Making solar density
    if returnpost:
        return post
    else:
        dens=np.percentile(post,[16,50,84])
        return np.array([dens[1],np.diff(dens)[0],np.diff(dens)[1]])

def dens(StarDat,nd=6000,returnpost=False):
    _,_,_,rad,raderr1,raderr2,mass,masserr1,masserr2,logg,loggerr1,loggerr2=StarDat
    #Returns a density as the weighted average of that from logg and mass
    dens1 = lambda logg,rad: np.power(10,logg-2.0)*(4/3.0*np.pi*6.67e-11*(rad*695500000))**-1.0
    dens2 = lambda mass,rad: (1.96e30*mass)/(4/3.0*np.pi*(695500000*rad)**3)

    d1_up=dens1(np.random.normal(logg,loggerr2,nd),np.random.normal(rad,raderr1,nd))
    d1_dn=dens1(np.random.normal(logg,loggerr1,nd),np.random.normal(rad,raderr2,nd))
    d2_up=dens2(np.random.normal(mass,masserr2,nd),np.random.normal(rad,raderr1,nd))
    d2_dn=dens2(np.random.normal(mass,masserr1,nd),np.random.normal(rad,raderr2,nd))
    #Combining up/down dists alone for up.down uncertainties. COmbining all dists for median.
    #Gives almost identical values as a weighted average of resulting medians/errors.
    print("logg/rad: "+str(np.median(np.vstack((d1_dn,d1_up))))+", mass/rad:"+str(np.median(np.vstack((d2_dn,d2_up))))+"")
    if returnpost:
        #Returning combined posterier...
        #This has the problem of down errors being present in upwards errors (which may be less...)
        post=np.vstack((d2_dn,d1_dn,d2_up,d1_up))
        return post
    else:
        dens=[np.percentile(np.vstack((d2_dn,d1_dn)),16),np.median(np.vstack((d2_dn,d2_up,d1_up,d1_dn))),np.percentile(np.vstack((d2_up,d1_up)),84)]
        return np.array([dens[1],np.diff(dens)[0],np.diff(dens)[1]])

def dens2(logg,loggerr1,loggerr2,rad,raderr1,raderr2,mass,masserr1,masserr2,nd=6000,returnpost=False):
    #Returns a density as the weighted average of that from logg and mass
    dens1 = lambda logg,rad: (np.power(10,logg-2)/(1.33333*np.pi*6.67e-11*rad*695500000))/1410.0
    dens2 = lambda mass,rad: ((mass*1.96e30)/(4/3.0*np.pi*(rad*695500000)**3))/1410.0

    loggs= np.random.normal(logg,0.5*(loggerr1+loggerr2),nd)
    rads= np.random.normal(rad,0.5*(raderr1+raderr2),nd)
    rads[rads<0]=abs(np.random.normal(rad,0.25*(raderr1+raderr2),np.sum(rads<0)))
    masses= np.random.normal(mass,0.5*(masserr1+masserr2),nd)
    masses[masses<0]=abs(np.random.normal(mass,0.25*(masserr1+masserr2),np.sum(masses<0)))
    d1=np.array([dens1(loggs[l],rads[l]) for l in range(nd)])
    d2=np.array([dens2(masses[m],rads[m]) for m in range(nd)])
    #Combining up/down dists alone for up.down uncertainties. COmbining all dists for median.
    #Gives almost identical values as a weighted average of resulting medians/errors.
    print("logg/rad: "+str(np.median(d1))+"+/-"+str(np.std(d1))+", mass/rad:"+str(np.median(d2))+"+/-"+str(np.std(d2)))
    post=d1 if np.std(d1)<np.std(d2) else d2
    if returnpost:
        #Returning combined posterier...
        return post
    else:
        dens=np.percentile(post,[16,50,84])
        return np.array([dens[1],np.diff(dens)[0],np.diff(dens)[1]])

def SpecTypes(teff,logg,specarr):
    #Takes teff and logg to give TRMl estimates...
    spec = specarr.iloc[np.nanargmin(abs(np.log10(float(teff))-specarr.logT_V.values))].name
    print(spec)
    lums=['ZAMS','V','IV','III','II','Ib','Iab','Ia']
    #print specarr.loc[spec,['logg_ZAMS','logg_V','logg_IV','logg_III','logg_II','logg_Ib','logg_Iab','logg_Ia']].values
    #print abs(float(logg)-specarr.loc[spec,['logg_ZAMS','logg_V','logg_IV','logg_III','logg_II','logg_Ib','logg_Iab','logg_Ia']].values)
    #print np.nanargmin(abs(float(logg)-specarr.loc[spec,['logg_ZAMS','logg_V','logg_IV','logg_III','logg_II','logg_Ib','logg_Iab','logg_Ia']].values))
    if logg=='':
        lum='V'
    else:
        lum = lums[np.nanargmin(abs(float(logg)-specarr.loc[spec,['logg_ZAMS','logg_V','logg_IV','logg_III','logg_II','logg_Ib','logg_Iab','logg_Ia']].values))]
    TRMl= m(spec,lum)
    TRMl2=[]
    for i in TRMl:
        TRMl2+=[[i[1],np.diff(i)[0],np.diff(i)[1]]]
    return TRMl2

def getK2LCs(epic,camp,fitsfolder=Namwd):
    if fitsfolder[-1]!='/':fitsfolder=fitsfolder+'/'
    from urllib.request import urlopen
    #VANDERBURG LC:
    camp=str(camp)
    try:
        lcvand=[]
        if len(camp)>3: camp=camp.split(',')[0]
        if camp=='10':
            camp='102'
        elif camp=='et' or camp=='E':
            camp='e'
            #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
        else:
            camp=str(camp.zfill(2))
        if camp=='09' or camp=='11':
            lc=np.zeros((3,3))
            #C91: https://archive.stsci.edu/missions/hlsp/k2sff/c91/226200000/35777/hlsp_k2sff_k2_lightcurve_226235777-c91_kepler_v1_llc.fits
            url1='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'1/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'1_kepler_v1_llc.fits'
            if not path.exists(fitsfolder+url1.split('/')[-1]):
                f1=fits.open(url1)
                f1.writeto(fitsfolder+url1.split('/')[-1])
                print("C91 downloaded and saved...")
            else:
                f1=fits.open(fitsfolder+url1.split('/')[-1])
            url2='http://archive.stsci.edu/missions/hlsp/k2sff/c'+str(int(camp))+'2/'+str(epic)[:4]+'00000/'+str(epic)[4:]+'/hlsp_k2sff_k2_lightcurve_'+str(epic)+'-c'+str(int(camp))+'2_kepler_v1_llc.fits'
            if not path.exists(fitsfolder+url2.split('/')[-1]):
                f2=fits.open(url2)
                f2.writeto(fitsfolder+url2.split('/')[-1])
            else:
                print('opening from '+fitsfolder+url2.split('/')[-1])
                f2=fits.open(fitsfolder+url2.split('/')[-1])
            lcvand=OpenFits(f1,fitsfolder+url1.split('/')[-1])[0]
            lcvand=np.vstack((lcvand,OpenFits(f2,fitsfolder+url1.split('/')[-1])[0]))
        elif camp=='e':
            print("Engineering data")
            #https://www.cfa.harvard.edu/~avanderb/k2/ep60023342alldiagnostics.csv
            url='https://www.cfa.harvard.edu/~avanderb/k2/ep'+str(epic)+'alldiagnostics.csv'
            df=pd.read_csv(url,index_col=False)
            lcvand=np.column_stack((df['BJD - 2454833'].values,df[' Corrected Flux'].values,np.tile(np.median(abs(np.diff(df[' Corrected Flux'].values))),df.shape[0])))
        else:
            print("Getting Vanderburg lc")
            lcvand=VandDL(camp,epic,fitsfolder)[0]
    except:
        logging.debug("Cannot get Vanderburg LC")
    lcpdc=[]
    logging.debug("getting pdc")
    try:
        logging.debug(str(epic)+"  +  "+str(camp))
        urlfilename1=''

        if camp == 10:
        #https://archive.stsci.edu/missions/k2/lightcurves/c1/201500000/69000/ktwo201569901-c01_llc.fits
            urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c102/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c102_llc.fits'
        else:
            urlfilename1='https://archive.stsci.edu/missions/k2/lightcurves/c'+str(int(camp))+'/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c'+str(camp).zfill(2)+'_llc.fits'#elif camp in ['e','et','0','1','2','3','9','00','01','02','09']:
        #    #NO LCs - TPFs only
        #    if camp in ['00','01','02']: camp=camp[1]
        #    if camp=='9' or camp=='09': camp = '92'
        #    urlfilename1='https://archive.stsci.edu/missions/k2/target_pixel_files/c'+str(camp)+'/'+str(epic)[:4]+'00000/'+str(epic)[4:6]+'000/ktwo'+str(epic)+'-c'+str(camp)+'_lpd-targ.fits.gz'
        #TPFs: https://archive.stsci.edu/missions/k2/target_pixel_files/c0/202000000/72000/ktwo202072917-c00_lpd-targ.fits.gz
        #ret = urllib2.urlopen(urlfilename1.replace(' ',''))
        if not path.exists(fitsfolder+urlfilename1.split('/')[-1]):
            f=fits.open(urlfilename1.replace(' ',''))
            print('writing to '+fitsfolder+urlfilename1.split('/')[-1])
            f.writeto(fitsfolder+urlfilename1.split('/')[-1])
        else:
            print('opening from '+fitsfolder+urlfilename1.split('/')[-1])
            f=fits.open(fitsfolder+urlfilename1.split('/')[-1])
        if urlfilename1[-9:]=='_llc.fits':
            print('opening PDC file...')
            lcpdc=OpenFits(f,fitsfolder+urlfilename1.split('/')[-1])[0]
    except:
        logging.debug("Cannot get PDC LC")
    lcev=[]
    try:
        import everest
        #logging.debug("getting lc from everest for "+str(epic))
        st=everest.Everest(int(epic))
        lcev=np.column_stack((st.time,st.flux,st.fraw_err))
        #logging.debug(str(len(lcev))+"-long lightcurve from everest")
        lcev=lcev[nonan(lcev)]
        lcev=lcev[CutAnomDiff(lcev[:,1])]
        lcev[:,1:]/=np.median(lcev[:,1])
    except:
        logging.debug("Cannot get Everest LC")
    return lcvand,lcpdc,lcev


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
            print('No K2 EPIC in archive.')
            file=''
        return file
    else:
        return ''

def getLDs(Ts,feh=0.0,logg=4.43812,VT=2.0,mission='Kepler',type='Quadratic'):
    '''
    T_effs from 2500 to 40000
    types= linear, Quadratic, 3par, 4par
    '''
    import pandas as pd
    import scipy.interpolate
    mission='Kepler';logg=4.0
    if mission=='Kepler' or mission=='kepler' or mission=='K2' or mission=='k2':
        lds=pd.read_fwf('http://www.astro.ex.ac.uk/people/sing/LDfiles/LDCs.Kepler.Table2.txt',header=6,index_col=False,widths=[8,8,8,9,9,9,9,9,9,9,9,9,9])
        lds=lds.iloc[4:]
    elif mission=='CoRoT' or mission=='Corot' or mission=='corot':
        lds=pd.read_fwf('http://www.astro.ex.ac.uk/people/sing/LDfiles/LDCs.CoRot.Table1.txt',header=7,index_col=False,widths=[8,8,8,9,9,9,9,9,9,9,9,9,9])
        lds=lds.iloc[4:]
    else:
        logging.error('Mission must be Kepler or Corot')
    #Finding nearest logg and FeH...
    loggs=pd.unique(lds['Log g'])[~pd.isnull(pd.unique(lds['Log g']))].astype(float)
    fehs=pd.unique(lds['[M/H]'])[~pd.isnull(pd.unique(lds['[M/H]']))].astype(float)
    logg=loggs[np.argmin(abs(logg-loggs))]
    feh=fehs[np.argmin(abs(feh-fehs))]
    lds=lds[(lds['Log g'].astype(float)==logg)*(lds['[M/H]'].astype(float)==feh)]
    typecols=[['linear','Quadratic','3par','4par'],[['linear'],['Quadr','atic'],['3 para','meter non','-linar'],['4','parameter','non-line','ar']]]
    if type not in typecols[0]:
        logging.error('LD type not recognised')
    else:
        n=np.where(np.array(typecols[0])==type)[0]
        cols=typecols[1][n]
        ldouts=[]
        for i in cols:
            us=scipy.interpolate.interp1d(lds['T_eff'].astype(float),lds[i].astype(float))
            ldouts+=[us(float(Ts))]
        return ldouts

def LinearLDs(teff,teff_err1=200,teff_err2=200,logg=4.43,logg_err1=0.4,logg_err2=0.4,feh=0.0,mission='Kepler'):
    if mission=='Kepler' or mission=='kepler' or mission=='K2' or mission=='k2':
        lds=pd.read_fwf('http://www.astro.ex.ac.uk/people/sing/LDfiles/LDCs.Kepler.Table2.txt',header=6,index_col=False,widths=[8,8,8,9,9,9,9,9,9,9,9,9,9])
        lds=lds.iloc[4:-1]
    elif mission=='CoRoT' or mission=='Corot' or mission=='corot':
        lds=pd.read_fwf('http://www.astro.ex.ac.uk/people/sing/LDfiles/LDCs.CoRot.Table1.txt',header=7,index_col=False,widths=[8,8,8,9,9,9,9,9,9,9,9,9,9])
        lds=lds.iloc[4:-1]
    if np.isnan(feh):
        feh=0.0
    from scipy.interpolate import CloughTocher2DInterpolator as ct2d
    print(feh)
    fehs=pd.unique(lds['[M/H]'])[~pd.isnull(pd.unique(lds['[M/H]']))].astype(float)
    feh=fehs[np.nanargmin(abs(feh-fehs))]
    lds=lds[pd.to_numeric(lds['[M/H]'])==feh]
    xys=[(lds[['T_eff']].values.astype(float)[i][0],lds[['Log g']].values.astype(float)[i][0]) for i in range(lds.shape[0])]
    u1int=ct2d(xys,lds[['Quadr']].values.astype(float))
    u2int=ct2d(xys,lds[['atic']].values.astype(float))
    print("Teff = "+str(teff)+"  | logg = "+str(logg))
    print(u1int(teff,logg))
    print(u2int(teff,logg))

    nd=6000
    teffs=[np.random.normal(teff,teff_err1,nd),np.random.normal(teff,teff_err2,nd)]
    loggs=[np.random.normal(logg,logg_err1,nd),np.random.normal(logg,logg_err2,nd)]
    distsld1=[]
    distsld2=[]
    for t in teffs:
        for l in loggs:
            distsld1+=[u1int(t[i],l[i]) for i in range(nd)]
            distsld2+=[u2int(t[i],l[i]) for i in range(nd)]
    distsld1=np.array(distsld1)
    outlds1=np.percentile(distsld1[~np.isnan(distsld1)],[16,50,84])
    distsld2=np.array(distsld2)
    outlds2=np.percentile(distsld2[~np.isnan(distsld2)],[16,50,84])
    return np.array([outlds1[1],np.diff(outlds1)[0],np.diff(outlds1)[1],outlds2[1],np.diff(outlds2)[0],np.diff(outlds2)[1]])


def getLDWITHERRs(Ts,feh=0.0,logg=4.43812,VT=2.0,mission='Kepler',type='Quadratic'):
    '''
    T_effs from 2500 to 40000
    types= linear, Quadratic, 3par, 4par
    '''
    import pandas as pd
    import scipy.interpolate
    mission='Kepler';logg=4.0
    if mission=='Kepler' or mission=='kepler' or mission=='K2' or mission=='k2':
        lds=pd.read_fwf('http://www.astro.ex.ac.uk/people/sing/LDfiles/LDCs.Kepler.Table2.txt',header=6,index_col=False,widths=[8,8,8,9,9,9,9,9,9,9,9,9,9])
        lds=lds.iloc[4:]
    elif mission=='CoRoT' or mission=='Corot' or mission=='corot':
        lds=pd.read_fwf('http://www.astro.ex.ac.uk/people/sing/LDfiles/LDCs.CoRot.Table1.txt',header=7,index_col=False,widths=[8,8,8,9,9,9,9,9,9,9,9,9,9])
        lds=lds.iloc[4:]
    else:
        logging.error('Mission must be Kepler or Corot')
    #Finding nearest logg and FeH...
    loggs=pd.unique(lds['Log g'])[~pd.isnull(pd.unique(lds['Log g']))].astype(float)
    fehs=pd.unique(lds['[M/H]'])[~pd.isnull(pd.unique(lds['[M/H]']))].astype(float)
    logg=loggs[np.argmin(abs(logg-loggs))]
    feh=fehs[np.argmin(abs(feh-fehs))]
    lds=lds[(lds['Log g'].astype(float)==logg)*(lds['[M/H]'].astype(float)==feh)]
    typecols=[['linear','Quadratic','3par','4par'],[['linear'],['Quadr','atic'],['3 para','meter non','-linar'],['4','parameter','non-line','ar']]]
    if type not in typecols[0]:
        logging.error('LD type not recognised')
    else:
        n=np.where(np.array(typecols[0])==type)[0]
        cols=typecols[1][n]
        ldouts=[]
        for i in cols:
            us=scipy.interpolate.interp1d(lds['T_eff'].astype(float),lds[i].astype(float))
            ldouts+=[us(float(Ts))]
        return ldouts


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
    arr = np.genfromtxt(Namwd+'/Tables/KeplerLDLaws.txt',skip_header=2)

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
        print('Temperature outside limits')
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
    # It recognises the input of K2 PDCSAP, K2 Dave Armstrong's detrending, everest lightcurve, and Kepler lightcurves

    INPUTS:
    # filename

    OUTPUT:
    # Lightcurve in columns of [time, relative flux, relative flux error];
    # Header with fields [KEPLERID,RA_OB,DEC_OBJ,KEPMAG,SAP_FLUX]

    '''
    import sys
    logging.debug('fname = '+str(fname))
    fnametrue=str(np.copy(fname))
    logging.debug('fnametrue = '+str(fnametrue))
    fname=fname.split('/')[-1]

    logging.debug('fnametrue[-3] = '+str(fnametrue.split('/')[-3]))
    #logging.debug('fname fing kepler = '+str(fname.find('kplr')))
    try:
        if fnametrue[-1]=='/':
            logging.debug('Kepler file in KeplerFits folder')
            #Looks for kepler, kplr or KOI in name
            #If a load of lightcurves have been downloaded to the default folder, this sticks them all together:
            lc=np.zeros((3, 3))
            #Stitching lightcurves together
            for file in glob.glob(fnametrue+'*llc.fits'):
                newlc,mag=OpenLC(file)
                logging.debug(str(newlc))
                lc=np.vstack((lc, newlc))
            return lc[3:, :], mag
    except:
        logging.info("Failed during Kepler file check. Trying to open as other file type")
    try:
        return OpenLC(fnametrue)
    except:
        logging.error('Cannot read lightcurve file')

def Pmin(Tcen,lc,Tdur):
    return np.max(abs(Tcen-lc[:, 0]))-Tdur

def Vmax(dens,Tcen,lc,Tdur):
    return 86400*(((dens*1411*8*np.pi**2*6.67e-11)/(Pmin*86400*3))**(1/3.0))

def OpenFits(f,fname):
    if f[0].header['TELESCOP']=='Kepler':
        mag=f[0].header['KEPMAG']
        logging.debug('Kepler/K2 LC. Mag = '+str(mag))
        if fname.find('k2sff')!=-1:
            logging.debug('K2SFF file')#K2SFF (Vanderburg) detrending:
            lc=np.column_stack((f[1].data['T'],f[1].data['FCOR'],np.tile(np.median(abs(np.diff(f[1].data['FCOR']))),len(f[1].data['T']))))
        elif fname.find('everest')!=-1:
            logging.debug('Everest file')#Everest (Luger et al) detrending:
            lc=np.column_stack((f[1].data['TIME'],f[1].data['FLUX'],f[1].data['RAW_FERR']))
        elif fname.find('k2sc')!=-1:
            logging.debug('K2SC file')#K2SC (Aigraine et al) detrending:
            lc=np.column_stack((f[1].data['time'],f[1].data['flux'],f[1].data['error']))
        elif fname.find('kplr')!=-1 or fname.find('ktwo')!=-1:
            logging.debug('kplr/ktwo file')
            if fname.find('llc')!=-1 or fname.find('slc')!=-1:
                logging.debug('NASA/Ames file')#NASA/Ames Detrending:
                lc=np.column_stack((f[1].data['TIME'],f[1].data['PDCSAP_FLUX'],f[1].data['PDCSAP_FLUX_ERR']))
            elif fname.find('XD')!=-1 or fname.find('X_D')!=-1:
                logging.debug('Armstrong file')#Armstrong detrending:
                lc=np.column_stack((f[1].data['TIME'],f[1].data['DETFLUX'],f[1].data['APTFLUX_ERR']/f[1].data['APTFLUX']))
        else:
            logging.debug("no file type for "+str(f))
    elif f[0].header['TELESCOP']=='COROT':
        try:
            mag=f[0].header['MAGNIT_V']
            if type(f[0].header['MAGNIT_V'])==str or np.isnan(f[0].header['MAGNIT_V']): mag=f[0].header['MAGNIT_R']
        #If THE MAG_V COLUMN is broken, getting the R-mag
        except:
            mag=f[0].header['MAGNIT_R']
        if f[0].header['FILENAME'][9:12]=='MON':
            #CHJD=2451545.0
            logging.debug('Monochrome CoRoT lightcurve')
            lc=np.column_stack((f[1].data['DATEJD'],f[1].data['WHITEFLUX'],f[1].data['WHITEFLUXDEV']))
        elif f[0].header['FILENAME'][9:12]=='CHR':
            logging.debug('3-colour CoRoT lightcurve')
            #Adding colour fluxes together...
            lc=np.column_stack((f[1].data['DATEJD'],\
                                f[1].data['REDFLUX']+f[1].data['BLUEFLUX']+f[1].data['GREENFLUX'],\
                                f[1].data['GREENFLUXDEV']+f[1].data['BLUEFLUXDEV']+f[1].data['REDFLUXDEV']))
            lc[lc[:,2]==0.0,2]+=np.median(lc[lc[:,2]>0.0,2])
            #Adding this to make up for any times there are no errors at all...
            if np.sum(np.isnan(lc[:,2]))>len(lc[:,2])*0.5:
                lc[:,2]=np.tile(np.median(abs(np.diff(lc[:,1]))),len(lc[:,2]))
    else:
        logging.debug('Found fits file but cannot identify fits type to identify with')
    return lc,mag

def OpenLC(fname):
    '''#Opens lightcurve from filename. Works for:
    * .fits files from Kepler (check OpenFits)
    * .npy 3-column saved Lightcurves
    * .txt 3-column saved Lightcurves
    '''
    fnametrue=str(np.copy(fname))
    fname=fname.split('/')[-1]
    if fname[-5:]=='.fits':
        try:
            f=fits.open(fnametrue)
        except IOError:
            #May have trailing
            f=fits.open(fnametrue,ignore_missing_end=True)
        logging.debug('Fits file')
        lc,mag=OpenFits(f,fname)
    elif fname[-3:]=='npy':
        lc=np.load(fnametrue)
        print(np.shape(lc))
        if len(lc)==2:
            lc=lc[0]
        mag=11.9-2.5*np.log10(np.median(lc[lc[:,1]/lc[:,1]==1.0,1])/313655.0)
    elif fname[-3:]=='txt':
        logging.debug('Found text file. Unpacking with numpy')
        lc=np.genfromtxt(fnametrue)
        mag=11.9-2.5*np.log10(np.median(lc[lc[:,1]/lc[:,1]==1.0,1])/313655.0)
    try:
        #removing nans
        lc=lc[np.logical_not(np.isnan(np.sum(lc,axis=1)))]
        #normalising flux and flux errs:
        lc[:,1:]/=np.median(lc[:,1])
        return lc, mag
    except:
        logging.error("No Lightcurve")
        return None

def Test():
    return 50

def OpenK2_DJA(fnametrue):
    a = fits.open(fnametrue)
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
    a = fits.open(fnametrue)
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
    a = fits.open(fnametrue)
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
        logging.error("Stellar temperature outside spectral type limits")
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
        logging.error("Temp too low")

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

def getStarDat(kic,file):
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
    Ts=None;count=0
    funcs=[ExoFopCat,K2Guess,KOIRead,FitsGuess,AstropyGuess,SetT]
    #Five different ways to estimate temperature (the last one being, pick 5500K+/-1800...)
    while Ts==None:
        func=funcs[count]
        Ts=func(str(kic),file)
        #print func(str(kic),file)
        count+=1
    if len(Ts)<3:
        R,Rerr=RfromT(Ts[0],Ts[1])
        M,Merr=MfromR(R,Rerr)
        return [Ts[0],Ts[1],Ts[1],R,Rerr,Rerr,M,Merr,Merr]
    else:
        #In the HuberCat func we get all the stardat...
        return Ts

def HuberCat(kic,file):
    try:
        import requests
        global dump
        response=requests.get("https://exofop.ipac.caltech.edu/k2/download_target.php?id="+str(kic))
        catfile=response.raw.read()
        T=catfile[catfile.find('Teff'):catfile.find('Teff')+50].split()[1:]
        R=catfile[catfile.find('Radius'):catfile.find('Radius')+50].split()[1:]
        M=catfile[catfile.find('Mass'):catfile.find('Mass')+50].split()[1:]
        return list(np.array([T[0],T[1],T[1],R[0],R[1],R[1],M[0],M[1],M[1]]).astype(float))
    except:
         return None

def HuberCat_2(kic,file=''):
    #Strips online file for a given epic
    #try:
    import requests
    if 1==1:
        global dump
        response=requests.get("https://exofop.ipac.caltech.edu/k2/download_target.php?id="+str(kic))
        catfile=response.text
        outdat=pd.DataFrame(index={"EPIC":int(kic)})
        catsplit=catfile.split('\n')
        MAG=-1000;mags=[]
        STPS=-1000;stpars=[]
        Aliases=[]
        for row in catsplit:
            row=str(row)
            if row=='':
                pass
            elif row[0:7]=='Proposa':
                a=row[13:].replace(' ','')
                outdat['K2_Proposal']=a
            elif row[0:7]=='Aliases':
                a=row[13:].split(',')
                for n in range(len(a)):
                    outdat['ALIAS'+str(n)]=a[n]
            elif row[0:8]=='Campaign':
                outdat['camp']=row[13:].split(',') #list of camps
            elif row[0:2]=='RA':
                outdat['RA']=row[13:30].replace(' ','')
            elif row[0:3]=='Dec':
                outdat['Dec']=row[13:30].replace(' ','')
            elif row.replace(' ','')=='MAGNITUDES:':
                MAG=0
            elif row.replace(' ','')=='STELLARPARAMETERS:':
                MAG=-1000;STPS=0
            elif row=='2MASS NEARBY STARS:':
                STPS=-1000;MAG=-1000
            elif row.replace(' ','')=='SPECTROSCOPY:':
                STPS=-1000
                outdat['Notes']='fu_spectra'
            if MAG>1 and row!='':
                a=row[18:50].split()
                nom=row[0:17].replace(' ','')
                outdat[nom]=float(a[0])
                if len(a)>2: outdat[nom+'_ERR']=float(a[1])
            if STPS>1 and row!='':
                a=row[16:50].split()
                nom=str(row[0:16]).translate(None, " (){}[]/")
                nom='ExoFOP_ST_'+nom[:4]
                if nom in outdat.columns: nom=nom+'_2'
                outdat[nom]=float(a[0])
                if len(a)>2: outdat[nom+'_ERR']=float(a[1])
            MAG+=1;STPS+=1
            return outdat

def HuberCat3(epic):
    return pd.read_csv('http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?&table=k2targets&format=csv&where=epic_number='+str(epic))

def ExoFop(kic):
    if type(kic)==float: kic=int(kic)
    import requests
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO
    response=requests.get("https://exofop.ipac.caltech.edu/k2/download_target.php?id="+str(kic))
    catfile=response.text
    arr=catfile.split('\n\n')
    arrs=pd.Series()
    for n in range(1,len(arr)):
        if n==1:
            df=pd.read_fwf(StringIO(arr[1]),header=None,widths=[12,90]).T
            df.columns = list(df.iloc[0])
            df = df.ix[1]
            df.name=kic
            df=df.rename(index={'RA':'rastr','Dec':'decstr','Campaign':'campaign'})
            arrs=arrs.append(df)
        elif arr[n]!='' and arr[n][:5]=='MAGNI':
            sps=pd.read_fwf(StringIO(arr[2]),header=1,index=None).T
            sps.columns = list(sps.iloc[0])
            sps = sps.ix[1:]
            sps=sps.rename(columns={'B':'mag_b','V':'mag_v','r':'mag_r','Kep':'kepmag','u':'mag_u','z':'mag_z','i':'mag_i','J':'mag_j','K':'mag_k','H':'mag_h','WISE 3.4 micron':'mag_W1','WISE 4.6 micron':'mag_W2','WISE 12 micron':'mag_W3','WISE 22 micron':'mag_W4'})
            newarr=pd.Series()
            for cname in sps.columns:
                newarr[cname]=sps.loc['Value',cname]
                newarr[cname+'_err1']=sps.loc['Uncertainty',cname]
                newarr[cname+'_err2']=sps.loc['Uncertainty',cname]
            newarr.name=kic
            arrs=arrs.append(newarr)
        elif arr[n]!='' and arr[n][:5]=='STELL':
            sps=pd.read_fwf(StringIO(arr[n]),header=1,index=None,widths=[17,14,19,19,22,20]).T
            #print sps
            sps.columns = list(sps.iloc[0])
            sps = sps.ix[1:]
            #print sps
            newarr=pd.Series()
            #print sps.columns
            sps=sps.rename(columns={'Mass':'mass','log(g)':'logg','Log(g)':'logg','Radius':'radius','Teff':'teff','[Fe/H]':'feh','Sp Type':'spectral_type','Density':'dens'})
            for n,i in enumerate(sps.iteritems()):
                if i[1]["User"]=='huber':
                    newarr[i[0]]=i[1]['Value']
                    newarr[i[0]+'_err']=i[1]['Uncertainty']
                elif i[1]["User"]!='macdougall' and i[0] not in sps.columns[:n]:
                    #Assuming all non-Huber
                    newarr[i[0]+'_spec']=i[1]['Value']
                    newarr[i[0]+'_spec'+'_err']=i[1]['Uncertainty']
            newarr.name=kic
            #print newarr

            arrs=arrs.append(newarr)

            #print 'J' in arrs.index
            #print 'WISE 3.4 micron' in arrs.index
        elif arr[n]!='' and arr[n][:5]=='2MASS' and 'mag_J' not in arrs.index:
            #print "SHOULDNT BE THIS FAR..."
            sps=pd.read_fwf(StringIO(arr[n]),header=1)
            #print sps.iloc[0]
            sps=sps.iloc[0]
            if 'rastr' not in arrs.index or 'decstr' not in arrs.index:
                #Only keeping 2MASS RA/Dec if the ExoFop one is garbage/missing:
                arrs['rastr']=sps.RA
                arrs['decstr']=sps.Dec
            elif arrs.rastr =='00:00:00' or arrs.decstr =='00:00:00':
                arrs['rastr']=sps.RA
                arrs['decstr']=sps.Dec
            else:
                sps.drop('RA',inplace=True)
                sps.drop('Dec',inplace=True)
            sps.drop('Pos Angle',inplace=True)
            sps=sps.rename(index={'Designation':'Alias_2MASS','J mag':'mag_J','K mag':'mag_K','H mag':'mag_H','J err':'mag_J_err1','H err':'mag_H_err1','K err':'mag_K_err1','Distance':'2MASS_DIST'})
            sps.name=kic
            arrs=arrs.append(sps)
        elif arr[n]!='' and arr[n][:5]=='WISE ' and 'mag_W1' not in arrs.index:
            #print "SHOULDNT BE THIS FAR..."
            sps=pd.read_fwf(StringIO(arr[n]),header=1,widths=[14,14,28,12,12,12,12,12,12,12,12,13,13])
            #print sps.iloc[0]
            sps= sps.iloc[0]
            sps.drop('RA',inplace=True)
            sps.drop('Dec',inplace=True)
            sps.drop('Pos Angle',inplace=True)
            #Band 1 mag  Band 1 err  Band 2 mag  Band 2 err  Band 3 mag  Band 3 err  Band 4 mag  Band 4 err
            sps=sps.rename(index={'Designation':'Alias_WISE','Band 1 mag':'mag_W1','Band 1 err':'mag_W1_err1','Band 2 mag':'mag_W2','Band 2 err':'mag_W2_err1','Band 3 mag':'mag_W3','Band 3 err':'mag_W3_err1','Band 4 mag':'mag_W4','Band 4 err':'mag_W4_err1','Distance':'WISE_DIST'})
            sps.name=kic
            arrs=arrs.append(sps)
    arrs['StarComment']='Stellar Data from ExoFop/K2 (Huber 2015)'
    arrs.drop_duplicates(inplace=True)
    arrs['id']=kic
    if 'kepmag' not in arrs.index or type(arrs.kepmag) != float or arrs.rastr=='00:00:00':
        extra=HuberCat3(kic)
        if ('k2_kepmag' in extra.index)+('k2_kepmag' in extra.columns)*(arrs.rastr!='00:00:00'):
            arrs['kepmag']=extra.k2_kepmag
            arrs['kepmag_err1']=extra.k2_kepmagerr
            arrs['kepmag_err2']=extra.k2_kepmagerr
            arrs['StarComment']='Stellar Data from VJHK color temperature and main seq. assumption'
        else:
            #No Kepmag in either ExoFop or the EPIC. Trying the vanderburg file...
            try:
                ra,dec,mag=VandDL('91',kic,Namwd,v=1,returnradecmag=True)
                coord=co.SkyCoord(ra,dec,unit=u.degree)
                arrs['rastr']=coord.to_string('hmsdms',sep=':').split(' ')[0]
                arrs['decstr']=coord.to_string('hmsdms',sep=':').split(' ')[1]
                arrs['kepmag']=mag
                arrs['StarComment']='Stellar Data from VJHK color temperature and main seq. assumption'
            except:
                print("no star dat")
                return None
    if 'rastr' in arrs.index and 'decstr' in arrs.index and ('teff' not in arrs.index):#+(type(arrs.teff)!=float)):
        teff,tefferr=AstropyGuess2(arrs.rastr,arrs.decstr)
        arrs['teff']=teff
        arrs['teff_err1']=tefferr
        arrs['teff_err2']=tefferr
    if 'radius' not in arrs.index:
        if 'logg' not in arrs.index:
            T,R,M,l,spec=StellarNorm(arrs['teff'],'V')
            arrs['logg']=l[1]
            arrs['logg_err']=np.average(np.diff(l))
            arrs['StarComment']=arrs['StarComment']+". Assuming Main sequence"
        else:
            T,R,M,l,spec=StellarNorm(arrs['teff'],arrs['logg'])
            arrs['StarComment']=arrs['StarComment']+". Using Lum Class from logg"
        arrs['radius']=R[1]
        arrs['radius_err']=np.average(np.diff(R))
        arrs['mass']=M[1]
        arrs['mass_err']=np.average(np.diff(M))
        arrs['spectral_type']=spec
        arrs['StarComment']=arrs['StarComment']+" R,M,etc from Teff fitting"
    return arrs

def KeplerDat(kic):
    return pd.read_csv('http://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?&table=keplerstellar&where=KepID='+str(kic)+'&format=csv&order=st_vet_date_str%20desc')

def KOIRead(kic,file):
    if str(kic).zfill(8)[0]!=2:
        try:
            import pandas as pd
            #One problem is that this table is not particularly up to date and has huge Radii errors...
            koitable=pd.read_csv(Namwd+'/Tables/KeplerKOItable.csv',skiprows=155)
            match=koitable[koitable['kepid']==int(kic)]
            return list(abs(np.array([match['koi_steff'].values[0], match['koi_steff_err1'].values[0],  match['koi_steff_err2'].values[0], match['koi_srad'].values[0], match['koi_srad_err1'].values[0],  match['koi_srad_err2'].values[0], match['koi_smass'].values[0], match['koi_smass_err1'].values[0],  match['koi_smass_err2'].values[0]])))
        except:
            return None
    else:
        return None

def SetT(kic,file):
    return 5500,1800

def AstropyGuess(kic, file,ra=None,dec=None,kepmag=None):
    #Searches 2MASS catalogue for JHK band fluxes and uses these to estimate a teff.
    try:
        import astropy.io.fits as fits
        from astropy.vo.client import conesearch
        from astropy import coordinates as co
        from astropy import units as u
        if path.exists(file):
            a=fits.open(file)
            ra,dec=a[0].header['RA_OBJ'], a[0].header['DEC_OBJ']
            c=co.SkyCoord(ra*u.deg,dec*u.deg)
            if type(a[0].header['GMAG'])==float and type(a[0].header['RMAG'])==float:
                Vmag=float(a[0].header['GMAG']) - 0.58*(float(a[0].header['GMAG'])-float(a[0].header['RMAG'])) - 0.01
                #Verr=((0.42*float(catdat[24]))**2+(0.58*float(catdat[26]))**2)**0.5
            else:
                #Assuming kepler mag is the same as V (correct for ~Gtype)
                Vmag=a[0].header['KEPMAG']
        elif ra!=None and dec!=None and kepmag!=None:
            c=co.SkyCoord(ra,dec,frame='icrs')
            Vmag=kepmag
        #result = conesearch.conesearch(c, 0.025 * u.degree,catalog_db='2MASS All-Sky Point Source Catalog 2',verbose=False)
        dists=np.sqrt((abs(c.ra.value-result.array['ra'])**2+(abs(c.ra.value-result.array['ra'])))**2)
        mags=result.array['j_m']
        nearest=np.argmin(dists/mags**2)
        print("Found 2MASS source "+str(dists[nearest]*3600)+"\" away with Jmag "+str(mags[nearest]))
        row = result.array.data[nearest]
        T,Terr=coloursCheck(np.array([0, Vmag, row['j_m'],row['h_m'],row['k_m']]))
        #print str(T)+" from 2MASS source"
        return T,Terr
    except:
        return None

def AstropyGuess2(ra,dec):
    from astroquery.vizier.core import Vizier
    import astropy.coordinates as coord
    import astropy.units as u
    result = Vizier.query_region(coord.SkyCoord(ra, dec, unit=(u.hourangle, u.deg)),width="0d0m40s", height="0d0m40s",catalog=["NOMAD","allwise"])
    nomadseps=coord.SkyCoord(ra, dec, unit=(u.hourangle, u.deg)).separation(coord.SkyCoord(result[0]['RAJ2000'],result[0]['DEJ2000']))
    wiseseps=coord.SkyCoord(ra, dec, unit=(u.hourangle, u.deg)).separation(coord.SkyCoord(result[1]['RAJ2000'],result[1]['DEJ2000']))
    print(str(np.min(nomadseps))+" < Nomad | Wise > "+str(np.min(wiseseps)))
    wise=result[1][np.argmin(wiseseps)]
    nomad=result[0][np.argmin(nomadseps)]
    if 'e_Jmag' in wise.columns:
        match=wise
        colnames=['Bmag','Vmag','Rmag']
        for c in range(3):
            if colnames[c] in nomad.columns and type(nomad.columns[colnames[c]])==float:
                match.table[colnames[c]]=nomad[colnames[c]]
                match.table['e_'+colnames[c]]=nomad['r_'+colnames[c]]
    return coloursCheck2(match)

def coloursCheck2(match,niter=1800):
    import scipy.interpolate as interp
    '''
    #Estimate stellar Temperature from b,v,r,j,h,k magnitudes
    Args:
    mags: array of b,v,r,j,h,k magnitudes
    Returns:
    T : Stellar Temperature
    Terr: Stellar Temperature error from the STD of the distribution of each temp estimate \
        (to a minimum of 150K precision)
    '''
    phot=np.array(['B','V','R','J','H','K','W1','W2','W3','W4'])
    photerrs=np.array(['e_'+p+'mag' for p in phot])
    mask=np.zeros(len(phot)).astype(bool)
    #Checking if the magnitude in the match is a float, else masking it out
    for n in range(len(phot)):
        try:
            if float(match[phot[n]+'mag'])/float(match[phot[n]+'mag'])==1.0:
                mask[n]=True
            else:
                print("no "+phot[n])
        except:
            print("no "+phot[n])
    #Checking if the error is a valid float, otherwise setting err as 0.8
    for err in photerrs[mask]:
        try:
            if float(match[err])/float(match[err])!=1.0:
                match.table[err]=0.8
        except:
            #Need to remove the column to set a new one...
            if err in match.columns:
                match.table.remove_column(err)
            match.table[err]=0.8
    #Getting table
    MScols=pd.DataFrame.from_csv(Namwd+"/Tables/ColoursTab3.txt")
    #MScols=pd.read_table(Namwd+"/Tables/ColoursTab2.txt",header=20,delim_whitespace=True,na_values=['...','....','.....','......'])
    order=['B-V','V-R','R-J','J-H','H-K','K-W1','W1-W2','W2-W3','W3-W4']#Good from B0 to K5 with Wise. Without Wise good to M8V
    os=np.array([o.split('-') for o in order]).swapaxes(0,1)
    cols=[]
    #print phot[mask]
    for p in range(np.sum(mask)-1):
        #print phot[mask][p]+'-'+phot[mask][p+1]+"  -  "+str(float(match[phot[mask][p]+'mag'])-float(match[phot[mask][p+1]+'mag']))
        col=phot[mask][p]+'-'+phot[mask][p+1]
        cols+=[col]
        if col not in MScols:
            #Making it from others
            if phot[mask][p] in os[0] and phot[mask][p+1] in os[1]:
                #Adding column to MS cols df.
                c0=np.where(os[0]==phot[mask][p])[0]
                c1=np.where(os[1]==phot[mask][p+1])[0]
                MScols[phot[mask][p]+'-'+phot[mask][p+1]]=np.sum(order[c0:c1])
            else:
                print("problem with "+phot[mask][p]+'-'+phot[mask][p+1])
    T=np.zeros((len(cols),2))
    for c in range(len(cols)):
        func=interp.interp1d(MScols[cols[c]],MScols['logT'])
        m1=float(match[cols[c].split('-')[0]+'mag'])
        m2=float(match[cols[c].split('-')[1]+'mag'])
        mag1s=np.random.normal(m1,float(match['e_'+cols[c].split('-')[0]+'mag']),niter)
        mag2s=np.random.normal(m2,float(match['e_'+cols[c].split('-')[1]+'mag']),niter)
        tT=[]
        for n in range(niter):
            try:
                tT+=[func(mag1s[n]-mag2s[n])]
            except:
                tT+=[np.nan]
        tT=np.array(tT)
        tT=tT[~np.isnan(tT)]
        if (len(tT)/float(niter))>0.5:
            T[c,0]=np.median(tT)
            T[c,1]=np.std(tT)
            #print cols[c]+"  -  "+str(np.power(10,T[c,0]))
        else:
            T[c,0]=np.nan
            T[c,1]=np.nan
    T=T[~np.isnan(np.sum(T,axis=1))]
    T[T[:,1]==0.0,1]=0.5
    avT=np.average(T[:,0],weights=T[:,1]**-2)
    sig=np.sqrt(np.average((T[:,0]-avT)**2,weights=T[:,1]**-2))
    return np.power(10,avT),sig*np.power(10,avT)

'''
def getStarDat(kic,file=''):

    # Uses photometric information in the Kepler fits file of each lightcurve to estimate Temp, SpecType, Radius & Mass
    #Colours include B-V,V-J,V-H,V-K.
    # V is estimated from g and r mag if available
    # otherwise assumes KepMag=V (which is true for G-type stars, but diverges fot hotter/cooler stars)
    # JHK magnitudes also calculated independant of Vmags (which are often inaccurate).
    # Any disagreement is therefore obvious in Terr

    #INPUTS:
    #kplr filename

    #OUTPUTS:
    #StarData Object [T,Terr,R,Rerr,M,Merr]


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
            koitable=pd.read_csv(Namwd+'/Tables/KeplerKOItable.csv',skiprows=155)
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
    logging.warning("Unable to get Star Data from catalogues or colours.")
    #Does this kill the script here?
    pass
'''

def FitsGuess(kic, file):
    #Otherwise using fits colours
    from astropy.io import fits
    try:
        lcfits=fits.open(file)
    except ValueError:
        print("Lightcurve is not a fits file and data not in EPIC/KIC catalogues.")
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
    try:
        Temp,Terr= coloursCheck(np.array([0,Vmag, Jmag, Hmag, Kmag]))
        return Temp,Terr
    except:
        return None


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
    MScols=np.genfromtxt(Namwd+"/Tables/MainSeqColours.csv", delimiter=',')
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
    if type(Rad) is float or type(Rad) is int:
        Rad=[Rad]
    M=np.zeros_like(Rad)
    M[Rad<=1.5]=((2430+3390*Rad[Rad<=1.5])*(1.3333*np.pi*(Rad[Rad<=1.5]*Rearth)**3))/Mearth
    M[(Rad>1.5)*(Rad<=4.0)]=2.69*(Rad[(Rad>1.5)*(Rad<=4.0)]**0.93)
    giants=(Rad>4.)*(Rad<=16.0)
    if giants.sum()>0:
        import scipy.interpolate
        o=np.array([[3.99, 9.0,10.5,12.0,13.0,14.0,14.5, 15.5, 16.0],[10.0, 100.0,300.0,1000.0,1100.0,1200.0, 1350.0, 5000.0,  10000.0]])
        f= scipy.interpolate.interp1d(o[0, :],o[1, :])
        M[giants]=f(Rad[giants])
    M[Rad>16.0]=333000*(Rad[Rad>16.0]/109.1)**1.5#(such that 1.5 ~10000.0)
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
    #print "Plotting Lightcurve"
    if col==0:
        col='#DDDDDD'#'#28596C'
        #col= '#'+"".join([random.choice('0123456789abcdef') for n in xrange(6)])
    plt.ion(); plt.figure(fig); plt.title(titl);plt.xlabel('time');plt.ylabel('Relative flux');
    plt.errorbar(lc[:, 0]+hoff, lc[:, 1]+offset, yerr=lc[:, 2], fmt='.',color=col)

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
    reader=csv.reader(open(Namwd+'/Tables/EPICcatalogue.csv'))
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

def MS_approx(colours, pm=0, errors=0):
    '''#If K2, finds magnitudes from EPIC catalogue.
    #If CoRoT, uses online CoRoT database.
    #If neither, uses astropy to search for NOMAD catalogue data
    #It then uses these to make a best-fit guess for spectral type, mass and radius
    #It can also take just the B-V, V-J, V-H and V-K colours.
    #Fitzgerald (1970 - A&A 4, 234) http://www.stsci.edu/~inr/intrins.html
    #[N Sp. Type 	U-B 	(B-V) 	(V-R)C 	(V-I)C 	(V-J) 	(V-H) 	(V-K) 	(V-L) 	(V-M) 	(V-N) 	(V-R)J 	(V-I)J  Temp    Max Temp]
    Returns:
    #   Spec Types, Temps, Radii, Masses.
    '''
    import requests
    global dump
    #ROWS: EPIC,RA,Dec,Data Availability,KepMag,HIP,TYC,UCAC,2MASS,SDSS,Object type,Kepflag,pmra,e_pmra,pmdec,e_pmdec,plx,e_plx,Bmag,e_Bmag,Vmag,e_Vmag,umag,e_umag,gmag,e_gmag,rmag,e_rmag,imag,e_imag,zmag,e_zmag,Jmag,e_Jmag,Hmag,e_Hmag,Kmag,e_Kmag
    if mission=='K2':
        ExoFopCat(kic)
    elif mission=='CoRoT':
        tbd=0
    reader=csv.reader(open(Namwd+'/Tables/EPICcatalogue.csv'))
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

def CoRoTDat(id,returnlc=True,fitsfolder='.'):
    import requests
    global dump
    #response=requests.get('http://cesam.oamp.fr/exodat/web-services/get-data-by-corot-id?catalog=exo2mass&corotid='+str(id)+'&header=true&format=csv')
    response=requests.get('http://cesam.oamp.fr/exodat/web-services/get-data-by-corot-id?catalog=exo2mass&corotid='+str(id)+'&header=true&format=csv')
    arr=response.text.encode('ascii','ignore').split('\n')
    arr=np.vstack((np.array(arr[0].split(',')),np.array(arr[1].split(','))))
    corotdf=pd.DataFrame(arr[1:],columns=arr[0])

    response2=requests.get("http://cesam.oamp.fr/exodat/web-services/get-data-by-corot-id?catalog=observations&corotid="+str(id)+"&header=true&format=csv")
    arr2=response2.text.encode('ascii','ignore').split('\n')
    arr2=np.vstack(([arr2[n].split(',') for n in range(len(arr2))][:-1]))
    obsdf=pd.DataFrame(arr2[1:],columns=arr2[0])

    #response3=requests.get("http://cesam.oamp.fr/exodat/web-services/get-data-by-corot-id?catalog=obscat&corotid="+str(id)+"&header=true&format=csv")
    #arr3=response3.text.encode('ascii','ignore').split('\n')
    #arr3=np.vstack(([arr3[n].split(',') for n in range(len(arr3))][:-1]))
    #obscatdf=pd.DataFrame(arr3[1:],columns=arr3[0])
    #print obscatdf

    if returnlc:
        #Using the observations dataframe to get the lightcurve
        fs,files=[],[]
        for row in obsdf.iterrows():
            f,fname=getCoRoTlc(row[1].run_id,row[1].corot_id,fitsfolder)
            fs+=[f]
            files+=[fname]
        count=0
        lc=np.zeros((300000,3))
        #May be multiple fields observed...
        for n in range(len(fs)):
            templc,mag=OpenFits(fs[n],files[n])
            templc=templc[nonan(templc)]
            templc[:,1:]/=np.median(templc[:,1])
            templc=templc[AnomMedianCut(templc,n=31,kind='high')]
            lc[count:count+len(templc[:,0])]
            count+=len(templc[:,0])
        #Removing unused lightcurve rows
        lc=lc[:count,:]
        return lc, pd.concat([corotdf,obsdf], axis=1)
    else:
        if np.shape(obsdf)[0]==1:
            return pd.concat([corotdf,obsdf], axis=1)
        else:
            df0= pd.concat([corotdf.iloc[0],obsdf.iloc[0]], axis=1)
            return df0.append([pd.concat([corotdf.iloc[0],obsdf.iloc[n]], axis=0) for n in range(1,np.shape(obsdf)[0])])

#def getCoToRlc(obsdf):
def getCoRoTlc(field,corotid,fitsfolder='./'):
    from urllib.request import urlopen
    import astropy.io.fits as fits
    poss_times_by_field=np.load('/Users/Hugh/ownCloud/PhD/K2/SingleTransits/CoRoT_wget_ts.npy')
    possts=poss_times_by_field[poss_times_by_field[:,0]==field]
    f=[];n=0
    tinset='failed'
    while n<len(possts):
        tinsert=possts[n,2].replace('XXXXXXXXXX',str(corotid).zfill(10))
        try:
            urlfitsname="http://exoplanetarchive.ipac.caltech.edu/data/ETSS/corot_"+possts[n,1]+"/FITSfiles/"+field+"/"+tinsert+".fits"
            ret = urlopen(urlfitsname)
            if ret.code == 200:
                if path.exists(fitsfolder+tinsert+".fits"):
                    f=fits.open(fitsfolder+tinsert+".fits")
                else:
                    f=fits.open(urlfitsname)
                    f.writeto(fitsfolder+tinsert+".fits")
                n+=100
            else:
                n+=1
            #os.system("wget -O "+tinsert+".fits http://exoplanetarchive.ipac.caltech.edu/data/ETSS/corot_"+ft[1]+"/FITSfiles/"+field+"/"+tinsert+".fits -a corot_"+ft[1]+"_"+field+"_wget.log")
        except:
            #print "No luck with "+str(n)
            n+=1
    print(f)
        #"wget -O AN2_STAR_"+obsdf.loc[0,"corot_id"]+"_20070203T130553_20070401T235518.fits http://exoplanetarchive.ipac.caltech.edu/data/ETSS/corot_exo/FITSfiles/"+field+"/EN2_STAR_MON_"+obsdf.loc[0,"corot_id"]+"_20070203T130553_20070401T235518.fits -a corot_exo_"+field+"_wget.log",\,\
                   #"wget -O AN2_STAR_"+obsdf.loc[0,"corot_id"]+"_20070203T130553_20070401T235518.fits http://exoplanetarchive.ipac.caltech.edu/data/ETSS/corot_exo/FITSfiles/"+field+"/EN2_STAR_MON_"+obsdf.loc[0,"corot_id"]+"_20070203T130553_20070401T235518.fits -a corot_exo_"+field+"_wget.log",\\
    return f,fitsfolder+tinsert+".fits"
    #for field in
    #"wget -O EN2_STAR_MON_0102693924_20070203T130553_20070401T235518_lc.tbl http://exoplanetarchive.ipac.caltech.edu/data/ETSS/corot_exo/lightcurve/IRa01//EN2_STAR_MON_0102693924_20070203T130553_20070401T235518_lc.tbl -a corot_exo_IRa01_wget.log"
    #%"wget -O EN2_STAR_MON_0102693924_20070203T130553_20070401T235518.fits http://exoplanetarchive.ipac.caltech.edu/data/ETSS/corot_exo/FITSfiles/IRa01/EN2_STAR_MON_0102693924_20070203T130553_20070401T235518.fits -a corot_exo_IRa01_wget.log"


def CoRoTStarPars(corotdf,specarr=None,retcomment=False):
    #Loading array to compute stellar parameters from corot dataframe
    #returns T,R,M,logg
    #Always using values[0] - ie. assuming only the top line of the df has results
    if type(specarr) is not pd.core.frame.DataFrame:
        specarr=pd.DataFrame.from_csv('../TypeClassTable.csv')
    out=[None,None]
    if corotdf.color_temperature=='':
        if corotdf.spectral_type!='' and corotdf.luminosity_class!='':
            #row=specarr.loc[corotdf.spectral_type.values[0]]
            StarDat=StellarNorm(corotdf.spectral_type,corotdf.luminosity_class,retsig=True)
            out=[StarDat,"Stellar data from CoRoT spectype & CoRoT lumclass"]
        elif corotdf.spectral_type!='' and corotdf.luminosity_class=='':
            #Assuming Main Sequence:
            out=[StellarNorm(corotdf.spectral_type,['V'],retsig=True),"Stellar data from spectype & assuming main seq."]
            #T=np.exp(10,specarr.loc[corotdf.spectral_type,'logT_V'].values[0][0])
            #stardat=MS_fit(T)
            #return stardat
        else:
            #No data...
            logging.error("No stellar data from CoRoT ExoDat")
            return None
    else:
        if corotdf.luminosity_class=='':
            spec = specarr.iloc[np.nanargmin(abs(np.log10(float(corotdf.color_temperature))-specarr.logT_V.values))].name
            out=[StellarNorm(spec,['V'],retsig=True),"Stellar data from CoRoT color temperature & assuming main seq."]
            #stardat=MS_fit(corotdf.color_temperature.values[0])
            #return stardat
        elif corotdf.luminosity_class!='':
            lum=corotdf.luminosity_class
            if lum=='I': lum='Iab'
            Tspec='logT_'+lum
            spec = specarr.iloc[np.nanargmin(abs(np.log10(float(corotdf.color_temperature))-specarr[Tspec].values))].name
            #row=specarr.loc[corotdf.spectral_type]
            out=[StellarNorm(spec,corotdf.luminosity_class,retsig=True),"Stellar data from CoRoT color temperature & CoRoT luminosity class"]

    if retcomment:
        return out
    else:
        return out[0]

def StellarNorm(spec,lum,retsig=True,n=1000):
    #Gets TRMl from spectral type and luminosity class... OR from Temperature and logg
    try:
        specarr=pd.DataFrame.from_csv('~/ownCloud/PhD/K2/SingleTransits/TypeClassTable.csv')
    except:
        specarr=pd.DataFrame.from_csv('~/PhD/K2/SingleTransits/TypeClassTable.csv')
    if type(spec)!=str:
        #Spec is actually a temp...
        spec=GetSpecType(spec,specarr)
        retspec=True
    else:
        retspec=False
    if type(lum)!=str:
        #Lum is actually
        try:
            lum=GetLumClass(lum,spec,specarr)
        except:
            lum='V'
    if lum=='I':lum='Iab'
    #Returns stellar T,R,M and logg, but with either errors (assuming 1-type error in T, and 0.5-class error in lum)
    from scipy.interpolate import CloughTocher2DInterpolator as ct2d
    #interpolating...
    Ms=specarr.columns[:7]
    Rs=specarr.columns[7:14]
    Ts=specarr.columns[14:21]
    gs=specarr.columns[21:]
    nl=np.where(np.array(['ZAMS','V','IV','III','II','Ib','Iab','Ia'])==lum)[0]
    ns=np.where(specarr.index.values==spec)[0]

    xyarr=np.array([(x,y) for x in range(len(specarr.index.values)) for y in range(8)])
    Marr=specarr.iloc[:,:8].as_matrix().reshape(-1,1)
    Mintrp=ct2d(xyarr[np.where(Marr/Marr==1.0)[0]],Marr[np.where(Marr/Marr==1.0)[0]])

    Tarr=specarr.iloc[:,16:24].as_matrix().reshape(-1,1)
    Tintrp=ct2d(xyarr[np.where(Tarr/Tarr==1.0)[0]],Tarr[np.where(Tarr/Tarr==1.0)[0]])

    Rarr=specarr.iloc[:,8:16].as_matrix().reshape(-1,1)
    Rintrp=ct2d(xyarr[np.where(Rarr/Rarr==1.0)[0]],Rarr[np.where(Rarr/Rarr==1.0)[0]])

    garr=specarr.iloc[:,24:].as_matrix().reshape(-1,1)
    gintrp=ct2d(xyarr[np.where(garr/garr==1.0)[0]],garr[np.where(garr/garr==1.0)[0]])

    n=6000 if retsig else 1
    #Normal dists around spectral type (+/-1.5 types) and lum class (+/-0.33 classes)
    logT=Tintrp(np.random.normal(ns,1.5,n),np.random.normal(nl,0.3,n))
    logR=Rintrp(np.random.normal(ns,1.5,n),np.random.normal(nl,0.3,n))
    logM=Mintrp(np.random.normal(ns,1.5,n),np.random.normal(nl,0.3,n))
    logg=gintrp(np.random.normal(ns,1.5,n),np.random.normal(nl,0.3,n))
    if n>1:
        if retspec:
            return np.power(10,np.percentile(logT[logT/logT==1.0],[16, 50, 84])),\
                   np.power(10,np.percentile(logR[logR/logR==1.0],[16, 50, 84])),\
                   np.power(10,np.percentile(logM[logM/logM==1.0],[16, 50, 84])),\
                   np.percentile(logg[logg/logg==1.0],[16, 50, 84]),spec
        else:
            return np.power(10,np.percentile(logT[logT/logT==1.0],[16, 50, 84])),\
                   np.power(10,np.percentile(logR[logR/logR==1.0],[16, 50, 84])),\
                   np.power(10,np.percentile(logM[logM/logM==1.0],[16, 50, 84])),\
                   np.percentile(logg[logg/logg==1.0],[16, 50, 84])
    else:
        if retspec:
            return np.power(10,logT[0][0]),np.power(10,logR[0][0]),np.power(10,logM[0][0]),logg[0][0],spec
        else:
            return np.power(10,logT[0][0]),np.power(10,logR[0][0]),np.power(10,logM[0][0]),logg[0][0]

def GetSpecType(T,specarr=None):
    if type(specarr)==type(None):
        specarr=pd.DataFrame.from_csv('~/ownCloud/PhD/K2/SingleTransits/TypeClassTable.csv')
    return specarr.iloc[np.nanargmin(abs(np.log10(float(T))-specarr.logT_V.values))].name

def GetLumClass(logg,spect,specarr=None):
    if type(specarr)==type(None):
        specarr=pd.DataFrame.from_csv('~/ownCloud/PhD/K2/SingleTransits/TypeClassTable.csv')
    loggarr=specarr.loc[spect,['logg_ZAMS','logg_V','logg_IV','logg_III','logg_II','logg_Ib','logg_Iab','logg_Ia']].values
    return ['ZAMS','V','IV','III','II','Ib','Iab','Ia'][np.argmin(abs(logg-loggarr))]

def EstMRT(df):
    mks=pd.DataFrame.from_csv('../SpecTypes.csv')
    match=mks[(mks['Spec Type']==df.spectral_type.values[0])*(mks['Luminosity Class']==df.luminosity_class.values[0])]
    cols=['Mass Mstar/Msun','Luminosity Lstar/Lsun','Radius Rstar/RsunvTemp K','Color Index B-V','Abs Mag','Mv','Bolo Corr BC(Temp)','Bolo Mag Mbol']
    df[col] = [match.col for col in cols]
    return df

def ExoFopCat(kic,file=''):
    import pandas as pd
    EPICreqs=pd.DataFrame.from_csv(Namwd+'/Tables/EPICreqs.csv')
    if kic in EPICreqs.index.values:
        logging.debug("in Exofop csv already")
        return list(EPICreqs.loc[kic,['ExoFOP_Teff','ExoFOP_Teff_ERR','ExoFOP_Radi','ExoFOP_Radi_ERR','ExoFOP_Mass','ExoFOP_Mass_ERR']])
    else:
    #Strips online file for a given epic
        import requests
        global dump
        response=requests.get("https://exofop.ipac.caltech.edu/k2/download_target.php?id="+str(kic))
        catfile=response.text
        outdat=pd.DataFrame(index={int(kic)})
        catsplit=catfile.split('\n')
        MAG=-1000;mags=[]
        STPS=-1000;stpars=[]
        Aliases=[]
        for row in catsplit:
            row=str(row)
            if row=='':
                pass
            elif row[0:7]=='Proposa':
                a=row[13:].replace(' ','')
                outdat['K2_Proposal']=a
            elif row[0:7]=='Aliases':
                a=row[13:].split(',')
                for n in range(len(a)):
                    outdat['ALIAS'+str(n)]=a[n]
            elif row.replace(' ','')=='MAGNITUDES:':
                MAG=0
            elif row.replace(' ','')=='STELLARPARAMETERS:':
                MAG=-1000;STPS=0
            elif row=='2MASS NEARBY STARS:':
                STPS=-1000;MAG=-1000
            elif row.replace(' ','')=='SPECTROSCOPY:':
                STPS=-1000
                outdat['Notes']='fu_spectra'
            if MAG>1 and row!='':
                a=row[18:50].split()
                nom=row[0:17].replace(' ','')
                outdat[nom]=float(a[0])
                if len(a)>2: outdat[nom+'_ERR']=float(a[1])
            if STPS>1 and row!='':
                a=row[16:50].split()
                nom=str(row[0:16]).translate(None, " (){}[]/")
                nom='ExoFOP_'+nom[:4]
                if nom in outdat.columns: nom=nom+'_2'
                outdat[nom]=float(a[0])
                if len(a)>=2: outdat[nom+'_ERR']=float(a[1])
            MAG+=1;STPS+=1
        EPICreqs.append(outdat)
    return list(outdat.loc[outdat.index.values[0],['ExoFOP_Teff','ExoFOP_Teff_ERR','ExoFOP_Radi','ExoFOP_Radi_ERR','ExoFOP_Mass','ExoFOP_Mass_ERR']])

def CutAnomDiff(flux,thresh=4.2):
    #Uses differences between points to establish anomalies.
    #Only removes single points with differences to both neighbouring points greater than threshold above median difference (ie ~rms)
    #Fast: 0.05s for 1 million-point array.
    #Must be nan-cut first
    diffarr=np.vstack((np.diff(flux[1:]),np.diff(flux[:-1])))
    diffarr/=np.median(abs(diffarr[0,:]))
    anoms=np.hstack((True,((diffarr[0,:]*diffarr[1,:])>0)+(abs(diffarr[0,:])<thresh)+(abs(diffarr[1,:])<thresh),True))
    return anoms


def AnomMedianCut(lc,sigclip=3.2,n=18,iter=5,kind='both'):
    #Cuts regions that are [sigclip] times away than the median of the 2*n-long region around each point
    #Performs this [iter] times. n must be > largest length of anomalous groups of points
    #Can be used for only 'high' points (kind='high'),'low' points (kind='low'), or both (kind='both': DEFAULT)
    #Purely in numpy version: about 4 times quicker than a single for loop on the equivalent data.

    #creating mask:
    mask=np.ones(len(lc), np.bool)
    #four iterations...
    for p in range(iter):
        #creating iteration-specific lc and mask from masters:
        inlc=lc[mask]
        inmask=np.ones(len(inlc),np.bool)
        #stacking data into rows n*2 long (each excluding the central value)
        stacks=np.column_stack(([inlc[i:(i-(n*2+1)),1] for i in range(n)+range(n+1,n*2+1)]))
        #performing median and std on these datarows and finding where the value is [sigclip]*sigma higher than median
        if kind=='both':
            wh=n+np.where( abs(inlc[n:(-1-1*n),1]-np.median(stacks,axis=1)) > (sigclip*np.std(stacks,axis=1)) )[0]
        elif kind=='high':
            wh=n+np.where(inlc[n:(-1-1*n),1]>(np.median(stacks,axis=1)+sigclip*np.std(stacks,axis=1)))[0]
        elif kind=='low':
            wh=n+np.where(inlc[n:(-1-1*n),1]<(np.median(stacks,axis=1)-sigclip*np.std(stacks,axis=1)))[0]
        else:
            #kind must be 'both', 'high' or 'low'
            raise ValueError(kind)
        #Can find + and - anomalies with ( abs(inlc[n:(-1-1*n),1]-np.median(stacks,axis=1)) > (sigclip*np.std(stacks,axis=1)) )
        #setting the mask of those anomalous values to False
        inmask[wh]=False
        #Outgoing mask must unchanged in length. Newly masked points are added to the master mask
        mask[mask]=inmask
    return mask

def nonan(lc):
    return np.logical_not(np.isnan(np.sum(lc,axis=1)))

def K2Centroid(kic,camp,Tcen,Tdur):
    '''
    #Looking for centroid shifts due to blended eclipsing binary.
    #If PDC/hlsff/etc - can use shifts given in lightcurve file
    '''
    "https://archive.stsci.edu/missions/k2/target_pixel_files/c102/229000000/21000/ktwo229021605-c102_lpd-targ.fits.gz"
    tpfname='ktwo'+str(int(kic))+'-c'+str(int(camp))+'_llc.fits'
    if not os.path.exists(tpfname):
        print('Downloading K2 TPF file from MAST...')
        comm='wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -R \'index*\' -erobots=off https://archive.stsci.edu/pub/k2/lightcurves/c'+str(int(camp))+'/'+str(int(kic))[:4]+'00000/'+str(int(kic))[4:6]+'000/'+tpfname
        os.system(comm)
        #Getting K2 LC
    orglc=fits.open(tpfname)
    os.system('rm '+tpfname)
    cents=np.column_stack((orglc[1].data['TIME'],orglc[1].data['MOM_CENTR1'],orglc[1].data['MOM_CENTR1_ERR'],orglc[1].data['MOM_CENTR2'],orglc[1].data['MOM_CENTR2_ERR']))
    cents=cents[nonan(cents)]
    intr=abs(cents[:,0]-Tcen)<(0.5*Tdur)
    #Median centroid difference for in-transit compared to out-of-transit
    meddiff0=np.sqrt((np.median(cents[intr,1])-np.median(cents[~intr,1]))**2+(np.median(cents[intr,3])-np.median(cents[~intr,3]))**2)
    #Generating random centroid positions and computing centroid shift for these:
    randtcens=np.random.random(100)*(cents[-1,0]-cents[0,0])+cents[0,0]
    meddiffs=[]
    for tc in randtcens:
        intr=abs(cents[:,0]-tc)<(0.5*Tdur)
        meddiffs+=[np.sqrt((np.median(cents[intr,1])-np.median(cents[~intr,1]))**2+(np.median(cents[intr,3])-np.median(cents[~intr,3]))**2)]
    meddiffs=np.array(meddiffs)
    #Comparing transit centroid shift to actual centroid shift.
    centshift=meddiff0/np.median(abs(meddiffs[np.logical_not(np.isnan(meddiffs))]))#Centroid shift in sigma (compared to 100 randomly selected transits across the lightcurve)
    return centshift
