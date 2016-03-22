import numpy as np
import pandas as pd
import glob
from os import path
import csv

'''THIS IS NOW REDUNDANT

Namwd = path.dirname(path.realpath(__file__))

def KicData(kic):
    if str(int(kic)).zfill(9)[0]=='2':
        #Getting EPIC table
        print 'K2 lightcurve'
        return None
    else:
        koitable=pd.read_csv('KeplerKOItable.csv',skiprows=155)
        match=koitable[koitable['kepid']==kic]
        return match

def getStarDat(kic, file):
    '''
    Gets Star Data from EPIC catalogue or from BVJHK colours
    '''
    if str(int(kic)).zfill(9)[0]=='2':
        #Getting EPIC table

        Ts, specs,  Rss, Mss = K2Guess(int(kic))
        #If a value from the above arrays is 2sigma out in temperature, cut...
        ans=anomaly(Ts,0.5,2.2)
        T=np.array([np.average(Ts[ans]), np.std(Ts[ans]), np.std(Ts[ans])])
        R=np.array([np.average(Rss[ans]), np.std(Rss[ans]), np.std(Rss[ans])])
        M=np.array([np.average(Mss[ans]), np.std(Mss[ans]), np.std(Mss[ans])])
        StarDat=list(np.hstack((T, R, M)))
        mins=np.array([0, 0.035, 0.035, 0, 0.1, 0.1, 0, 0.1, 0.1])
        tile=np.hstack((np.tile(StarDat[0],3),np.tile(StarDat[3],3),np.tile(StarDat[6],3)))
        #[Ts,Ts,Ts,Rs,Rs,Rs,Ms,Ms,Ms * 
        return list(np.where(np.array(StarDat)<mins*tile, mins*tile, StarDat))
        #np.where(StarDat*mins < np.hstack((np.tile(StarDat[0],3),np.tile(StarDat[2],3),np.tile(StarDat[5],3))), StarDat*mins, StarDat)       
    else:
        import pandas as pd
        koitable=pd.read_csv(Namwd+'KeplerKOItable.csv',skiprows=155)
        match=koitable[koitable['kepid']==int(kic)]
        return list(abs(np.array([match['koi_steff'].values[0], match['koi_steff_err1'].values[0],  match['koi_steff_err2'].values[0], match['koi_srad'].values[0], match['koi_srad_err1'].values[0],  match['koi_srad_err2'].values[0], match['koi_smass'].values[0], match['koi_smass_err1'].values[0],  match['koi_smass_err2'].values[0]])))

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
    reader=csv.reader(open('/home/astro/phrnbe/ownCloud/PhD/K2/EPICcatalogue.csv'))
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

def GiantCheck(J, H, PMra, PMdec):
    #Uses J & H mag as well as proper motions to check if a star is likely main sequence or a giant
    # Formula from Collier-Cameron 2007. Within 1.0 RPM of curve gives threshold probabilities that have been pulled from my ass
    #Returns -1.0 if parameters missing
    if J!='' and H!='' and PMra!='' and PMdec!='':
        JminH=float(J)-float(H)
        PM=(float(PMra)**2+float(PMdec)**2)**0.5
        RPM=float(J)+5*np.log10(PM)
        diff=RPM-np.polyval(np.array([-141.25,473.18,-583.6,313.42,-58]),JminH)
        if diff<-1:
            return 1.0
        elif diff<0:
            return 0.75
        elif diff<-1:
            return 0.25
        elif diff>=1:
            return 0.0
    else:
        return -1.0

def anomaly(arr,StdStabilised=0.95, StdLimit=2):
    #This module returns the index of array members who are within 2sigma of the median, and repeats removing outer members until the std change has sufficiently stabilised (eg to within 2%).
    # syntax: 
    StdDiff=0
    arrlen0=len(arr);arrind=np.arange(0,len(arr),1)
    while StdDiff<StdStabilised and len(arr)>0.9*arrlen0:
        std0=np.std(arr[arrind])
        arrind=arrind[np.where(abs(arr[arrind]-np.median(arr[arrind]))<(StdLimit*np.std(arr[arrind])))]
        std1=np.std(arr[arrind])
        StdDiff=std1/std0 #Parameter is the difference in stds from before to now
    return arrind

def coloursCheck(colours0):
    import scipy.interpolate
    MScols=np.genfromtxt('/home/astro/phrnbe/ownCloud/PhD/K2/MainSeqColours.csv', delimiter=',',  dtype=[('f0','<f8'), ('f1','S4'),('f2','<f8'), ('f3','<f8'),('f4','<f8'),('f5','<f8'),('f6','<f8'),('f7','<f8'),('f8','<f8'),('f9','<f8'),('f10','<f8'),('f11','<f8'),('f12','<f8'),('f13','<f8'),('f14','<f8'),('f15','<f8')])
    colmax=np.array([np.max(MScols['f3']), np.max(MScols['f6']), np.max(MScols['f8']), np.max(MScols['f7'])])
    colmin=np.array([np.min(MScols['f3']), np.min(MScols['f6']), np.min(MScols['f8']), np.min(MScols['f7'])])
    colOff=np.where((colours0>colmax) | (colours0<colmin))[0]
    #print colours0;  print colOff  #print colmax;    print colmin;#    
    if 0 in colOff:
        BVi = lambda n: 9000.0
    else:
       BVi= scipy.interpolate.interp1d(MScols['f3'],MScols['f14'])
    if 1 in colOff:
        VJi= lambda n: 9000.0
    else:
        VJi=scipy.interpolate.interp1d(MScols['f6'],MScols['f14'])
    if 2 in colOff:
        VHi= lambda n: 9000.0
    else:
       VHi= scipy.interpolate.interp1d(MScols['f8'],MScols['f14'])
    if 3 in colOff:
        VKi= lambda n: 9000.0
    else:
       VKi= scipy.interpolate.interp1d(MScols['f7'],MScols['f14'])
    #colours=np.copy(colours0)
    colours=colours0
    Ts=np.array([BVi(colours[0]), VJi(colours[1]), VHi(colours[2]), VKi(colours[3])])
    Specs=np.array([TypefromT(Ts[0]), TypefromT(Ts[1]), TypefromT(Ts[2]), TypefromT(Ts[3])])
    Rs=np.array([RfromT(Ts[0]), RfromT(Ts[1]), RfromT(Ts[2]), RfromT(Ts[1])])
    Ms=np.array([MfromT(Ts[0]), MfromT(Ts[1]), MfromT(Ts[2]), MfromT(Ts[3])])
    Mult=np.where((Ts==9000.0) | (colours0==0.0), 0.0,1).astype(int)
    return Mult*Ts, Specs, Mult*Rs, Mult*Ms

def K2Lookup(EPIC):
    '''Finds the filename and location for any K2 EPIC (ID)'''
    EPIC=str(EPIC).split('.')[0]
    tab=np.genfromtxt('/home/astro/phrnbe/PyFolder/KTwo/EPIC_FileLocationTable2.txt', dtype=str)
    if EPIC in tab[:, 0]:
        file=tab[np.where(tab[:,0]==EPIC)[0][0]][1]
    else:
        print 'No K2 EPIC in archive.'
        file=''
    return file

'''
