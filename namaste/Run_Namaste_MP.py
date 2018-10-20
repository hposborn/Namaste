import logging
import pandas as pd
import sys
import glob

logger = logging.getLogger('namaste')
from Namaste_2.namaste import namaste

def Run(epic):
    st=namaste.Star(epic,namaste.Settings(nsteps=30000,nthreads=4,outfilesloc='Outputs/',fitsloc='Lightcurves',verbose=False))
    st.csvfile_dat('Best_Stellar_params_NoGRIZ.csv')
    st.initLD()
    st.addLightcurve(glob.glob('Lightcurves/EPIC'+epic+'_bestlc_*')[0])
    st.addGP()
    st.Optimize_GP()
    params=pd.DataFrame.from_csv('orb_params_sorted.csv')
    #2121.02,0.5,0.0032
    st.AddMonotransit(params.loc[int(epic),'cen'],params.loc[int(epic),'dur'],params.loc[int(epic),'dep'])
    st.SaveAll()
    st.RunMCMC()
    st.SaveAll()
    st.SaveMCMC()
    st.PlotMCMC()

if __name__=='__main__':
    Run(sys.argv[1])
