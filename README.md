#Namaste: An Mcmc Analysis of Single Transiting Exoplanets

Namaste is a Bayesian transit-fitting programme to predict the size, orbit, etc of exoplanets from just a single transit. 



### Usage ###
To bring up the argument options type

	python Namaste.py --help

Which displays:

	usage: Namaste.py [-h] [-n NSTEP] [-g] [-tc TCEN] [-td TDUR] [-d DEPTH]
			  [-ts TS] [-rs RS] [-ms MS] [-q]
			  KIC lcurve

	positional arguments:
	  KIC                   Kepler (KIC) or K2 (EPIC) id
	  lcurve                File location of lightcurve file. Type "get" to
				download lightcurve (Kepler only).

	optional arguments:
	  -h, --help            show this help message and exit
	  -n NSTEP, --nstep NSTEP
				Number of MCMC steps. Default 15000
	  -g, --gproc           Use to turn on Gaussian Processes
	  -tc TCEN, --tcen TCEN
				Initial central transit time estimate
	  -td TDUR, --tdur TDUR
				Initial transit length/duration estimate
	  -d DEPTH, --dep DEPTH
				Initial transit depth estimate
	  -ts TS, --stemp TS    Stellar temperature and error (eg 5500,150). Can be
				estimated from colours
	  -rs RS, --srad RS     Stellar radius and error (eg 1.1,0.1). Can be
				estimated from colours
	  -ms MS, --smass MS    Stellar mass and error (eg 1.2,0.2). Can be estimated
				from colours
	  -q, --quiet           don't print status messages to stdout

Example lightcurves for KIC 3239945 are provided and can be run with the line:
	python Namaste.py 3239945 get

### Pre-Requisites###
Standard python packages: glob, numpy, scipy, matplotlib, csv,
Non-standard: [george](https://github.com/dfm/george) (version 1.0 or higher), [emcee](https://github.com/dfm/emcee), [corner](https://github.com/dfm/corner)


###Acknowledgements###
The python modules emcee and george form the engine of this script, and all credit to Dan Foreman-Mackey for that
Transit-fitting code courtesy of Ian Crossfield.

###Attribution ###
If you make use of this code please cite:

"Single Transit Candidates from K2: Detection and Period Estimation" by Osborn et al, 2016 (http://mnras.oxfordjournals.org/content/457/3/2273.full.pdf or http://arxiv.org/abs/1512.03722)
