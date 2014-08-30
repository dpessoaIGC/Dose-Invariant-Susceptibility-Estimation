""" Estimation of infection and mortality parameters from survival over time. 

Parameters estimated:
(infection-related)
- p: probability that a single virion will cause infection in a host of the first group
- a,b: shape parameters for the distribution of susceptibility of the second group compared to the first
- eps: probability of ineffective challenge
(mortality-related)
- meanI: mean time to death of infected hosts
- sI: shape parameter of the distribution of time to death of infected hosts
- meanU: mean time to death from old-age (i.e. from uninfected hosts)
- sU: shape parameter of the distribution of time to death of old-age
- k: background probability of death, independent of infection or old-age
(extra)
- Ig1dX, Ig2dX: estimated number of infected hosts from group 1 (or 2) when challenged with dose number X

Assumptions:
- infected flies cannot outlive natural mortality (meanI<meanU)
- prior distributions for parameters governing natural mortality set from those estimated from control survival
"""
from matplotlib import use
use('Agg') # To save figures to disk, comment to have figures as pop-ups
import sys
import pickle
import pymc as py
        
# Import libraries
sys.path.append('lib')
import timeEst

# Import Data - see TimeData documentation for more information: help(timeEst.TimeData)
data=timeEst.TimeData.fromCSV(dataPath1='./data/Wneg.csv',dataPath2='./data/Wpos.csv',dataName='wolb2012')

# Initialize model - see Model documentation for more information: help(timeEst.Model)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Estimating infection parameters
niterations=5
burnin=0
thinF=1

mod=timeEst.Model.setup(data=data,bRandomIni=False, bOverWrite=True)
M=py.MCMC(mod,db='pickle', dbname=mod.saveTo+'-MCMC.pickle')
M.sample(niterations, burnin, thinF)
M.db.close()

# Check traces
#py.Matplot.plot(M,path=mod.path)

# The following can always be done in a later session using the folder to the results:
# mod= timeEst.savedModel(folder)

# Posterior calculations and plots. see mod.calcPosterior documentation for help
# Burnin can be also be set to 0 above, and thinning to 1, and be determined only after analysing the traces
# In such cases, set burnin and thinF parameters in the call below.
mod.calcPosterior()

# The posterior samples of parameter called X (see in priors) can be accessed in mod.Xs
# For example, the posterior samples of p are in mod.ps 
