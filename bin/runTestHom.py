"""Compare the fit of the homogeneous and heterogeneous models to each group, independently. 

Parameters estimated:
(infection-related)
- p1hom: parameter of the homogeneous model for group 1
- p1het,a1,b1: parameters of the heterogeneous model for group 1
- p2hom: parameter of the homogeneous model for group 2
- p2het,a2,b2: parameters of the heterogeneous model for group 2
- eps: probability of ineffective challenge
(mortality-related)
- meanI: mean time to death of infected hosts
- sI: shape parameter of the distribution of time to death of infected hosts
- meanU: mean time to death from old-age (i.e. from uninfected hosts)
- sU: shape parameter of the distribution of time to death of old-age
- k: background probability of death, independent of infection or old-age

Assumptions:
- infected flies cannot outlive natural mortality (meanI<meanU)
- prior distributions for parameters governing natural mortality set from those estimated from control survival
"""
from matplotlib import use
use('Agg') # To save figures to disk, comment to have figures as pop-ups
import sys
import pymc as py

# Import libraries
sys.path.append('lib')
import timeTestHom as testHom

# Import Data - see help(testHom.TimeData)
data=testHom.TimeData.fromCSV(dataPath1='./data/Wneg.csv',dataPath2='./data/Wpos.csv',dataName='wolb2012')

# Import model - see help(timeTestHom.Model)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Estimating infection parameters
niterations=5
burnin=0
thinF=1

mod=testHom.Model.setup(data,bRandomIni=False, bOverWrite=True)
M=py.MCMC(mod,db='pickle', dbname=mod.saveTo+'-MCMC.pickle')
M.sample(niterations, burnin, thinF)
M.db.close()

# Check traces
#py.Matplot.plot(M,path=mod.path)

# The following can always be done in a later session using the folder to the results:
# mod= timeTestHom.savedModel(folder)

# Posterior calculations and plots. see mod.calcPosterior documentation for help
# Burnin can be also be set to 0 above, and thinning to 1, and be determined only after analysing the traces
# In such cases, set burnin and thinF parameters in the call below.
mod.calcPosterior()

# The posterior samples of parameter called X (see in priors) can be accessed in mod.Xs
# For example, the posterior samples of p are in mod.ps 
