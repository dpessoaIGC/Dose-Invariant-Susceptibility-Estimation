""" Estimation of infection parameters from day mortality. 

Parameters estimated:
- p: probability that a single virion will cause infection in a host of the first group
- a,b: shape parameters for the distribution of susceptibility of the second group compared to the first
- eps: probability of ineffective challenge
""" 
from matplotlib import use
use('Agg') # To save plots to disk, comment out if you want pop-up windows
import sys
import pymc as py

# Import libraries
sys.path.append('lib')
import dayEst

# Import Data - see DayData documentation for more information: help(dayEst.DayData)
data=dayEst.DayData.fromCSV(dataPath='./data/wolb2012_day30.csv',dataName='wolb2012')

# Initialize model - see Model documentation for more information: help(dayEst.Model) 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Running MCMC
niterations=5
burnin=0
thinF=1

mod=dayEst.Model.setup(data=data, bRandomIni=False, bOverWrite=True)
M=py.MCMC(mod,db='pickle', dbname=mod.saveTo+'-MCMC.pickle')
M.sample(niterations, burnin, thinF)
M.db.close()

# Check traces
#py.Matplot.plot(M,path=mod.path)

# The following can always be done in a later session using the folder to the results:
# mod= dayEst.savedModel(folder)

# Posterior calculations and plots. see mod.calcPosterior documentation for help
# Burnin can be also be set to 0 above, and thinning to 1, and be determined only after analysing the traces
# In such cases, set burnin and thinF parameters in the call below.
mod.calcPosterior()

# The posterior samples of parameter called X (see in priors) can be accessed in mod.Xs
# For example, the posterior samples of p are in mod.ps 
