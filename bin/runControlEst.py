""" Estimation of mortality parameters from survival of control hosts over time. 

Parameters estimated:
- meanU: mean time to death from old-age (i.e. from uninfected hosts)
- sU: shape parameter of the distribution of time to death of old-age
- k: background probability of death, independent of infection or old-age
"""
from matplotlib import use
use('Agg') # To save figures to disk, comment to have figures as pop-ups
import sys
import pymc as py
# Import libraries
sys.path.append('lib')
import timeControlEst as controlEst

# Import Data - see help(controlEst.TimeData)
data=controlEst.TimeData.fromCSV('./data/Wneg.csv','./data/Wpos.csv','wolb2012')

# Import model - see help(controlEst.Model)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Estimating control survival
niterations=5
burnin=0
thinF=1

mod=controlEst.Model.setup(data,bRandomIni=False, bOverWrite=True)
M=py.MCMC(mod,db='pickle', dbname=mod.saveTo+'-MCMC.pickle')
M.sample(niterations, burnin, thinF)
M.db.close()

# Check traces
#py.Matplot.plot(M,path=mod.path)

# The following can always be done in a later session using the folder to the results:
# mod= timeControlEst.savedModel(folder)

# Posterior calculations and plots. see mod.calcPosterior documentation for help
# Burnin can be also be set to 0 above, and thinning to 1, and be determined only after analysing the traces
# In such cases, set burnin and thinF parameters in the call below.
mod.calcPosterior()
mod.plotSurvival(grouplabels=[r'Wolb$^-$',r'Wolb$^+$']) # Put different group labels on survival plot

# You can then use the Normal distributions (see file named '-posteriors.py'), fitted to the posterior samples, as priors for estimating all parameters from other curves (not control, see runTimeEst.py)

# The posterior samples of parameter called X (see in priors) can be accessed in mod.Xs
# For example, the posterior samples of k are in mod.ks 
