# Instructions to run a specific model
from matplotlib import use
use('Agg') # To save figures to disk, comment to have figures as pop-ups
import sys
import pickle
import pymc as py
# Import libraries
sys.path.append('lib')
import utils as ut
import timeControlEst as est

# Import Data - see ut.Data.__doc__ for help
dataTime=ut.DataFromCSV('./data/Wneg.csv','./data/Wpos.csv','wolb2012',(r'Wolb$^-$',r'Wolb$^+$'))

# Import model - see Model.__doc__ for help
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Estimating control survival
niterations=5

mod=est.initializeModel(dataTime)
M=py.MCMC(mod,db='pickle', dbname=mod.saveTo+'-MCMC.pickle')
M.sample(niterations)
M.db.close()
# Check traces and determine burnin and thining factor
#py.Matplot.plot(M,path=mod.path)

# The following can always be done in a later session using the folder to the results:
# mod= est.savedModel(folder)
burnin=0
thinF=1
mod.calcPosterior(burnin,thinF)
# You can then use the estimated posterior (see file named '-posteriors.py') as priors for estimating all parameters from other curves (not control)