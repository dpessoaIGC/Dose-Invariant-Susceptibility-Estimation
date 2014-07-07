# Instructions to run a specific model
from matplotlib import use
use('Agg')
import sys
import pickle
import pymc as py
# Import libraries
sys.path.append('lib')
import utils as ut
import dayEst as est

# Set Data - see format in './data/wolb2012_day30.py'
dataPath='./data/'
dataFile='wolb2012_day30'

# Initialize model - see Model documentation for help
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Estimating infection parameters
niterations=5

mod=est.initializeModel(dataFile,dataPath)
M=py.MCMC(mod,db='pickle', dbname=mod.saveTo+'-MCMC.pickle')
M.sample(niterations)
M.db.close()
# Check traces and determine burnin and thining factor
#py.Matplot.plot(M,path=mod.path)

# The following can always be done in a later session using the folder to the results:
# mod= est.savedModel(folder)
burnin=0
thinF=1
# Posterior calculations and plots. see mod.calcPosterior documentation for help
mod.calcPosterior(burnin,thinF)