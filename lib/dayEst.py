""" Estimation of infection parameters from day mortality. 

Parameters estimated:
- p: probability that a single virion will cause infection in a host of the first group
- a,b: shape parameters for the distribution of susceptibility of the second group compared to the first
- eps: probability of ineffective challenge

"""

# Import libraries
from copy import deepcopy
import pickle, pymc as py, numpy as np, scipy as sc, scipy.special as sp, scipy.stats as st, pylab as pl, sys, os, importlib, shutil
import utils as ut
import modelFunctions as mf
import dataFunctions as df

class Model(mf.DoseResponseModels):
    """ Estimation of infection parameters from day mortality.  

Initialize from scratch with Model.setup(), and from saved model with Model.savedModel(). 

Parameters estimated:
- p: probability that a single virion will cause infection in a host of the first group
- a,b: shape parameters for the distribution of susceptibility of the second group compared to the first
- eps: probability of ineffective challenge

Possible plots:
- plotPosterior
- plotDoseResponse
- plotBeta
"""
    __defaultPrior__='priors_dayEst'
    __defaultName__='_dayEst'
    #~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ Setting up the MCMC ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~#
    @ut.doc_inherit
    def __init__(self,data, priors, name, path, bRandomIni):
        m=self
        d=data
        self.d=d
        
        # The following are the variables needed for plots
        m.vals=('x2','pi1_ci','pi2_ci')
        super(Model,self).__init__(data,priors,name,path,bRandomIni)
    
    @ut.doc_inherit
    def likelihood_setup(self,bRandomIni):
        m=self
        d=m.d 
        if not hasattr(m,'pi1'):
            #~~ Likelihood ~~
            m.pi1=py.Lambda('pi1',lambda p=m.p,eps=m.eps: ut.pi_hom(d.doses[d.doses>0],p,eps))
            m.pi2=py.Lambda('pi2',lambda p=m.p,a=m.a2,b=m.b2,eps=m.eps: ut.pi_het(d.doses[d.doses>0],p,a,b,eps))
        super(Model,self).likelihood_setup(bRandomIni)
    
    def __lik_setup__(self):
        m=self
        d=m.d
        zeroprob=0
        try:
            m.L1=py.Binomial('L1',n=d.nhosts1[d.doses>0].tolist(),p=m.pi1,value=d.response1[d.doses>0],observed=True)
            m.L2=py.Binomial('L2',n=d.nhosts2[d.doses>0].tolist(),p=m.pi2,value=d.response2[d.doses>0],observed=True)
            m.liks=['L1','L2']
            sum([getattr(m,l).logp for l in m.liks])
        except py.ZeroProbability, e:
            zeroprob=1
        
        return zeroprob
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ Calculating posterior predictive distributions ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def __calc__(self):
        progBar=ut.ProgressBar("Calculating")
        # Calculations for the plots of the posterior fittings
        for p in self.parameters:
            exec('%ss=self.%ss'%(p,p))
        
        d=self.d
        
        x2=10**np.arange(np.log10(d.doses[d.doses>0][0])-1,np.log10(d.doses[-1])+1,0.1)
        pi1_ci=np.zeros([3,len(x2)])
        pi2_ci=np.zeros([3,len(x2)])
        nx=0
        for i in range(len(x2)):
            pi1_ci[:,i]=ut.confint(ut.pi_hom(x2[i],ps,epss))
            pi2_ci[:,i]=ut.confint(ut.pi_het(x2[i],ps ,a2s,b2s,epss))
            progBar.iter(1./len(x2))
        
        progBar.finish()
        
        res={'burnin':self.burnin,'thinF':self.thinF}
        for v in self.vals:
            setattr(self,v,eval(v))
            res[v]=eval(v)
        for v in self.parameters:
            res[v+'s']=eval(v+'s')
        pickle.dump(res,open(self.saveTo+'-postcalc.pickle','w'))
        
        self.__plot__()
    
    def __plot__(self):
        print "Results will be saved in "+self.path
        self.write_vals()
        self.plotPosterior()


DayData=df.DayData
