""" Estimation of infection parameters from day mortality. 

Parameters estimated:
* p - probability that a single virion will cause infection in a host of the first group
* a,b - shape parameters for the distribution of susceptibility of the second group compared to the first
* eps - probability of ineffective challenge

Created on the 06/07/2014 by Delphine Pessoa.
"""

# Import libraries
from copy import deepcopy
import pickle, pymc as py, numpy as np, scipy as sc, scipy.special as sp, scipy.stats as st, pylab as pl, sys, os, importlib, shutil
import utils as ut

def initializeModel(dataFile, dataPath='./data',resultsName='',savePath=None, bOverWrite=True,priorsFile=None, figFormat='png'):
    """ Estimation of infection parameters from day mortality. 

Parameters estimated:
* p - probability that a single virion will cause infection in a host of the first group
* a,b - shape parameters for the distribution of susceptibility of the second group compared to the first
* eps - probability of ineffective challenge

Input:
data - path to '.py' file containing dose-response data (see format in './data/wolb2012_day30.py')
resultsName - a short name that identifies this MCMC estimation. If None, a name will be created that concatenates data.dataName and the name of the model, in this case, 'timeEst'.
savePath - folder to save all results, if None, will create a folder in ./results called resultsName
bOverWrite - boolean. If a folder already exists, should it be overwritten (True, default) or should a subfolder be created (False)?
priorsFile - name of python file in ./lib/priors containing the definition of the prior distributions of the parameters.

Possible plots (only after MCMC has been run):
- plotSurvival
"""    
    sys.path.append(dataPath)
    data=importlib.import_module(dataFile)
    name=data.dataName+'_dayEst'
    
    priorsFile=priorsFile
    fi=1
    if savePath==None:
        savePath='./results/'
    path=savePath+('/' if savePath[-1]!='/' else '')+name+'/'
    if not bOverWrite:
        print "changing folder"
        if os.path.exists(m.path):
            while os.path.exists(m.path+'/'+'(%i)'%fi):
                fi+=1
            path+='/(%i)'%fi
        path+='/'
        
        os.makedirs(path)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    print "Results will be saved in "+path
    shutil.copyfile(dataPath+dataFile+'.py', path+'dataD.py')
    
    #~~ Priors ~~
    if priorsFile==None:
        priorsFile='priors_dayEst'
    #Copy priors files to results folder
    priors=importlib.import_module('lib.priors.'+priorsFile)
    shutil.copyfile('./lib/priors/'+priorsFile+'.py', path+'prior.py')
    
    return Model(data, priors, name, path)

def savedModel(path):
    path+=('/' if path[-1]!='/' else '')
    sys.path.append(path)
    data=importlib.import_module('dataD')
    priors=importlib.import_module('prior')
    model=pickle.load(open(path+'model.pickle'))
    mod=Model(data,priors,model['name'], path)
    try:
        traces=pickle.load(open(path+mod.name+'.pickle'))
        for v in mod.parameters:
            setattr(self,v+'s', saved[v+'s'])
    except IOError:
        pass
    return mod

class Model(object):
    """ Estimation of infection parameters from day mortality.  

Parameters estimated:
* p - probability that a single virion will cause infection in a host of the first group
* a,b - shape parameters for the distribution of susceptibility of the second group compared to the first
* eps - probability of ineffective challenge
"""
    #~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ Setting up the MCMC ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~#    
    def __init__(self,data, priors, name, path):
        m=self
        d=data
        self.d=d
        self.name=name
        self.saveTo=path+name
        self.path=path
        
        #~~ Priors ~~
        m.parameters=priors.parameters
        for key in m.parameters:
            setattr(m,key,getattr(priors,key))
        
        #~~ Likelihood ~~
        pi1=py.Lambda('pi1',lambda p=m.p,eps=m.eps: ut.pi_hom(d.doses[d.doses>0],p,eps))
        pi2=py.Lambda('pi2',lambda p=m.p,a=m.a2,b=m.b2,eps=m.eps: ut.pi_het(d.doses[d.doses>0],p,a,b,eps))
        
        L1=py.Binomial('L1',n=d.nhosts1[d.doses>0].tolist(),p=pi1,value=d.response1[d.doses>0],observed=True)
        L2=py.Binomial('L2',n=d.nhosts2[d.doses>0].tolist(),p=pi2,value=d.response2[d.doses>0],observed=True)
        
        
        # The following are the variables needed for plots
        m.vals=('x2','pi1_ci','pi2_ci')
        for v in m.vals:
            setattr(m,v,None)
        self.pickle()
    
    def pickle(self):
        save={'path':self.path,'saveTo':self.saveTo, 'name':self.name}
        pickle.dump(save,open(self.path+'model.pickle','w'))    
        
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ Calculating posterior predictive distributions ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def calcPosterior(self,burnin=0,thinF=1, bOverWrite=False,figFormat='png'):
        """Calculates posterior distributions needed for creating figures.

Input:
M2 - dictionnary from a MCMC loaded from a pickle
burnin - how many iterations from the begining should be discarded
thinF - thining factor.
bOverWrite - boolean. If these calculations have already been done in the given results folder, should they be calculated again (True) or not (False, default)?
figFormat - format in which figures should be saved. Examples: 'png' (default),'tiff','pdf','jpg'
"""        
        self.figFormat=figFormat
        if bOverWrite:
            M2=pickle.load(open(self.saveTo+'-MCMC.pickle'))
            self.__calc__(M2,burnin,thinF)
        else:
            try:
                saved=pickle.load(open(self.saveTo+'-postcalc.pickle'))
                if (saved['burnin']==burnin) & (saved['thinF']==thinF):
                    print "Imported previous calculations"
                    for v in self.vals:
                        setattr(self,v,saved[v])
                    for v in self.parameters:
                        setattr(self,v+'s', saved[v+'s'])
                    self.__plot__()
            except IOError:
                M2=pickle.load(open(self.saveTo+'-MCMC.pickle'))
                self.__calc__(M2,burnin,thinF)
    
    def __calc__(self,M2,burnin,thinF):
        progBar=ut.ProgressBar("Calculating")
        # Calculations for the plots of the posterior fittings
        for p in self.parameters:
            vals=M2[p][0][burnin:None:thinF].tolist()
            
            nchains=len(M2[p].keys())
            if nchains>1:
                for c in nchains[1:]:
                    vals.extend(M2[p][c][burnin:None:thinF])
            
            setattr(self,p+'s',np.array(vals))
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
        
        res={'burnin':burnin,'thinF':thinF}
        for v in self.vals:
            setattr(self,v,eval(v))
            res[v]=eval(v)
        for v in self.parameters:
            res[v+'s']=eval(v+'s')
        pickle.dump(res,open(self.saveTo+'-postcalc.pickle','w'))
        ut.write_vals(self)
        self.__plot__()
    
    def __plot__(self):
        print "Results will be saved in "+self.path
        self.plotPosterior()
    
    def plotBeta(self):
        """Plots the estimated beta distribution with confidence interval. (Same as panel B of Figure 5 of the article). """
        if not hasattr(self,'ps'):
            print "Please run calcPosterior first!"
            return
        else:
            return ut.plotBeta(self)
    
    def plotDoseResponse(self): 
        """Plots the estimated dose-response function with confidence intervals. (Same as panel A of Figure 5 in the article)"""
        if not hasattr(self,'ps'):
            print "Please run calcPosterior first!"
            return
        else:
            return ut.plotDoseResponse(self)
    
    def plotPosterior(self):
        """Plots the posterior intervals for the dose-response curve, the beta distribution of the second group and the correlation between the parameters a and b from the beta distribution.

Equivalent figure in article: Figure 3.

Returns:
- f: Figure
- ax1, ax2, ax3: Axes from each of the panels"""
        if not hasattr(self,'ps'):
            print "Please run calcPosterior first!"
            return
        else:
            return ut.plotPosterior(self)