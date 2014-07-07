""" Estimation of mortality parameters from survival of control hosts over time. 

Parameters estimated:
* meanU - mean time to death from old-age (i.e. from uninfected hosts)
* sU - shape parameter of the distribution of time to death of old-age
* k - background probability of death, independent of infection or old-age

Created on 06/07/2014 by Delphine Pessoa.
"""

# Import libraries
from copy import deepcopy
import pickle, pymc as py, numpy as np, scipy as sc, scipy.special as sp, scipy.stats as st, pylab as pl, sys, os, importlib, shutil
import utils as ut

def initializeModel(data, resultsName='',savePath=None, bOverWrite=True,priorsFile=None, figFormat='png'):
    """ Estimation of mortality parameters from control survival over time. 

Parameters estimated:
* meanU - mean time to death from old-age (i.e. from uninfected hosts)
* sU - shape parameter of the distribution of time to death of old-age
* k - background probability of death, independent of infection or old-age

Input:
data - Dobject from Data class, see ./lib/utils
resultsName - a short name that identifies this MCMC estimation. If None, a name will be created that concatenates data.dataName and the name of the model, in this case, 'control'.
savePath - folder to save all results, if None, will create a folder in ./results called resultsName
bOverWrite - boolean. If a folder already exists, should it be overwritten (True, default) or should a subfolder be created (False)?
priorsFile - name of python file in ./lib/priors containing the definition of the prior distributions of the parameters.

Possible plots (only after MCMC has been run):
- plotSurvival
"""    
    
    name=data.dataName+'_control'
    
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
    data.pickle(path+'data.pickle')
    
    #~~ Priors ~~
    if priorsFile==None:
        priorsFile='priors_timeControlEst'
    #Copy priors files to results folder
    priors=importlib.import_module('lib.priors.'+priorsFile)
    shutil.copyfile('./lib/priors/'+priorsFile+'.py', path+'prior.py')
    
    return Model(data, priors, name, path)

def reduceData(data):
    alldata=data.copy()
    d=data.copy()
    alldata.nhosts1=data.nhosts1.astype(float)
    alldata.nhosts2=data.nhosts2.astype(float)
    d.nhosts1=data.nhosts1.astype(float)[data.doses==0]
    d.nhosts2=data.nhosts2.astype(float)[data.doses==0]
    d.timesDeath1=np.array(data.timesDeath1)[data.doses==0][0]
    d.timesDeath2=np.array(data.timesDeath2)[data.doses==0][0]
    d.survivors1=data.survivors1[data.doses==0]
    d.survivors2=data.survivors2[data.doses==0]
    d.doses=data.doses[data.doses==0]
    d.ndoses=len(d.doses)
    return d,alldata

def savedModel(path):
    path+=('/' if path[-1]!='/' else '')
    data=ut.DataFromPickle(path+'data.pickle')
    sys.path.append(path)
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
    """ Estimation of mortality parameters from control survival over time. 

Parameters estimated:
* meanU - mean time to death from old-age (i.e. from uninfected hosts)
* sU - shape parameter of the distribution of time to death of old-age
* k - background probability of death, independent of infection or old-age
"""
    #~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ Setting up the MCMC ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~#    
    def __init__(self,data, priors, name, path):
        m=self
        #Reduce data to control data only
        d,alldata=reduceData(data)
        self.d=d
        self.alldata=alldata
        self.name=name
        self.saveTo=path+name
        self.path=path
        
        # Reducing times to those where a change occurs at least once
        # (compute the probabilities only at the times there was change)
        
        (chgT, iTD1, iTD2) = ut.changingTimes(d.timesDeath1, d.timesDeath2)
        
        #~~ Priors ~~
        m.parameters=priors.parameters
        for key in m.parameters:
            setattr(m,key,getattr(priors,key))
        m.tauU=py.Lambda('tauU',lambda mean=m.meanU, s=m.sU: mean/s)
        
        #~~ Likelihood ~~
        
        # Calculate the probabilities of deaths at each of the changing times
        m.probdU=py.Lambda('probdU',lambda s=m.sU, tau=m.tauU, k=m.k,t1=d.times[chgT-1],t2=d.times[chgT]: ut.kpdfInt(t1,t2,s,tau,k), trace=False)
        
        # Calculate the probabilities of survival at each of the changing times
        m.probsU=py.Lambda('probsU',lambda s=m.sU, tau=m.tauU, k=m.k: 1-(ut.kpdfInt(0,d.tmax,s,tau,k)), trace=False)
        
        def likelihood_deaths(value,probdU):
            res=probdU[value]
            inf0=res<0
            if any(inf0): 
                res[res<0]=0
            return np.log(res).sum()
        
        def likelihood_survivors(value,probsU):
            res=(probsU)**value
            inf0=res<0
            if inf0: 
                res=0
            return np.log(res)
        
        # Calculate the likelihoods
        m.liks=[]
        LD1=py.Stochastic(logp=likelihood_deaths,doc='',name='LD1',parents={'probdU':m.probdU}, trace=False, observed=True, dtype=int, value=iTD1)
        m.liks+=['LD1']
        LD2=py.Stochastic(logp=likelihood_deaths,doc='',name='LD2',parents={'probdU':m.probdU}, trace=False, observed=True, dtype=int, value=iTD2)
        m.liks+=['LD2']
        if d.survivors1>0:
            LS1=py.Stochastic(logp=likelihood_survivors,doc='',name='LS1',parents={'probsU':m.probsU}, trace=False, observed=True, dtype=int, value=d.survivors1)
            m.liks+=['LS1']
        if d.survivors2>0:
            LS2=py.Stochastic(logp=likelihood_survivors,doc='',name='LS2',parents={'probsU':m.probsU}, trace=False, observed=True, dtype=int, value=d.survivors2)
            m.liks+=['LS2']
        
        # The following are the variables needed for plots
        m.vals=('ts','cdf1_ci','cdf2_ci')
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
burnin - how many iterations from the begining should be discarded
thinF - thining factor.
bOverWrite - boolean. If these calculations have already been done in the given results folder, should they be calculated again (True) or not (False, default)?
figFormat - format in which figures should be saved. Examples: 'png' (default),'tiff','pdf','jpg'
"""     
        self.figFormat=figFormat
        if bOverWrite:
            try:
                M2=pickle.load(open(self.saveTo+'-MCMC.pickle'))
                self.__calc__(M2,burnin,thinF)
            except IOError:
                print "Please run MCMC first!"
                return
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
                try:
                    M2=pickle.load(open(self.saveTo+'-MCMC.pickle'))
                    self.__calc__(M2,burnin,thinF)
                except:
                    print "Please run MCMC first!"
                    return
    
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
        tauUs=meanUs/sUs
        setattr(self,'tauUs',tauUs)
        d=self.d
        
        ts=np.arange(0,d.times[-1]+1)
        cdf1_ci=np.zeros([3,len(ts)]) # Interval for the probability of death per dose at each day
        cdf2_ci=np.zeros([3,len(ts)])
        
        kprobcdf=np.zeros((len(ts),len(sUs)))
        kprobcdf[0,:]=0.   
        for ti in range(1,len(ts)):
            kprobcdf[ti,:]=kprobcdf[ti-1,:]+ut.kpdfInt(ts[ti-1],ts[ti],sUs,tauUs,ks)
        
        for ti in range(len(ts)):
            md_negi=1-kprobcdf[ti,:]
            md_posi=1-kprobcdf[ti,:]
            cdf1_ci[:,ti]=ut.confint(md_negi)
            cdf2_ci[:,ti]=ut.confint(md_posi)
            progBar.iter(1./len(ts))
        
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
        self.plotSurvival()
        print "\nPosterior fitted to normal distributions (can be used as priors for timeEst):"
        post=self.normalPosterior()
        print post
        f=open(self.saveTo+'-posteriors.py','w')
        f.write(post)
        f.close()
    
    def plotSurvival(self):
        """Plots posterior distribution for the times to death estimated from control mortality.
        
        Equivalent figure in article: Figure 1.
        
        Returns:
        f - Figure
        ax1,ax2 - Axes from panel A and B, respectively
        ax3 - Axes that only has the legend
        """ 
        if not hasattr(self,'sUs'):
            print "Please run calcPosterior first!"
            return
        else:        
            return ut.plotControlSurvival(self)
    
    def normalPosterior(self):
        return """k=py.Normal('k',mu=%e,tau=1/(%e)**2)
meanU=py.Normal('meanU',mu=%e,tau=1/(%e)**2)
sU=py.Normal('sU',mu=%e,tau=1/(%e)**2)
"""%(np.mean(self.ks),np.std(self.ks),np.mean(self.meanUs),np.std(self.meanUs),np.mean(self.sUs),np.std(self.sUs))