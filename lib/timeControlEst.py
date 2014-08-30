""" Estimation of mortality parameters from survival of control hosts over time. 

Parameters estimated:
- meanU: mean time to death from old-age (i.e. from uninfected hosts)
- sU: shape parameter of the distribution of time to death of old-age
- k: background probability of death, independent of infection or old-age
"""

# Import libraries
from copy import deepcopy
import pickle, pymc as py, numpy as np, scipy as sc, scipy.special as sp, scipy.stats as st, pylab as pl, sys, os, importlib, shutil
from matplotlib import rcParams
import dataFunctions as df
import modelFunctions as mf
import utils as ut

class Model(mf.TimeModels):
    """ Estimation of mortality parameters from control survival over time. 
    Initialize with Model.setup(). To restore a saved model, use 
    Model.savedModel().

Parameters estimated:
- meanU: mean time to death from old-age (i.e. from uninfected hosts)
- sU: shape parameter of the distribution of time to death of old-age
- k: background probability of death, independent of infection or old-age

Possible plots:
- plotSurvival
"""
    __defaultPrior__='priors_timeControlEst'
    __defaultName__='_control'
     
    def __init__(self,data, priors, name, path,bRandomIni):
#        """Returns a Model object, used to launch MCMC and process posterior distributions.
#
#        Input:
#        - data (df.Data)
#        - priors (dict) - a dictionnary with a PYMC object for each parameter
#        - name (str) - descriptor for the MCMC results
#        - path (str) - path to folder where results should be saved
#        """
        #Reduce data to control data only
        data.reduce(data.doses==0)
        
        # The following are the variables needed for plots
        self.vals=('ts','cdf1_ci','cdf2_ci')
        super(Model,self).__init__(data,priors,name,path,bRandomIni)
    
    def __lik_setup__(self):
        m=self
        d=m.d
        chgT=m.chgT
        iTd1=m.iTd1
        iTd2=m.iTd2
        zeroprob=0
        try:
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
            m.LD1=py.Stochastic(logp=likelihood_deaths,doc='',name='LD1',parents={'probdU':m.probdU}, trace=False, observed=True, dtype=int, value=iTd1)
            m.liks+=['LD1']
            m.LD2=py.Stochastic(logp=likelihood_deaths,doc='',name='LD2',parents={'probdU':m.probdU}, trace=False, observed=True, dtype=int, value=iTd2)
            m.liks+=['LD2']
            if bool(d.survivors1)>0:
                m.LS1=py.Stochastic(logp=likelihood_survivors,doc='',name='LS1',parents={'probsU':m.probsU}, trace=False, observed=True, dtype=int, value=d.survivors1)
                m.liks+=['LS1']
            if bool(d.survivors2)>0:
                m.LS2=py.Stochastic(logp=likelihood_survivors,doc='',name='LS2',parents={'probsU':m.probsU}, trace=False, observed=True, dtype=int, value=d.survivors2)
                m.liks+=['LS2']
        
            sum([getattr(m,l).logp for l in m.liks])
            
        except py.ZeroProbability:
            zeroprob=1
        return zeroprob
    
    # Calculate posterior predictive distributions and plot figures.
    def __calc__(self):
        progBar=ut.ProgressBar("Calculating")
        # Calculations for the plots of the posterior fittings
        for p in self.parameters:
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
        
        res={'burnin':self.burnin,'thinF':self.thinF}
        for v in self.vals:
            setattr(self,v,eval(v))
            res[v]=eval(v)
        for v in self.parameters:
            res[v+'s']=eval(v+'s')
        pickle.dump(res,open(self.saveTo+'-postcalc.pickle','w'))
        
        self.__plot__()
    
    # Plot figures.
    def __plot__(self):
        print "Results will be saved in "+self.path
        self.write_vals()
        self.plotSurvival()
        print "\nPosterior samples fitted to normal distributions (also saved in posteriors.py, can be used as priors for timeEst):"
        post=self.normalPosterior()
        print post
        f=open(self.saveTo+'-posteriors.py','w')
        f.write(post)
        f.close()
    
    def plotSurvival(self, grouplabels=[]):
        """Plots posterior distributions for the time to death estimated from control mortality.

Input:
- grouplabels (list of str, optional): labels for each of the two groups

Equivalent figure in article: Figure 1.

Returns:
f (Figure)
ax1,ax2 (Axes) - Axes from panel A and B, respectively
ax3 (Axes) - Axes that only has the legend
"""
        m=self
        if not hasattr(self,'sUs'):
            print "Please run calcPosterior first!"
            return
        else:        
            ts=m.ts
            cdf1_ci=m.cdf1_ci
            cdf2_ci=m.cdf2_ci
            d=m.d.alldata
            prevfsize=rcParams['font.size']
            prevlsize=rcParams['axes.labelsize']
            rcParams.update({'font.size': 10})
            rcParams['axes.labelsize']='large'
            f=pl.figure(figsize=(6,1.875))
            f.subplots_adjust(wspace=0.75,left=0.05, right=1.1)
            ax1=pl.subplot2grid((1,5),(0,0),colspan=2)
            ax2=pl.subplot2grid((1,5),(0,2),colspan=2)
            ax3=pl.subplot2grid((1,5),(0,4))
            
            
            di=0
            l=ax1.plot(ts,cdf1_ci[1,:],'-',lw=0.7,color=pl.cm.RdYlBu_r(0))#Posterior mean
            l=ax2.plot(ts,cdf2_ci[1,:],'-',lw=0.7,color=pl.cm.RdYlBu_r(0))
            l=ax1.fill_between(ts,cdf1_ci[0,:],cdf1_ci[2,:],facecolor=pl.cm.RdYlBu_r(0),lw=0,alpha=0.12)
            l=ax2.fill_between(ts,cdf2_ci[0,:],cdf2_ci[2,:],facecolor=pl.cm.RdYlBu_r(0),lw=0,alpha=0.12)
            
            pushblue=2
            maxred=250
            n=8
            n+=2
            cols_prev=np.array((np.arange(n)+pushblue)*maxred/(n-1.+pushblue),int)
            cols_prev=cols_prev[(cols_prev<100)|(cols_prev>150)]
            cols=[pl.cm.RdYlBu_r(0)]+pl.cm.RdYlBu_r(cols_prev).tolist()
            marks=['o']*8
            
            
            for di,dose in enumerate(d.doses):
                cneg=cols[di]
                cpos=cols[di]
                l=ax1.plot(d.times,np.array([(d.timesDeath1[di]>ti).sum()+d.survivors1[di] for ti in range(len(d.times))])/d.nhosts1[di],'-'+marks[di],color=cneg,mec=cneg,mew=1.5,alpha=1,ms=1)
                l=ax2.plot(d.times,np.array([(d.timesDeath2[di]>ti).sum()+d.survivors2[di] for ti in range(len(d.times))])/d.nhosts2[di],'-'+marks[di],color=cpos,mec=cpos,mew=1.5,alpha=1,ms=1)
                l=ax3.plot(-1,-1,'-'+marks[di],color=cpos,mec=cpos,mew=1.5,alpha=1,ms=1,label='control' if dose==0 else r'10$^{%i}$ TCID$_{50}$'%int(np.log10(dose)))
                l=ax1.set_ylim([-0.03,1.03])
                l=ax2.set_ylim([-0.03,1.03])
            
            if grouplabels==[]:
                grouplabels=['group 1', 'group 2']
            
            ax3.set_xlim(0,1)
            ax3.set_ylim(0,1)
            ax3.axison=False
            ax3.legend(frameon=False,numpoints=1, loc='center right',prop={'size':10})
            labels=['A','B']
            for axi,ax in enumerate([ax1,ax2]):
                ax.set_ylabel('survival')
                ax.set_xlabel('days post challenge')
                ax.text(-0.15, 1.15, labels[axi], transform=ax.transAxes,
                          fontsize=12, fontweight='bold', va='top', ha='right')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()
                l=ax.text(139,1.05,s=grouplabels[axi],ha='right',va='top',fontsize=12)
            f.savefig(m.saveTo+'-posteriorSurvival.'+m.figFormat, bbox_inches='tight',dpi=600)
            rcParams.update({'font.size': prevfsize})
            rcParams['axes.labelsize']=prevlsize    
            print "Plotted posterior survival, see "+m.name+'-posteriorSurvival.'+m.figFormat
            return f,ax1,ax2,ax3
    
    def normalPosterior(self):
        return """k=py.Normal('k',mu=%e,tau=1/(%e)**2)
meanU=py.Normal('meanU',mu=%e,tau=1/(%e)**2)
sU=py.Normal('sU',mu=%e,tau=1/(%e)**2)
"""%(np.mean(self.ks),np.std(self.ks),np.mean(self.meanUs),np.std(self.meanUs),np.mean(self.sUs),np.std(self.sUs))
        

TimeData=df.TimeData
