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

# Import libraries
from copy import deepcopy
import pickle, pymc as py, numpy as np, scipy as sc, scipy.special as sp, scipy.stats as st, pylab as pl, sys, os, importlib, shutil
import dataFunctions as df
import modelFunctions as mf
import utils as ut

class Model(mf.TimeModels,mf.DoseResponseModels):
    """ Estimation of infection and mortality parameters from survival over time. 

Initialize from scratch with Model.setup(), and from saved model with Model.savedModel(). 

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

Possible plots (only after MCMC has been run):
- plotSurvival
- plotBeta
- plotDoseResponse
- plotPosterior
- plotBestDays
"""
    __defaultPrior__='priors_timeEst'
    __defaultName__='_timeEst'
    #~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ Setting up the MCMC ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~#
    @ut.doc_inherit
    def __init__(self,data, priors, name, path, bRandomIni):
        m=self
        # The following are the variables needed for plots
        m.vals=('ts','cdf1_ci','cdf2_ci','x2','pi1_ci','pi2_ci','pdfU','cdfU','pdfI1','pdfI2')
        super(Model,self).__init__(data,priors,name,path,bRandomIni)
        
    @ut.doc_inherit
    def likelihood_setup(self,bRandomIni):
        """ Sets up likelihoods. If bRandomIni, will reset all variables to random values."""
        #~~ Saving variable names ~~
        m=self
        d=self.d
        m.parameters.extend(['Ig1d%i'%i for i in range(sum(d.doses==0),len(d.doses))])
        m.parameters.extend(['Ig2d%i'%i for i in range(sum(d.doses==0),len(d.doses))])
        super(Model,self).likelihood_setup(bRandomIni)
    
    def __lik_setup__(self):
        m=self
        d=m.d
        chgT=m.chgT
        iTd1=m.iTd1
        iTd2=m.iTd2
        zeroprob=0
        try:
            #~~ Other stochastic variables needed to calculate the likelihood ~~
            for di in range(0+sum(d.doses==0),len(d.doses)):
                setattr(m,'pi_hom%i'%di, py.Lambda('pi_hom%i'%di,lambda p=m.p,eps=m.eps,idose=di: ut.pi_hom(d.doses[idose],p,eps)))
                setattr(m,'Ig1d%i'%di,py.Binomial('Ig1d%i'%di,n=d.nhosts1[di],p=getattr(m,'pi_hom%i'%di)))
                setattr(m,'pi_het%i'%di, py.Lambda('pi_het%i'%di,lambda p=m.p,a=m.a2,b=m.b2,eps=m.eps,idose=di: ut.pi_het(d.doses[idose],p,a,b,eps)))
                setattr(m,'Ig2d%i'%di,py.Binomial('Ig2d%i'%di,n=d.nhosts2[di],p=getattr(m,'pi_het%i'%di)))
            
            m.tauU=py.Lambda('tauU',lambda mean=m.meanU, s=m.sU: mean/s)
            m.tauI1=py.Lambda('tauI1',lambda mean=m.meanI1, s=m.sI1: mean/s)
            m.tauI2=py.Lambda('tauI2',lambda mean=m.meanI2, s=m.sI2: mean/s)
            
            
            #~~ Likelihood ~~
            
            # Calculate the probabilities of deaths at each of the changing times
            m.probdU=py.Lambda('probdU',lambda s=m.sU, tau=m.tauU, k=m.k,t1=d.times[chgT-1],t2=d.times[chgT]: ut.kpdfInt(t1,t2,s,tau,k), trace=False)
            m.probdI1=py.Lambda('probdI1',lambda s=m.sI1, tau=m.tauI1, k=m.k,t1=d.times[chgT-1],t2=d.times[chgT]: ut.kpdfInt(t1,t2,s,tau,k), trace=False)
            m.probdI2=py.Lambda('probdI2',lambda s=m.sI2, tau=m.tauI2, k=m.k,t1=d.times[chgT-1],t2=d.times[chgT]: ut.kpdfInt(t1,t2,s,tau,k), trace=False)
            
            # Calculate the probabilities of survival at each of the changing times
            m.probsU=py.Lambda('probsU',lambda s=m.sU, tau=m.tauU, k=m.k: 1-(ut.kpdfInt(0,d.tmax,s,tau,k)), trace=False)
            m.probsI1=py.Lambda('probsI1',lambda s=m.sI1, tau=m.tauI1, k=m.k: 1-(ut.kpdfInt(0,d.tmax,s,tau,k)), trace=False)
            m.probsI2=py.Lambda('probsI2',lambda s=m.sI2, tau=m.tauI2, k=m.k: 1-(ut.kpdfInt(0,d.tmax,s,tau,k)), trace=False)
            
            def likelihood_deaths(value,nf,I,probdI,probdU):
                res=(I/nf)*probdI[value]+(1-(I/nf))*probdU[value]
                inf0=res<0
                if any(inf0): 
                    res[res<0]=0
                return np.log(res).sum()
            
            def likelihood_survivors(value,nf,I,probsI,probsU):
                res=((I/nf)*probsI+(1-(I/nf))*probsU)**value
                inf0=res<0
                if inf0: 
                    res=0
                return np.log(res)
            
            # Calculate the likelihoods
            m.liks=[]
            for i in range(0+sum(d.doses==0),len(d.doses)):
                setattr(m,'LD1_d%i'%i,py.Stochastic(logp=likelihood_deaths,doc='',name='LD1_d%i'%i,parents={'nf':d.nhosts1[i], 'I':getattr(m,'Ig1d%i'%i), 'probdI':m.probdI1,'probdU':m.probdU}, trace=False, observed=True, dtype=int, value=iTd1[i]))
                m.liks+=['LD1_d%i'%i]
                
                if d.survivors1[i]>0:
                    setattr(m,'LS1_d%i'%i,py.Stochastic(logp=likelihood_survivors,doc='',name='LS1_d%i'%i,parents={'nf':d.nhosts1[i], 'I':getattr(m,'Ig1d%i'%i), 'probsI':m.probsI1,'probsU':m.probsU}, trace=False, observed=True, dtype=int, value=d.survivors1[i]))
                    m.liks+=['LS1_d%i'%i]
                
                setattr(m,'LD2_d%i'%i,py.Stochastic(logp=likelihood_deaths,doc='',name='LD2_d%i'%i,parents={'nf':d.nhosts2[i], 'I':getattr(m,'Ig2d%i'%i), 'probdI':m.probdI2,'probdU':m.probdU}, trace=False, observed=True, dtype=int, value=iTd2[i]))
                m.liks+=['LD2_d%i'%i]
                
                if d.survivors2[i]>0:
                    setattr(m,'LS2_d%i'%i,py.Stochastic(logp=likelihood_survivors,doc='',name='LS2_d%i'%i,parents={'nf':d.nhosts2[i], 'I':getattr(m,'Ig2d%i'%i), 'probsI':m.probsI2,'probsU':m.probsU}, trace=False, observed=True, dtype=int, value=d.survivors2[i]))
                    m.liks+=['LS2_d%i'%i]
            
            # Set likelihood to 0 if, for the first group, there is higher chance of infected surviving to the end of the study compared to non-infected.
            @py.potential
            def potIdeaths1(sI=m.sI1,tauI=m.tauI1, sU=m.sU,tauU=m.tauU): 
                return 0.0 if st.gamma.cdf(max(d.times),sI,loc=0,scale=tauI)>=st.gamma.cdf(max(d.times),sU,loc=0,scale=tauU) else -np.Inf
            
            @py.potential
            def potIdeaths2(sI=m.sI2,tauI=m.tauI2, sU=m.sU,tauU=m.tauU): 
                return 0.0 if st.gamma.cdf(max(d.times),sI,loc=0,scale=tauI)>=st.gamma.cdf(max(d.times),sU,loc=0,scale=tauU) else -np.Inf        
            
            setattr(m,'potIdeaths1',potIdeaths1)
            setattr(m,'potIdeaths2',potIdeaths2)
        
            sum([getattr(m,l).logp for l in m.liks])+potIdeaths1.logp+potIdeaths2.logp
            
        except py.ZeroProbability:
            zeroprob=1
        return zeroprob
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #~~ Calculating posterior predictive distributions ~~#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    def __calc__(self):
        progBar=ut.ProgressBar("Preparing calculations of posterior probabilities")
        
        d=self.d
        for p in self.parameters:
            exec('%ss=self.%ss'%(p,p))
        
        tauUs=meanUs/sUs
        tauI1s=meanI1s/sI1s
        tauI2s=meanI2s/sI2s
        setattr(self,'tauUs',tauUs)
        setattr(self,'tauI1s',tauI1s)
        setattr(self,'tauI2s',tauI2s)        
        
        ts=np.arange(0,d.times[-1]+1,0.2)
        cdf1_ci=np.zeros([3,d.ndoses,len(ts)]) # Interval for the probability of death per dose at each day
        cdf2_ci=np.zeros([3,d.ndoses,len(ts)])
        
        #PDFs
        pdfI1=eval(ut.callArray('ut.kpdf(',['ts','sI1s','tauI1s','ks'],(len(ts),len(sUs))))
        progBar.iter(0.25)
        
        pdfI2=eval(ut.callArray('ut.kpdf(',['ts','sI2s','tauI2s','ks'],(len(ts),len(sUs))))
        progBar.iter(0.25)        
        
        #CDFs
        cdfU=eval(ut.callArray('ut.kpdfInt(0,',['ts','sUs','tauUs','ks'],(len(ts),len(sUs))))
        
        cdfI1=eval(ut.callArray('ut.kpdfInt(0,',['ts','sI1s','tauI1s','ks'],(len(ts),len(sUs))))
        progBar.iter(0.25)
        
        cdfI2=eval(ut.callArray('ut.kpdfInt(0,',['ts','sI2s','tauI2s','ks'],(len(ts),len(sUs))))
        progBar.iter(0.25)
        progBar.finish()
        
        progBar.start("Calculating mortalities")
        di=0
        
        cdf1_ci[:,0,:]=ut.confint(1-cdfU)
        cdf2_ci[:,0,:]=ut.confint(1-cdfU)
        progBar.iter(1./d.ndoses)
        
        for di in range(1,d.ndoses):
            pi1s=ut.pi_hom(d.doses[di],ps,epss)
            pi2s=ut.pi_het(d.doses[di],ps,a2s,b2s,epss)
            
            cdf1_ci[:,di,:]=ut.confint(1-((np.array([pi1s.tolist()]*len(ts)).T)*cdfI1)-((1-np.array([pi1s.tolist()]*len(ts)).T)*cdfU))
            cdf2_ci[:,di,:]=ut.confint(1-((np.array([pi2s.tolist()]*len(ts)).T)*cdfI2)-((1-np.array([pi2s.tolist()]*len(ts)).T)*cdfU))
            
            progBar.iter(1./d.ndoses)
            
        
        progBar.finish()
        progBar.start("Calculating probabilities of infection")
        
        x2=10**np.arange(np.log10(d.doses[d.doses>0][0])-1,np.log10(d.doses[-1])+1,0.1)
        pi1_ci=np.zeros([3,len(x2)])
        pi2_ci=np.zeros([3,len(x2)])
        nx=0
        for i in range(len(x2)):
            pi1_ci[:,i]=ut.confint(ut.pi_hom(x2[i],ps,epss))
            pi2_ci[:,i]=ut.confint(ut.pi_het(x2[i],ps ,a2s,b2s,epss))
            progBar.iter(1./len(x2))
        
        progBar.finish()
        
        pdfU=ut.confint(cdfU[:,1:]-cdfU[:,:-1]) # Calculate pdf from cdf to avoid nan from high sU, len= len(ts)-1
        cdfU=ut.confint(cdfU)
        pdfI1=ut.confint(pdfI1)
        pdfI2=ut.confint(pdfI2)
        
        res={'burnin':self.burnin,'thinF':self.thinF}
        for v in self.vals:
            setattr(self,v,eval(v))
            res[v]=eval(v)
        for v in self.parameters:
            res[v+'s']=eval(v+'s')
        pickle.dump(res,open(self.saveTo+'-postcalc.pickle','w'))
        
        self.__plot__()
    
    def __plot__(self):
        print "Results saved in "+self.path
        self.write_vals()
        self.plotSurvival()
        #self.plotBeta()
        #self.plotDoseResponse()
        self.plotPosterior()
    
    def calcBestDays(m):
        """Square distance between observed daily mortality and estimated infected numbers. FIGURE 4 in the manuscript (26/01/2014)"""
        d=m.d
        
        pbar=ProgressBar("Calculating best days")
        # FIRST: compare mortality directly to number of infected
        distnegmat=np.zeros((len(d.times),len(m.sUs)))
        distposmat=np.zeros((len(d.times),len(m.sUs)))
        for dayi,day in enumerate(d.times):
            mortdayneg=np.array([(d.timesDeath1[di]<=day).sum() for di in range(len(d.doses))])
            mortdaypos=np.array([(d.timesDeath2[di]<=day).sum() for di in range(len(d.doses))])    
            for iti,it in enumerate(np.arange(len(m.sUs))):
                #DOC: obsneg=mortdayneg[di]
                #DOC: expneg=eval("Ig1d%is[iti]"%di)
                #DOC: distnegmat=((obsneg-expneg)**2).sum()
                distnegmat[dayi,iti]=np.array([((mortdayneg[di]-eval("m.Ig1d%is[iti]"%di))/d.nhosts1[di].astype(float))**2 for di in range(sum(d.doses==0),len(d.doses))]).sum()/sum(d.doses>0)
                
                distposmat[dayi,iti]=np.array([((mortdaypos[di]-eval("m.Ig2d%is[iti]"%di)) /d.nhosts2[di].astype(float))**2 for di in range(sum(d.doses==0),len(d.doses))]).sum()/float(sum(d.doses>0))
                pbar.iter(1./len(d.times))
                
        pbar.finish()
        
        disttogethermat=(distnegmat+distposmat)/(2.)
        m.distneg=confint(1-distnegmat.T**.5)
        m.distpos=confint(1-distposmat.T**.5)    
        m.disttogether=confint(1-disttogethermat.T**.5)
        m.disttogether/=m.disttogether.max() 
        

    def asgood(listy,arr,alpha):
            mini=arr.max()
            #asgoods=arr<=sap(arr,alpha*100)
            asgoods=arr>=(alpha*mini)
            return (listy[arr.argmax()], listy[asgoods][0], listy[asgoods][-1])

    def plotBestDays(m, alpha=0.95):
        """Calculates the best day following the formula shown in manuscript given the chosen alpha (days for which the score is at least (1-alpha)*maximumScore. Plots the score over time.

    Equivalent figure in manuscript: Figure 6.

    Returns:
    - f (Figure)
    - ax1, ax2 (Axes): ax2 corresponds to the axes on the right.
        """
        if not hasattr(m, 'disttogether'):
            calcBestDays(m)
        
        disttogether=m.disttogether
        distneg=m.distneg
        distpos=m.distpos
        pdfU=m.pdfU
        pdfI1=m.pdfI1
        pdfI2=m.pdfI2
        ts=m.ts
        d=m.d
        
        bestdays=asgood(d.times,disttogether[1,:],alpha)
        print "Best days for both groups: %i [%i-%i]"%bestdays
        print "Best day for first group: %i" %d.times[np.argmax(distneg[1,:])] 
        print "Best day for second group: %i" %d.times[np.argmax(distpos[1,:])] 
        
        f=pl.figure(figsize=(3.27,2.25))
        host1=host_subplot(111)
        f.subplots_adjust(hspace=0.3)
        ax1=host1
        ax2=host1.twinx()
        l=ax2.plot([0,140],[alpha*(disttogether[1,:].max())]*2,color='0.5',alpha=1,lw=0.75)
        ax2.plot([bestdays[0]]*2,[0,1],'-.',color='0.5',alpha=.75,lw=.75)
        ax2.plot([bestdays[1]]*2,[0,1],'-',color='0.5',alpha=.75,lw=.75)
        ax2.plot([bestdays[2]]*2,[0,1],'-',color='0.5',alpha=.75,lw=.75)    
        
        ax2.plot(disttogether[1,:],'r-')
        ax2.fill_between(d.times,disttogether[0,:],disttogether[2,:],facecolor='r', lw=0,alpha=0.12)
        ax1.set_xlabel('days post challenge')
        ax2.set_ylabel('day-selection score, $Q$')
        l=ax1.plot(ts,pdfI1[1,:],'-k')
        l=ax1.fill_between(ts,pdfI1[0,:], pdfI1[2,:],facecolor='k', lw=0,alpha=0.12)
        l=ax1.plot(ts[1:],pdfU[1,:],'--',color='k')
        l=ax1.fill_between(ts[1:],pdfU[0,:], pdfU[2,:],facecolor='k', lw=0,alpha=0.12)
        l=ax1.plot(ts,pdfI2[1,:],'-b')
        l=ax1.fill_between(ts,pdfI2[0,:], pdfI2[2,:],facecolor='b', lw=0,alpha=0.12)
        l=ax1.plot(ts[1:],pdfU[1,:],'--',color='b')
        l=ax1.fill_between(ts[1:],pdfU[0,:], pdfU[2,:],facecolor='b', lw=0,alpha=0.12)
        ax1.set_ylabel('density')
        f.savefig(m.saveTo+'-plotBestDays.'+m.figFormat,bbox_inches='tight',dpi=600)
        print "Plotted best day score, see "+m.name+"-plotBestDays."+m.figFormat
        return f,ax1,ax2


TimeData=df.TimeData
