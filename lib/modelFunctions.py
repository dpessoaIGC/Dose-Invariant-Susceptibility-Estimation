""" Super class Models includes all functions that are common to all models. """
import os, sys, pickle, numpy as np, scipy.stats as st, pylab as pl
import importlib, shutil, pymc as py
from scipy.stats import scoreatpercentile as sap
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib import rcParams
import utils as ut
import dataFunctions as df
import logging
logging.captureWarnings(True)

class Models(object):
    colors=['k','b']
    def __init__(self, data, priors, name, path, bRandomIni):
        
        #Save runtime warnings in log file
        logging.basicConfig(filename=path+'warning.log', level=logging.WARNING)
        #console=logging.StreamHandler()
        #console.setLevel(logging.ERROR)
        
        self.name=name
        self.saveTo=path+name
        self.path=path
        self.priors=priors
        if not hasattr(self,'d'):
            self.d=data.copy()
        m=self
        for v in m.vals:
            setattr(m,v,None)
        
        #~~ Priors ~~
        m.parameters=priors.parameters
        for key in m.parameters:
            setattr(m,key,getattr(priors,key))
        
        
        #~~ Likelihood ~~
        self.pickle()
        self.likelihood_setup(bRandomIni)
    
    @classmethod    
    def setup(Model,data, resultsName=None,savePath=None, bOverWrite=False,priorsFile=None, bRandomIni=True):
        """Setting up Model.

Input:
- data (df.Data): see help(df.Data) for further information on Data objects
- resultsName (str): a short name that identifies this MCMC estimation. If None, a name will be created that concatenates data.dataName and the name of the model, in this case, 'timeEst'.
- savePath (str): folder to save all results, if None, will create a folder in ./results called resultsName
- bOverWrite (bool): If a folder already exists, should it be overwritten (True) or should a subfolder be created (False, default)?
- priorsFile (str): name of python file in ./lib/priors containing the definition of the prior distributions of the parameters.
- bRandomIni (bool): should initial values be sampled randomly from prior distribution (True, default)? If not (False), parameter values set in prior file will be used (each parameter should have value=XX set in prior file).

Returns a Model object.
"""    
        d=data.copy()
        if resultsName==None:
            name=d.dataName+Model.__defaultName__
        else:
            name=resultsName
        
        priorsFile=priorsFile
        path= ut.initializeFolder(savePath,name,bOverWrite)
        d.pickle(path+'data.pickle')
        
        #~~ Priors ~~
        if priorsFile==None:
            priorsFile=Model.__defaultPrior__
        #Copy priors files to results folder
        priors=importlib.import_module('lib.priors.'+priorsFile)
        shutil.copyfile(os.path.join('.','lib','priors',priorsFile+'.py'), path+'prior.py')
        return Model(d, priors, name, path,bRandomIni)
    
    def pickle(self):
        save={'path':self.path,'saveTo':self.saveTo, 'name':self.name}
        pickle.dump(save,open(self.path+'model.pickle','w')) 
    
    def resetParameters(self):
        """ Resets all parameters to random initial values. """
        print "Resetting parameter values to random values."
        self.likelihood_setup(True)
    
    def likelihood_setup(self, bRandomIni):
        """ Sets up likelihoods. If bRandomIni, will reset all variables to random values."""
        m=self
        
        if bRandomIni:
            print "Looking for random initial values with non-zero likelihood..."""
            zeroprob=1
            while zeroprob:
                [getattr(m,par).random() for par in m.parameters]
                zeroprob=self.__lik_setup__()
            print "Found initial values, moving on."
        else:
            zeroprob=self.__lik_setup__()
            if zeroprob==1:
                raise ZeroError("Initial values cause likelihood to be zero. Try other initial values or set bRandomIni to True.")
    
    def calcPosterior(self,burnin=0,thinF=1,bOverWrite=False,figFormat='png'):
        """Calculates posterior distributions needed for creating figures.

Input:
M2 (dict) - dictionnary from a MCMC loaded from a pickle
burnin (int) - how many iterations from the begining should be discarded
thinF (int) - thining factor.
bOverWrite (bool) - if these calculations have already been done in the given results folder, with the same burn-in and thining factor, should they be calculated again (False) or not (True, default)?
figFormat (str) - format in which figures should be saved. Examples: 'png' (default),'tiff','pdf','jpg'
"""
        self.figFormat=figFormat
        # Determines if results can be loaded from previous calculations in a pickle
        # (if the file is more recent than the MCMC pickle and the burnin and thinning factors were the same). 
        # In which case, only the plots are created again. 
        # Else, reload the traces and calculate posterior distributions.
        if bOverWrite:
            self.loadMCMC(burnin, thinF)
            self.__calc__()
        else:
            try:
                bMostRecent=os.path.getctime(self.saveTo+'-postcalc.pickle')>os.path.getctime(self.saveTo+'-MCMC.pickle')
                saved=pickle.load(open(self.saveTo+'-postcalc.pickle'))
                par=self.parameters[0]
                # For multiple (independent) chains, uncomment the two following lines and comment the third (see also loadMCMC)
                #nchains=len(M2[par].keys())
                #bSameIter=(getattr(self,par+'s').shape[0]==(len(M2[par][0][burnin:None:thinF])*nchains))
                bSameIter=1
                if (bMostRecent&bSameIter&(saved['burnin']==burnin) & (saved['thinF']==thinF)):
                    print "Imported previous calculations"
                    for v in self.vals:
                        setattr(self,v,saved[v])
                    for v in self.parameters:
                        setattr(self,v+'s', saved[v+'s'])
                    self.__plot__()
                else:
                    self.loadMCMC(burnin, thinF)
                    self.__calc__()
            except (IOError, OSError):
                self.loadMCMC(burnin, thinF)
                self.__calc__()
    
    def loadMCMC(self, burnin, thinF):
        M2=pickle.load(open(self.saveTo+'-MCMC.pickle'))
        for p in self.parameters:
            vals=M2[p][0][burnin:None:thinF].tolist()
            
            # For multiple (independent) chains, see also Model.resetParameters()
            #nchains=len(M2[p].keys())
            #if nchains>1:
            #    for c in nchains[1:]:
            #        vals.extend(M2[p][c][burnin:None:thinF])
            
            setattr(self,p+'s',np.array(vals))
        self.burnin=burnin
        self.thinF=thinF

    def write_vals(self,saveTo=None):
        """ Saves estimated parameters to csv file. 
        
        Input:
        - m (Model)
        - saveTo (str) - path and name of csv file where results should be 
        saved. Defaults to mod.saveTo+'-posteriorValues.csv'.
        """
        m=self
        if saveTo==None:
            fname=m.saveTo+'-posteriorValues.csv'
        else:
            fname=saveTo
        f=open(fname,'w')
        f.write('\t'.join(['Parameter','mean','median','95% HPD','std'])+'\n')
        for v in m.parameters:
            trac=getattr(m,v+'s')
            hpdi=ut.hpd(trac,0.95)
            form='%.2f'
            if v.startswith('p') or v.startswith('k') or v.startswith('e'):
                form='%.2e'
            
            f.write('\t'.join([v,form%trac.mean(),form%sap(trac,50),('['+form+', '+form+']')%hpdi,form%trac.std()])+'\n')
        
        f.close()
        if saveTo==None:
            print "Saved posterior median and confidence intervals for each parameter, see "+ m.name+'-posterior_values.csv'
        else:
            print "Saved posterior median and confidence intervals for each parameter, see "+ saveTo

class DoseResponseModels(Models):
    """ Includes all functions common to dose-response models. """
    def __init__(self, data, priors, name, path, bRandomIni):
        super(DoseResponseModels,self).__init__(data, priors, name, path, bRandomIni)
    
    def plotDoseResponse(self,name=None,colors=None):
        """Plots the estimated dose-response function with confidence intervals. 

        Equivalent figure in article: Figure 5A.

        Returns:
        - f (Figure)
        - ax (Axes)"""
        if colors==None:
            colors=self.colors
        m=self
        x2=m.x2
        pi1_ci=m.pi1_ci
        pi2_ci=m.pi2_ci
        f=pl.figure(figsize=(7,6))
        ax=f.add_subplot(111)
        ax.set_xlabel(r'dose')
        ax.set_ylabel(r'infection')
        ax.fill_between(x2,pi1_ci[0,:],pi1_ci[2,:],facecolor=colors[0],lw=0,alpha=0.12)
        ax.fill_between(x2,pi2_ci[0,:],pi2_ci[2,:],facecolor=colors[1],lw=0,alpha=0.12)
        ax.plot(x2,pi1_ci[1,:],colors[0]+'-')
        ax.plot(x2,pi2_ci[1,:],colors[1]+'-')
        if isinstance( m.d,df.DayData):
            ax.plot(m.d.doses,m.d.response1/m.d.nhosts1.astype(float),colors[0]+'.',alpha=0.8)
            ax.plot(m.d.doses,m.d.response2/m.d.nhosts2.astype(float),colors[1]+'.',alpha=0.8)
        ax.set_xscale('log')
        xl=[0.15*10**4,0.85*10**11]
        ax.set_xlim(xl)
        ax.set_ylim([-0.09,1.09])
        if name==None:
            name='-DoseResponse'
        f.savefig(m.saveTo+name+'.'+m.figFormat, bbox_inches='tight')
        print "Plotted dose-response curve, see "+m.name+"."+m.figFormat
        return f,ax
    
    def plotPosterior(self,name=None, colors=None):
        """Plots the posterior intervals for the dose-response curve, the beta 
        distribution of the second group and the correlation between the 
        parameters a and b from the beta distribution.

        Equivalent figure in article: Figure 3.

        Returns:
        - f (Figure)
        - ax1, ax2, ax3 (Axes) - axes from each of the panels"""
        if colors==None:
            colors=self.colors
        m=self
        d=m.d
        a2s=self.a2s
        b2s=self.b2s
        pi1_ci=m.pi1_ci
        pi2_ci=m.pi2_ci
        x2=m.x2
        prevfsize=rcParams['font.size']
        prevlsize=rcParams['axes.labelsize']
        rcParams.update({'font.size': 8})
        rcParams['axes.labelsize']='large'
        f=pl.figure(figsize=(8,1.95))
        f.subplots_adjust(wspace=0.45)
        ax1=f.add_subplot(131)
        ax1.set_xlabel(r'dose')
        ax1.set_ylabel(r'infection probability, $\pi$')
        ax1.fill_between(x2,pi1_ci[0,:],pi1_ci[2,:],facecolor=colors[0],lw=0,alpha=0.12)
        ax1.fill_between(x2,pi2_ci[0,:],pi2_ci[2,:],facecolor=colors[1],lw=0,alpha=0.12)
        ax1.plot(x2,pi1_ci[1,:],colors[0]+'-')
        ax1.plot(x2,pi2_ci[1,:],colors[1]+'-')
        if isinstance( m.d,df.DayData):
            ax1.plot(m.d.doses,m.d.response1/m.d.nhosts1.astype(float),colors[0]+'.',alpha=0.8)
            ax1.plot(m.d.doses,m.d.response2/m.d.nhosts2.astype(float),colors[1]+'.',alpha=0.8)
        ax1.set_xscale('log')
        ax1.plot(-1,0,colors[0]+'-',markersize=7,mew=2,label='Homogeneous')
        ax1.plot(-1,0,colors[1]+'-',mec=colors[1],markersize=7,mew=2,label='Heterogeneous')
        xl=[0.15*(d.doses[d.doses>0][0]),0.85*(d.doses[-1]*10)]
        x=xl[0]
        
        ax1.set_xlim(xl)
        ax1.set_ylim([-0.09,1.09])
        ax1.text(-0.25, 1.15, 'A', transform=ax1.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')    
        
        ax2=f.add_subplot(132)
        x=np.arange(0,1,0.005)
        N=np.array([st.beta.pdf(x,a2s[i],b2s[i]) for i in range(len(a2s))])
        ax2.plot(x,sap(N,50),colors[1]+'-',label='Heterogeneous')
        ax2.fill_between(x,sap(N,2.5),sap(N,97.5),facecolor=colors[1], lw=0,alpha=0.2)
        ax2.set_xlabel(r'susceptibility, $x$')
        ax2.set_ylabel(r'$q(x)$')    
        ax2.text(-0.25, 1.15, 'B', transform=ax2.transAxes,
                  fontsize=12, fontweight='bold', va='top', ha='right')    
        #ax2.set_ylim([0,10])
        
        ax3=f.add_subplot(133)
        ax3.scatter(a2s,b2s,c=colors[1],edgecolor='grey',lw=0.1,alpha=0.5,s=2)
        ax3.set_xlabel('estimated a')
        ax3.set_ylabel('estimated b')   
        ax3.set_xlim([0.1,10])
        ax3.set_ylim([0.1,10])
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.plot(sap(a2s,50),sap(b2s,50),'ro',mec='r',ms=3)
        ax3.text(-0.25, 1.15, 'C', transform=ax3.transAxes,
                  fontsize=12, fontweight='bold', va='top', ha='right')
        
        if name==None:
            name='-plotPosterior'
        f.savefig(m.saveTo+name+'.'+m.figFormat, bbox_inches='tight',dpi=600)
        rcParams.update({'font.size': prevfsize})
        rcParams['axes.labelsize']=prevlsize
        print "Plotted dose-response curve, beta distribution and a-b correlation, see "+ m.name+name+'.'+m.figFormat
        return f,ax1,ax2,ax3
    

class TimeModels(Models):
    """ Includes all methods that are common to all survival over time models. """
    
    def __init__(self, data, priors, name, path, bRandomIni):
        # Reducing times to those where a change occurs at least once
        # (compute the probabilities only at the times there was change)
        self.d=data.copy()
        d=self.d
        m=self
        (chgT, iTd1, iTd2) = self.changingTimes(d.timesDeath1, d.timesDeath2)
        m.chgT=chgT
        m.iTd1=iTd1
        m.iTd2=iTd2
        super(TimeModels,self).__init__(data, priors, name, path, bRandomIni)
    
    @classmethod
    def savedModel(Model,path):
        """Initializes a model from saved results folder. """
        path+=('' if path[-1]==os.path.sep else os.path.sep)
        data=df.Data.fromPickle(path+'data.pickle')
        # /!\ Only one model should be loaded per python session, or priors may get mixed up...
        sys.path.append(path)
        priors=importlib.import_module('prior')
        model=pickle.load(open(path+'model.pickle'))
        mod=Model(data,priors,model['name'], path)
        try:
            traces=pickle.load(open(path+mod.name+'.pickle'))
            for v in mod.parameters:
                setattr(self,v+'s', saved[v+'s'])
        except IOError:
            print "There was some error with the given path: %s"%path
        return mod
    
    def likelihood_setup(self,bRandomIni):
        m=self
        m.tauU=py.Lambda('tauU',lambda mean=m.meanU, s=m.sU: mean/s)
        super(TimeModels,self).likelihood_setup(bRandomIni)
    
    def changingTimes(self,timesDeath1, timesDeath2):
        """ Transforming observed times to death to indexes in chgT shortlist 
        (list of only the times where at least one individual changed state)."""
        l=[]
        if not hasattr(timesDeath1,'shape'):
            [l.extend(val) for val in timesDeath1]
            [l.extend(val) for val in timesDeath2]
        else:
            l.extend(timesDeath1)
            l.extend(timesDeath2)
        chgT=np.array(tuple(set(l)))
        chgT.sort()
        
        def t_ti(ti):
            "Returns the times from the time indexes"
            return chgT[ti]
            
        @np.vectorize
        def ti_t(t):
            "Returns the time indexes (in chgT) from the times"
            return np.arange(len(chgT))[chgT==t][0]
        
        #Changing from time interval of deaths to index of changing times
        iTd1=np.array([ti_t(val) for val in timesDeath1])
        iTd2=np.array([ti_t(val) for val in timesDeath2])
        return (chgT, iTd1, iTd2)
    
    def plotSurvival(self, name=None, colors=None, extra=''):
        """Plots survival over time for each of the doses. One dose per panel, 
        including surival from both groups. Group 1 in black, group 2 in blue. 
        
        Equivalent figure in article: Figure 4.

        Returns:
        - f (Figure)
        - ax (Axes)"""
        if colors==None:
            colors=self.colors
        m=self
        d=m.d
        ts=m.ts
        cdf1_ci=m.cdf1_ci
        cdf2_ci=m.cdf2_ci
        rcParams.update({'font.size': 8})
        rcParams['axes.labelsize'] = 'medium'
        rcParams['font.sans-serif'] = 'Arial'
        rcParams['axes.linewidth']=0.5
        rcParams['mathtext.default']='regular'
        
        f=pl.figure(figsize=(8,3))
        letters=['A','B','C','D','E','F','G','H']
        figxy=[np.array((.047,.91))]
        aa=.225
        figxy+=[figxy[0]+np.array((aa+.035,0))]
        figxy+=[figxy[1]+np.array((aa,0))]
        figxy+=[figxy[2]+np.array((aa,0))]
        figxy+=[figxy[0]+np.array((0,-0.43))]
        figxy+=[figxy[4]+np.array((aa+0.035,0))]
        figxy+=[figxy[5]+np.array((aa,0))]
        figxy+=[figxy[6]+np.array((aa,0))]
        
        for di,dose in enumerate(d.doses):
            ax=f.add_subplot(2,4,di+1)
            l=ax.plot(d.times,np.array([(d.timesDeath1[di]>ti).sum()+d.survivors1[di] for ti in range(len(d.times))])/d.nhosts1[di].astype(float),colors[0]+'o',mec=colors[0],mew=1,alpha=1,ms=1)
            l=ax.plot(d.times,np.array([(d.timesDeath2[di]>ti).sum()+d.survivors2[di] for ti in range(len(d.times))])/d.nhosts2[di].astype(float),colors[1]+'o',mec=colors[1],mew=1,alpha=1,ms=1)
            l=ax.plot(ts,cdf1_ci[1,di,:],colors[0]+'-',lw=0.7)#Posterior mean
            l=ax.plot(ts,cdf2_ci[1,di,:],colors[1]+'-',lw=0.7)
            l=ax.fill_between(ts,cdf1_ci[0,di,:],cdf1_ci[2,di,:],facecolor=colors[0],lw=0,alpha=0.12)
            l=ax.fill_between(ts,cdf2_ci[0,di,:],cdf2_ci[2,di,:],facecolor=colors[1],lw=0,alpha=0.12) 
            l=ax.set_xticks([0,50,100,140])
            l=ax.set_ylim([-0.03,1.05])
            l=ax.set_xticklabels([])
            l=ax.set_yticklabels([])
            l=ax.tick_params('both', length=2.5)
            l=ax.annotate(letters[di],xy=figxy[di], xycoords='figure fraction',fontsize=12,fontweight='bold')
            tt='control' if dose==0 else r'10$^{%i}$ TCID$_{50}$'%int(np.log10(dose))
            l=ax.text(139,1.05,s=tt,ha='right',va='top',fontsize=8)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)        
            ax.get_xaxis().tick_bottom()   # remove unneeded ticks 
            ax.get_yaxis().tick_left()        
        
        dis=[0,4]
        for di in dis:
            ax=f.add_subplot(2,4,di+1)
            l=ax.set_ylabel('survival')
            l=ax.set_yticklabels(ax.get_yticks())
        
        dis=[4,5,6,7]
        for di in dis:
            ax=f.add_subplot(2,4,di+1)
            l=ax.set_xlabel('days post challenge')
            l=ax.set_xticklabels(ax.get_xticks())
        
        if name==None:
            name='-posteriorSurvival'
        f.savefig(m.saveTo+name+'.'+m.figFormat, bbox_inches='tight',dpi=600)
        print "Plotted survival ",extra,", see "+m.name+name+"."+m.figFormat
        return f,ax
    
    def stdDeaths(m):
        """ Calculates standard deviation in times to death.

        Returns:
        - standard deviation in times to death of (group 1, group 2).
        """
    
        def stdGamma(shape, scale):
            return (shape*(scale)**2)**0.5
        
        stdnegs=ut.confint(stdGamma(m.sI1s,m.tauI1s))
        stdposs=ut.confint(stdGamma(m.sI2s,m.tauI2s))
        return stdnegs, stdposs


class ZeroError(Exception):
    """ A clean way to throw an exception in case the initial values 
    give 0 likelihood."""
    def __init__( self, value ):
        self.value = value

    def __str__( self ):
        return repr( self.value )
