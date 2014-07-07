""" Functions used in model files. """
import csv, sys, random, pickle
import numpy as np
import scipy as sc
import pylab as pl
import scipy.stats as st
import scipy.special as sp
from copy import deepcopy
from scipy.integrate import quad
from mpl_toolkits.axes_grid1 import host_subplot
from scipy.stats import scoreatpercentile as sap
from matplotlib import rcParams
rcParams.update({'font.size': 10})
rcParams['axes.labelsize'] = 'large'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.family']='serif'

# Dose-Response models
@np.vectorize
def pi_hom(dose,p,eps):
    """ Returns the probability of infection from the homogeneous model. 
    
    Input:
    - dose: amount of virus the hosts are challenged with.
    - p: probability of infection for each viral particle
    - eps: probability of ineffective challenge."""
    return((1-np.exp(-dose*p))*(1-eps))

@np.vectorize
def f_beta(s,dose,p,a,b):
    return(np.exp(-dose*p*s)*(s**(a-1))*((1-s)**(b-1))/sp.beta(a,b))

@np.vectorize
def pi_het(dose,p,a,b,eps):
    """ Returns the probability of infection from the homogeneous model. 
    
    Input:
    - dose: amount of virus the hosts are challenged with.
    - p: probability of infection for each viral particle
    - a,b: shape parameters for the Beta distribution of susceptibilities
    - eps: probability of ineffective challenge."""
    
    small=0.001
    return((1-(quad(f_beta,0,0+small,args=(dose,p,a,b),full_output=1)[0]+quad(f_beta,0+small,1-small,args=(dose,p,a,b),full_output=1)[0]+quad(f_beta,1-small,1,args=(dose,p,a,b),full_output=1)[0]))*(1-eps))

# Transforming observed times to death to indexes in chgT shortlist
def changingTimes(timesDeath1, timesDeath2):
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

# Gamma densities
@np.vectorize
def gpdfInt(t1,t2,c,tau):
    return st.gamma.cdf(t2,c,loc=0,scale=tau)-st.gamma.cdf(t1,c,loc=0,scale=tau)

# Uniform densities
@np.vectorize
def ucdf(t,tmax):
    return t/tmax if t<tmax else 1.

@np.vectorize
def updf(t,tmax):
    return 1./tmax if t<tmax else 0.

# Gamma*Uniform densities
@np.vectorize
def kpdf(t,c,tau,k):
    return k*(1-st.gamma.cdf(t,c,loc=0,scale=tau))+(1-k*t)*st.gamma.pdf(t,c,loc=0,scale=tau)

@np.vectorize
def kpdfInt(t1,t2,cg,tau,k):
    #print t1,t2,cg,tau,k, cI.value, cU.value
    return k*((t2-t1)-(quad(lambda t: sp.gammainc(cg,t/tau),t1,t2)[0]))+sp.gammainc(cg,t2/tau)-sp.gammainc(cg,t1/tau)-k*cg*tau*(sp.gammainc(cg+1,t2/tau)-sp.gammainc(cg+1,t1/tau))

def hpd(data, level=0.95) :
    """ The Highest Posterior Density (credible) interval of data at level level.
    :param data: sequence of real values
    :param level: (0 < level < 1)
    """ 
    d = list(data)
    d.sort()
    nData = len(data)
    nIn = int(round(level * nData))
    i = 0
    r = d[i+nIn-1] - d[i]
    for k in range(len(d) - (nIn - 1)) :
        rk = d[k+nIn-1] - d[k]
        if rk < r :
            r = rk
            i = k
    assert 0 <= i <= i+nIn-1 < len(d)
    return (d[i], d[i+nIn-1])

def confint(arr):
    """Returns the median in between the 95% HPD interval of the array.
    """
    res=[[],[],[]]
    #r=hpd(arr)
    r=(sap(arr,2.5),sap(arr,97.5))
    res[0]=r[0]
    res[1]=arr.mean(0)
    res[2]=r[1]
    return np.array(res)

def callArray(func,params,shape):
    """Method to make it more elegant to call a function for all values of parameters in the MCMC samples. 

Input:
func - string with the function name
params - list with the parameters traces
shape - tuple defining the number of rows and columns of the resulting array

Returns:
An executable string
    """
    call='%s'%func
    call+='np.array([%s.tolist()]*%i)'%(params[0],shape[1])
    for v in params[1:]:
        call+=', np.array([%s.tolist()]*%i).T'%(v,shape[0])
    return call+')'    

class Data(object):
    """Stores data.

Properties:
- timesDeath1, timesDeath2: a list of observed times of deaths for each dose (list of arrays) for group 1 (or 2).
- survivors1, survivors2): number of survivors up to tmax in group 1 (or 2)
- tmax: last day of observation
- times: an array with days. Starts from 0 (day of challenge) to tmax.
- doses: an array with the doses used to challenge hosts.
- ndoses: number of doses.
- nhosts1, nhosts2: number of challenged hosts per dose for group 1 (or 2).
- dataPath1, dataPath2: file from which the data was read
"""
    def __init__(self,timesDeath1=None,timesDeath2=None,survivors1=None,survivors2=None,nhosts1=None,nhosts2=None,tmax=None,times=None,doses=None,ndoses=None,dataName=None,dataLabels=('',''),dataPath1=None,dataPath2=None):
        
        self.timesDeath1=deepcopy(timesDeath1)
        self.timesDeath2=deepcopy(timesDeath2)
        self.survivors1=deepcopy(survivors1)
        self.survivors2=deepcopy(survivors2)
        self.nhosts1=deepcopy(nhosts1)
        self.nhosts2=deepcopy(nhosts2)        
        self.tmax=deepcopy(tmax)
        self.times=deepcopy(times)
        self.doses=deepcopy(doses)
        self.ndoses=deepcopy(ndoses)
        self.dataName=deepcopy(dataName)
        self.dataLabels=deepcopy(dataLabels)
        self.dataPath1=deepcopy(dataPath1)
        self.dataPath2=deepcopy(dataPath2)
    
    def copy(self):
        return Data(self.timesDeath1,self.timesDeath2,self.survivors1,self.survivors2,self.nhosts1,self.nhosts2,self.tmax,self.times,self.doses,self.ndoses,self.dataName,self.dataLabels,self.dataPath1,self.dataPath2)
    
    def pickle(self, filename):
        pickle.dump(self.__dict__,open(filename,'w'))

def DataFromPickle(filename):
    saved=pickle.load(open(filename))
    data=Data()
    for key in saved:
        setattr(data,key,saved[key])
    return data

def DataFromCSV(dataPath1,dataPath2, dataName, dataLabels=('','')):
    """Prepares data for model definition.

Input:
- dataPath1, dataPath2: paths to csv files corresponding to the survival over time of each group. The csv files should have the days of observations in each column, starting from 0 (which will indicate how many hosts were challenged in each dose) in second column (the first column will have the doses). The last column will indicate the last day of observation and the number of hosts which survived each of the challenges. The results from each of the challenge doses should be put in each line, starting from the second line (the first line will have the days of observation). The doses and final day of observation should be the same for both groups.

- dataName: a text string which should have no spaces and should be descriptive of the data. Will be used to name folder and files of saved results (ex: 'survWolb2012'). 

- dataLabels (optional): a tuple of labels for group 1 and 2 which will appear in plots. Can have spaces but should not be too long. Example: ('No Wolbachia','Wolbachia')

Properties of Data object:
- timesDeath1, timesDeath2: a list of observed times of deaths for each dose (list of arrays) for group 1 (or 2).
- survivors1, survivors2): number of survivors up to tmax in group 1 (or 2)
- tmax: last day of observation
- times: an array with days. Starts from 0 (day of challenge) to tmax.
- doses: an array with the doses used to challenge hosts.
- ndoses: number of doses.
- nhosts1, nhosts2: number of challenged hosts per dose for group 1 (or 2).
- dataPath1, dataPath2: file from which the data was read
"""    
    (timesDeath1,survivors1,tmax1,times1,doses1,ndoses1,nhosts1)=readcsv(dataPath1)
    (timesDeath2,survivors2,tmax2,times2,doses2,ndoses2,nhosts2)=readcsv(dataPath2)
    err=0
    if ~((tmax1==tmax2)&(sum(times1==times2)==len(times1))):
        print "Times of observation not the same in two datasets, please check the data in %s and %s"%(dataPath1,dataPath2)
        err=1
    if ~((ndoses1==ndoses2)&(sum(doses1==doses2)==ndoses1)):
        print "Doses not the same in two datasets, please check the data in %s and %s"%(dataPath1,dataPath2)
        err=1
    if err==1:
        return
    
    return Data(timesDeath1,timesDeath2,survivors1,survivors2,nhosts1,nhosts2,tmax1,times1,doses1,ndoses1,dataName,dataLabels,dataPath1,dataPath2)    


def readcsv(dataPath):
    """Reads data file.

Input:
- dataPath: path to csv file containing only numbers (',' delimiter).

Output:
- timesDeath: a list of observed times of deaths for each dose (list of arrays).
- survivors: number of survivors up to tmax
- tmax: last day of observation
- times: an array with days. Starts from 0 (day of challenge) to tmax.
- doses: an array with the doses used to challenge hosts.
- ndoses: number of doses.
- nhosts: number of challenged hosts per dose.
"""
    reader=csv.reader(open(dataPath))
    l=np.array(list(reader))
    times=l[0,1:].astype(int)
    tmax=times[-1]
    doses=l[1:,0].astype(float)
    ndoses=len(doses)
    survivalOverTime=l[1:,1:].astype(int)
    survivors=survivalOverTime[:,-1]
    nhosts=survivalOverTime[:,0]
    timesDeath=['']*ndoses
    for di in xrange(ndoses):
        surv=survivalOverTime[di,:]
        deaths=surv[:-1]-surv[1:]
        timesDeath[di]=np.array([item for sublist in [[times[ti+1]]*deaths[ti] for ti in xrange(tmax)] for item in sublist])
    
    return (timesDeath,survivors,tmax,times,doses,ndoses,nhosts)


    # now can use data.bob

    # Import data
    #from dataTimeW import *
    #name=data_description+'-GEv3'
    return data


def plotSurv(m):#(ts,cdf1_ci,cdf2_ci):
    """Plots survival over time for each of the doses. One dose per panel, including surival from both groups. Group 1 in black, group 2 in blue. 

Equivalent figure in article: Figure 4.

Returns:
- f: Figure
- ax: Axes"""
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
        l=ax.plot(d.times,np.array([(d.timesDeath1[di]>ti).sum()+d.survivors1[di] for ti in range(len(d.times))])/d.nhosts1[di].astype(float),'ko',mew=1,alpha=1,ms=1)
        l=ax.plot(d.times,np.array([(d.timesDeath2[di]>ti).sum()+d.survivors2[di] for ti in range(len(d.times))])/d.nhosts2[di].astype(float),'bo',mec='b',mew=1,alpha=1,ms=1)
        l=ax.plot(ts,cdf1_ci[1,di,:],'k-',lw=0.7)#Posterior mean
        l=ax.plot(ts,cdf2_ci[1,di,:],'b-',lw=0.7)
        l=ax.fill_between(ts,cdf1_ci[0,di,:],cdf1_ci[2,di,:],facecolor='black',lw=0,alpha=0.12)
        l=ax.fill_between(ts,cdf2_ci[0,di,:],cdf2_ci[2,di,:],facecolor='blue',lw=0,alpha=0.12) 
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
    
    f.savefig(m.saveTo+'-posteriorSurvival.'+m.figFormat, bbox_inches='tight',dpi=600)
    print "Plotted survival, see "+m.name+"-plotSurvival."+m.figFormat
    return f,ax

def plotBeta(m,a2s=None,b2s=None,name=None): # a2s,b2s
    """Plots the estimated beta distribution with confidence interval. 

Equivalent figure in article: Figure 5B. 

Returns:
- f: Figure
- ax: Axes"""
    if a2s ==None:
        a2s=m.a2s
    if b2s ==None:
        b2s=m.b2s
    x=np.arange(0,1,0.005)
    N=np.array([st.beta.pdf(x,a2s[i],b2s[i]) for i in range(len(a2s))])
    f=pl.figure(figsize=(5,5))
    ax=f.add_subplot(111)
    ax.plot(x,sap(N,50),'b-.',label='Estimated')
    ax.fill_between(x,sap(N,2.5),sap(N,97.5),facecolor='blue', edgecolor='blue',alpha=0.2)
    
    ax.set_title(m.d.dataLabels[1])
    ax.set_xlabel(r'susceptibility, $x$')
    ax.set_ylabel(r'$q(x)$')
    lg=ax.legend(loc='upper center',borderaxespad=2)
    lg.draw_frame(False)
    if name==None:
        name='-posteriorBeta.'
    f.savefig(m.saveTo+name+m.figFormat,bbox_inches='tight')
    print "Plotted beta distribution, see "+m.name+name+m.figFormat
    return f,ax

def plotDoseResponse(m): #(x2,pi1_ci,pi2_ci)
    """Plots the estimated dose-response function with confidence intervals. 

Equivalent figure in article: Figure 5A.

Returns:
- f: Figure
- ax: Axes"""
    x2=m.x2
    pi1_ci=m.pi1_ci
    pi2_ci=m.pi2_ci
    f=pl.figure(figsize=(7,6))
    ax=f.add_subplot(111)
    ax.set_xlabel(r'dose')
    ax.set_ylabel(r'infection')
    ax.fill_between(x2,pi1_ci[0,:],pi1_ci[2,:],facecolor='black',lw=0,alpha=0.12)
    ax.fill_between(x2,pi2_ci[0,:],pi2_ci[2,:],facecolor='blue',lw=0,alpha=0.12)
    ax.plot(x2,pi1_ci[1,:],'k-')
    ax.plot(x2,pi2_ci[1,:],'b-')
    ax.set_xscale('log')
    ax.plot(-1,0,'kx-',markersize=7,mew=2,label=m.d.dataLabels[0])
    ax.plot(-1,0,'bo-',mec='b',markersize=7,mew=2,label=m.d.dataLabels[1])
    xl=[0.15*10**4,0.85*10**11]
    ax.set_xlim(xl)
    ax.set_ylim([-0.09,1.09])
    lg=ax.legend(loc='lower right',borderaxespad=1,numpoints=1)
    lg.draw_frame(False)
    f.savefig(m.saveTo+'-DoseResponse.'+m.figFormat, bbox_inches='tight')
    print "Plotted dose-response curve, see "+m.name+"-posteriorBeta."+m.figFormat
    return f,ax

def plotPosterior(m,a2s=None,b2s=None,pi1_ci=None,pi2_ci=None,name=None):#a2s,b2s,x2,pi1_ci,pi2_ci
    """Plots the posterior intervals for the dose-response curve, the beta distribution of the second group and the correlation between the parameters a and b from the beta distribution.

Equivalent figure in article: Figure 3.

Returns:
- f: Figure
- ax1, ax2, ax3: Axes from each of the panels"""
    if a2s==None:
        a2s=m.a2s
    if b2s==None:
        b2s=m.b2s
    if pi1_ci==None:
        pi1_ci=m.pi1_ci
    if pi2_ci==None:
        pi2_ci=m.pi2_ci
    d=m.d
    
    x2=m.x2
    prevfsize=rcParams['font.size']
    prevlsize=rcParams['axes.labelsize']
    rcParams.update({'font.size': 8})
    rcParams['axes.labelsize']='large'
    f=pl.figure(figsize=(8,1.95)) # Without panel C: (4.86, 1.95) with: (8,1.95)
    f.subplots_adjust(wspace=0.45)
    ax1=f.add_subplot(131)
    ax1.set_xlabel(r'dose')
    ax1.set_ylabel(r'infection probability, $\pi$')
    #fill_between(x2,Mneg2[0,:],Mneg2[1,:],facecolor='black',alpha=0.04)
    ax1.fill_between(x2,pi1_ci[0,:],pi1_ci[2,:],facecolor='black',lw=0,alpha=0.12)
    #fill_between(x2,Mpos2[0,:],Mpos2[1,:],facecolor='magenta',alpha=0.04)
    ax1.fill_between(x2,pi2_ci[0,:],pi2_ci[2,:],facecolor='blue',lw=0,alpha=0.12)
    ax1.plot(x2,pi1_ci[1,:],'k-')
    ax1.plot(x2,pi2_ci[1,:],'b-')
    ax1.set_xscale('log')
    ax1.plot(-1,0,'k-',markersize=7,mew=2,label='Homogeneous')
    ax1.plot(-1,0,'b-',mec='b',markersize=7,mew=2,label='Heterogeneous')
    xl=[0.15*(d.doses[d.doses>0][0]),0.85*(d.doses[-1]*10)]
    x=xl[0]
    
    ax1.set_xlim(xl)
    ax1.set_ylim([-0.09,1.09])
    ax1.text(-0.25, 1.15, 'A', transform=ax1.transAxes,
          fontsize=12, fontweight='bold', va='top', ha='right')    
    
    ax2=f.add_subplot(132)
    x=np.arange(0,1,0.005)
    N=np.array([st.beta.pdf(x,a2s[i],b2s[i]) for i in range(len(a2s))])
    ax2.plot(x,sap(N,50),'b-',label='Heterogeneous')
    ax2.fill_between(x,sap(N,2.5),sap(N,97.5),facecolor='blue', lw=0,alpha=0.2)
    ax2.set_xlabel(r'susceptibility, $x$')
    ax2.set_ylabel(r'$q(x)$')    
    ax2.text(-0.25, 1.15, 'B', transform=ax2.transAxes,
              fontsize=12, fontweight='bold', va='top', ha='right')    
    ax2.set_ylim([0,10])
    
    ax3=f.add_subplot(133)
    ax3.scatter(a2s,b2s,c='b',edgecolor='grey',lw=0.1,alpha=0.5,s=2)
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
        name='-plotPosterior.'
    f.savefig(m.saveTo+name+m.figFormat, bbox_inches='tight',dpi=600)
    rcParams.update({'font.size': prevfsize})
    rcParams['axes.labelsize']=prevlsize
    print "Plotted dose-response curve, beta distribution and a-b correlation, see "+ m.name+name+m.figFormat
    return f,ax1,ax2,ax3

def stdDeaths(m):
    """ Calculates standard deviation in times to death.

Returns:
- standard deviation in times to death of group 1 and of group 2.
"""
    
    def stdGamma(shape, scale):
        return (shape*(scale)**2)**0.5
    
    stdnegs=ut.confint(stdGamma(m.sI1s,m.tauI1s))
    stdposs=ut.confint(stdGamma(m.sI2s,m.tauI2s))
    return stdnegs, stdposs

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
- f: Figure
- ax1, ax2: Axes. ax2 corresponds to the axes on the right.
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

class ProgressBar(object):
    def __init__(self,startText):
        self.start(startText)
    
    def start(self,text):
        print text
        self.width=40
        sys.stdout.write("[%s]" % (" " * self.width))
        sys.stdout.flush()
        sys.stdout.write("\b" * (self.width+1))
        self.printed=0        
    
    def iter(self,percentageProgress):
        if percentageProgress*self.width>=1:
            sys.stdout.write("-"*int(self.width*percentageProgress))
            self.printed+=int(self.width*percentageProgress)
        elif self.printed<40:
            prob=(random.random()<(percentageProgress*self.width))
            if prob:
                sys.stdout.write("-")
                self.printed += 1
        sys.stdout.flush()
    
    def finish(self):
        if self.printed<self.width:
            sys.stdout.write("-"*int(self.width-self.printed))
        print ""

        def hpd(data, level=0.95) :
            """ The Highest Posterior Density (credible) interval of data at level level.
            :param data: sequence of real values
            :param level: (0 < level < 1)
            """ 
            d = list(data)
            d.sort()
            nData = len(data)
            nIn = int(round(level * nData))
            i = 0
            r = d[i+nIn-1] - d[i]
            for k in range(len(d) - (nIn - 1)) :
                rk = d[k+nIn-1] - d[k]
                if rk < r :
                    r = rk
                    i = k
            assert 0 <= i <= i+nIn-1 < len(d)
            return (d[i], d[i+nIn-1])

def write_vals(m):#M2,de,a, chains=[0]):
    f=open(m.saveTo+'-posteriorValues.csv','w')
    f.write('\t'.join(['Parameter','mean','median','95% HPD','std'])+'\n')
    for v in m.parameters:
        trac=getattr(m,v+'s')
        hpdi=hpd(trac,0.95)
        form='%.2f'
        if v.startswith('p') or v.startswith('k') or v.startswith('e'):
            form='%.2e'
        
        f.write('\t'.join([v,form%trac.mean(),form%sap(trac,50),('['+form+', '+form+']')%hpdi,form%trac.std()])+'\n')
    
    f.close()
    print "Saved posterior median and confidence intervals for each parameter, see "+ m.name+'-posterior_values.csv'


def plotControlSurvival(m):
    """Plots posterior distribution for the times to death estimated from control mortality.

Equivalent figure in article: Figure 1.

Returns:
f - Figure
ax1,ax2 - Axes from panel A and B, respectively
ax3 - Axes that only has the legend
"""
    ts=m.ts
    cdf1_ci=m.cdf1_ci
    cdf2_ci=m.cdf2_ci
    d=m.alldata
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
    
    ax3.set_xlim(0,1)
    ax3.set_ylim(0,1)
    ax3.axison=False
    ax3.legend(frameon=False,numpoints=1, loc='center right',prop={'size':10})
    labels=['A','B']
    label2=d.dataLabels
    for axi,ax in enumerate([ax1,ax2]):
        ax.set_ylabel('survival')
        ax.set_xlabel('days post challenge')
        ax.text(-0.15, 1.15, labels[axi], transform=ax.transAxes,
                  fontsize=12, fontweight='bold', va='top', ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        l=ax.text(139,1.05,s=label2[axi],ha='right',va='top',fontsize=12)
    f.savefig(m.saveTo+'-posteriorSurvival.'+m.figFormat, bbox_inches='tight',dpi=600)
    rcParams.update({'font.size': prevfsize})
    rcParams['axes.labelsize']=prevlsize    
    print "Plotted posterior survival, see "+m.name+'-posteriorSurvival.'+m.figFormat
    return f,ax1,ax2,ax3

def plotSurvivalHomHet(self):
        """Plots survival over time for each of the doses. One dose per panel, including surival from both groups. Group 1 in black, group 2 in blue. (Same as figure S1,2 of the article)."""
        d=self.d
        ts=self.ts
        cdf1hom=self.cdf1hom_ci
        cdf1het=self.cdf1het_ci
        cdf2hom=self.cdf2hom_ci
        cdf2het=self.cdf2het_ci
        """Posterior survival curves = CDF"""
        rcParams.update({'font.size': 8})
        rcParams['axes.labelsize'] = 'medium'
        rcParams['font.sans-serif'] = 'Arial'
        rcParams['axes.linewidth']=0.5
        rcParams['mathtext.default']='regular'    
        f1=pl.figure(figsize=(8,3))
        f2=pl.figure(figsize=(8,3))  
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
            # Wolbachia negative
            ax1=f1.add_subplot(2,4,di+1)
            l=ax1.plot(d.times,np.array([(d.timesDeath1[di]>ti).sum()+d.survivors1[di] for ti in range(len(d.times))])/d.nhosts1[di],'ko',mew=1,ms=1,alpha=0.6)
            
            l=ax1.plot(ts,cdf1hom[1,di,:],'-',lw=0.5,color='black')#Posterior mean
            l=ax1.plot(ts,cdf1het[1,di,:],'--',lw=0.5,color='black')
            l=ax1.fill_between(ts,cdf1hom[0,di,:],cdf1hom[2,di,:],facecolor='black',alpha=0.12)
            l=ax1.fill_between(ts,cdf1het[0,di,:],cdf1het[2,di,:],facecolor='black',alpha=0.12) 
            l=ax1.set_xticks([0,50,100,140])
            l=ax1.set_ylim([-0.03,1.05])
            tt='control' if dose==0 else r'10$^{%i}$ TCID$_{50}$'%int(np.log10(dose))
            l=ax1.text(139,1.05,s=tt,ha='right',va='top',fontsize=8)        
            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            ax1.get_xaxis().tick_bottom()   # remove unneeded ticks 
            ax1.get_yaxis().tick_left()
            l=ax1.set_xticks([0,50,100,140])
            l=ax1.set_xticklabels([])
            l=ax1.set_yticklabels([])
            l=ax1.tick_params('both', length=2.5)
            l=ax1.annotate(letters[di],xy=figxy[di], xycoords='figure fraction',fontsize=10,fontweight='bold')
            
            ax2=f2.add_subplot(2,4,di+1)
            
            l=ax2.plot(d.times,np.array([(d.timesDeath2[di]>ti).sum()+d.survivors2[di] for ti in range(len(d.times))])/d.nhosts2[di],'bo',mec='b',mew=1,ms=1,alpha=0.6)
            l=ax2.plot(ts,cdf2hom[1,di,:],'-',lw=0.5,color='blue')#Posterior mean
            l=ax2.plot(ts,cdf2het[1,di,:],'--',lw=0.5,color='blue')
            l=ax2.fill_between(ts,cdf2hom[0,di,:],cdf2hom[2,di,:],facecolor='blue',alpha=0.12)
            l=ax2.fill_between(ts,cdf2het[0,di,:],cdf2het[2,di,:],facecolor='#add8e6',alpha=0.12) 
            l=ax2.set_xticks([0,50,100,140])
            l=ax2.set_ylim([-0.03,1.05])
            l=ax2.text(139,1.05,s=tt,ha='right',va='top',fontsize=8)        
            ax2.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.get_xaxis().tick_bottom()   # remove unneeded ticks 
            ax2.get_yaxis().tick_left()        
            l=ax2.set_xticks([0,50,100,140])
            l=ax2.set_xticklabels([])
            l=ax2.set_yticklabels([])        
            l=ax2.tick_params('both', length=2.5)
            l=ax2.annotate(letters[di],xy=figxy[di], xycoords='figure fraction',fontsize=10,fontweight='bold')
        
        dis=[0,4]
        for di in dis:
            ax=f1.add_subplot(2,4,di+1)
            l=ax.set_ylabel('survival')# proportion')
            l=ax.set_yticklabels(ax.get_yticks())
            
            ax=f2.add_subplot(2,4,di+1)
            l=ax.set_ylabel('survival')# proportion')
            l=ax.set_yticklabels(ax.get_yticks())        
        
        dis=[4,5,6,7]
        for di in dis:
            ax=f1.add_subplot(2,4,di+1)
            l=ax.set_xlabel('days post challenge')
            l=ax.set_xticklabels(ax.get_xticks())        
            ax=f2.add_subplot(2,4,di+1)
            l=ax.set_xlabel('days post challenge')
            l=ax.set_xticklabels(ax.get_xticks())         
        
        fil1=self.saveTo+'-plotSurvival1'+self.figFormat
        fil2=self.saveTo+'-plotSurvival2'+self.figFormat
        f1.savefig(fil1, bbox_inches='tight',dpi=600)
        f2.savefig(fil2, bbox_inches='tight',dpi=600)
        print "Plotted posterior survival of group 1 in",self.name+'-plotSurvival1'+self.figFormat
        print "Plotted posterior survival of group 2 in",self.name+'-plotSurvival2'+self.figFormat
        return f1,f2