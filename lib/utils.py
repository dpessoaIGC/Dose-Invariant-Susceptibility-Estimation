""" Functions used in model files. """
import numpy as np, random, os, sys, csv
import modelFunctions as mf
import dataFunctions as df
from scipy.integrate import quad
import scipy.stats as st
import scipy.special as sp
from copy import deepcopy
from functools import wraps
from scipy.stats import scoreatpercentile as sap
from matplotlib import rcParams
rcParams.update({'font.size': 10})
rcParams['axes.labelsize'] = 'large'
rcParams['font.serif'] = 'Times New Roman'
rcParams['font.family']='serif'

# Dose-Response models
@np.vectorize
def pi_hom(dose,p,eps):
    """Returns the probability of infection from the homogeneous model. 

Input:
- dose (float): amount of virus the hosts are challenged with.
- p (float): probability of infection for each viral particle
- eps (float): probability of ineffective challenge."""
    return((1-np.exp(-dose*p))*(1-eps))

@np.vectorize
def f_beta(s,dose,p,a,b):
    return(np.exp(-dose*p*s)*(s**(a-1))*((1-s)**(b-1))/sp.beta(a,b))

@np.vectorize
def pi_het(dose,p,a,b,eps):
    """Returns the probability of infection from the homogeneous model. 

Input:
- dose (float): amount of virus the hosts are challenged with.
- p (float): probability of infection for each viral particle
- a,b (float): shape parameters for the Beta distribution of susceptibilities
- eps (float): probability of ineffective challenge."""
    
    small=0.001
    return((1-(quad(f_beta,0,0+small,args=(dose,p,a,b),full_output=1)[0]+quad(f_beta,0+small,1-small,args=(dose,p,a,b),full_output=1)[0]+quad(f_beta,1-small,1,args=(dose,p,a,b),full_output=1)[0]))*(1-eps))

# Gamma densities
@np.vectorize
def gpdfInt(t1,t2,c,tau):
    """Gamma intensities, integrating from t1 to t2. Equals the probability of an event between t1 and t2."""
    return st.gamma.cdf(t2,c,loc=0,scale=tau)-st.gamma.cdf(t1,c,loc=0,scale=tau)

# Uniform densities
@np.vectorize
def ucdf(t,tmax):
    """Cumulative Density Function (cdf) of a Uniform distribution between 0 and tmax)."""
    return t/tmax if t<tmax else 1.

@np.vectorize
def updf(t,tmax):
    """Probability Density Function (cdf) of a Uniform distribution between 0 and tmax)."""
    return 1./tmax if t<tmax else 0.

# Gamma*Uniform densities
@np.vectorize
def kpdf(t,c,tau,k):
    """Probability Density Function (cdf) of a mixture of a time-independent Uniform distribution [0,1/k] and a Gamma distribution (c,tau)."""
    return k*(1-st.gamma.cdf(t,c,loc=0,scale=tau))+(1-k*t)*st.gamma.pdf(t,c,loc=0,scale=tau)

@np.vectorize
def kpdfInt(t1,t2,cg,tau,k):
    """Probability of an event between t1 and t2 of a mixture of a time-independent Uniform distribution [0,1/k] and a Gamma distribution (c,tau)."""
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
- func (str): function name
- params (array list): list with the parameters traces
- shape (tuple): number of rows and columns of the resulting array

Returns:
- An executable string (str)
    """
    call='%s'%func
    call+='np.array([%s.tolist()]*%i)'%(params[0],shape[1])
    for v in params[1:]:
        call+=', np.array([%s.tolist()]*%i).T'%(v,shape[0])
    return call+')'

class ProgressBar(object):
    """ Prints a progress bar in terminal. """
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


def initializeFolder(savePath,name,bOverWrite):
    """Creates a folder identified by user preferences or a mixture of a data descriptor and a model descriptor. If folder already exists, creates, inside the existing folder, a folder called Run 2. If Run 2 exists already, it checks if Run 3 exists, and so on. """
    if savePath==None:
        savePath=os.path.join('.','results')
    path=os.path.join(savePath,name)
    path_copy=deepcopy(path)
    if not bOverWrite:
        if os.path.exists(path_copy):
            fi=2
            poss_path=os.path.join(path_copy,'Run (%i)'%fi)
            while os.path.exists(poss_path):
                fi+=1
                poss_path=os.path.join(path_copy,'Run (%i)'%fi)
            path=poss_path
        
        os.makedirs(path)
    
    bexisted=True
    if not os.path.exists(path):
        os.makedirs(path)
        bexisted=False
    
    if path[-1]!=os.path.sep:
        path+=os.path.sep
        
    print "Results will be saved in %s folder '%s'"%('existing' if bexisted else 'new',path)    
    return path
    

class DocInherit(object):
    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden: break

        @wraps(self.mthd, assigned=('__name__','__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError, ("Can't find '%s' in parents"%self.name)
        func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit

def readcsv(dataPath):
    """Reads data file.

Input:
- dataPath (str): path to csv file containing only numbers (',' delimiter).

Output:
- timesDeath (list of arrays): a list of observed times of deaths for each 
dose (list of arrays).

- survivors (int arr): number of survivors up to tmax
- tmax (int): last day of observation
- times (int arr): an array with days. Starts from 0 (day of challenge) to tmax.
- doses (float arr): an array with the doses used to challenge hosts.
- ndoses (int): number of doses.
- nhosts (int arr): number of challenged hosts per dose.
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
