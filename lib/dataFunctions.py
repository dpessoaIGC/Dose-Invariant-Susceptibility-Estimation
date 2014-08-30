""" Includes all functions relative to reading/saving data. """
import csv, pickle
import numpy as np
from copy import deepcopy
import utils as ut

class Data(object):
    @classmethod
    def fromPickle(DayData,filename):
        saved=pickle.load(open(filename))
        return DayData(**saved)
    
    def copy(self):
        return self.__class__(**self.__dict__)
    
    def pickle(self, filename):
        pickle.dump(self.__dict__,open(filename,'w'))

class DayData(Data):
    """ Prepares data recorded a fixed amount of time after challenge for model definition. Initialize
    from a csv file using class method fromCSV.
        
    Properties:
    - response1, response2 (int arr): number of hosts for which the challenge 
    ellicited a response for group 1 or group 2.

    - nhosts1, nhosts2 (int arr): number of hosts from group 1 (or group 2) 
    challenged with each dose.

    - doses (float arr): doses used to challenge hosts. These should be the same for
     both groups (the model could deal with this not being the case, but the code 
     was written under this assumption).
     
    - ndoses (int): number of doses.
    - dataPath (str): path to file from which the data was read    
    """
    def __init__(self, response1, response2,nhosts1,nhosts2,doses,dataName):
        for v in ['response1','response2','nhosts1','nhosts2','doses','dataName']:
            setattr(self,v,deepcopy(eval(v)))
            
    
    @classmethod
    def fromCSV(DayData,dataPath,dataName):
        """Prepares data for model definition.

        Input:
        - dataPath(str): path to csv file corresponding to survival observed after a 
        fixed time since challenge. The csv files should have 4 columns, each with one
        of the headings: group, dose, response, nhosts. Each line corresponds to one 
        challenge, the group should be either 1 or 2 (to indicate which group the 
        response was observed in), the dose should be 0 for control, the response is the
        number of challenged hosts which gave rise to a response (in our case, the 
        number of flies dead) and nhosts is the total number of hosts from this group 
        challenged with this dose.

        - dataName (str): a text string which should have no spaces and should be 
        descriptive of the data. Will be used to name folder and files of saved 
        results (ex: 'wolb2012'). 

        Returns a Data object. 
        """
        reader=csv.reader(open(dataPath))
        l=np.array(list(reader))
        names=['group','dose','response','nhosts']
        if bool(set(names).difference(set(['group','dose','response','nhosts']))):
            print "The column labels do not correspond exactly to {group, dose, response, nhosts}. Please check that order of the data in the columns is correct."
        else:
            names=l[0]
        
        grouped={}
        for i in range(len(l[0])):
            grouped[names[i]]=l[1:,i]
        
        grouped['group']=grouped['group'].astype(int)
        grouped['dose']=grouped['dose'].astype(float)
        grouped['response']=grouped['response'].astype(int)
        grouped['nhosts']=grouped['nhosts'].astype(int)
        
        doses1=grouped['dose'][grouped['group']==1]
        doses2=grouped['dose'][grouped['group']==2]
        response1=grouped['response'][grouped['group']==1]
        response2=grouped['response'][grouped['group']==2]
        nhosts1=grouped['nhosts'][grouped['group']==1]
        nhosts2=grouped['nhosts'][grouped['group']==2]
        
        if np.any(doses1!=doses2):
            raise DataError("Doses not the same in two datasets, please check the data")
        
        return DayData(response1, response2,nhosts1,nhosts2,doses1,dataName)



class TimeData(Data):
    """Stores data from survival over time.

    Properties:
    - timesDeath1, timesDeath2 (list of arrays): a list of observed times of deaths 
    for each dose for group 1/2.
    - survivors1, survivors2 (arr): number of survivors up to tmax in group 1/2.
    - tmax (int): last day of observation
    - times (int arr): an array with days. Starts from 0 (day of challenge) to tmax.
    - doses (float arr): an array with the doses used to challenge hosts.
    - ndoses (int): number of doses.
    - nhosts1, nhosts2 (int arr): number of challenged hosts per dose for group 1/2.
    - dataPath1, dataPath2 (str): file from which the data was read.
    """
    def __init__(self,timesDeath1,timesDeath2,survivors1,survivors2,nhosts1,nhosts2,tmax,times,doses,ndoses,dataName,dataPath1,dataPath2, alldata=None):
        
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
        self.dataPath1=deepcopy(dataPath1)
        self.dataPath2=deepcopy(dataPath2)
        if alldata:
            self.alldata=alldata
    
    @classmethod
    def fromCSV(TimeData,dataPath1,dataPath2, dataName):
        """Prepares data of survival over time for model definition. Initialize
        from a csv file using class method fromCSV.

            Input:
            - dataPath1, dataPath2 (str): paths to csv files corresponding to the survival 
            over time of each group. The csv files should have the days of observations in 
            each column, starting from 0 (which will indicate how many hosts were challenged
            in each dose) in second column (the first column will have the doses). The last 
            column will indicate the last day of observation and the number of hosts which 
            survived each of the challenges. The results from each of the challenge doses 
            should be put in each line, starting from the second line (the first line will 
            have the days of observation). The doses and final day of observation should be 
            the same for both groups.

            - dataName (str): a text string which should have no spaces and should be 
            descriptive of the data. Will be used to name folder and files of saved results 
            (ex: 'survWolb2012'). 

            Returns a Data object. 
        """
        (timesDeath1,survivors1,tmax1,times1,doses1,ndoses1,nhosts1)=ut.readcsv(dataPath1)
        (timesDeath2,survivors2,tmax2,times2,doses2,ndoses2,nhosts2)=ut.readcsv(dataPath2)
        
        if ~((tmax1==tmax2)&(sum(times1==times2)==len(times1))):
            raise DataError("Times of observation not the same in two datasets, please check the data in %s and %s"%(dataPath1,dataPath2))
            
        if ~((ndoses1==ndoses2)&(sum(doses1==doses2)==ndoses1)):
            raise DataError("Doses not the same in two datasets, please check the data in %s and %s"%(dataPath1,dataPath2))
        
        return TimeData(timesDeath1,timesDeath2,survivors1,survivors2,nhosts1,nhosts2,tmax1,times1,doses1,ndoses1,dataName,dataPath1,dataPath2)         
    
    def reduce(self,index):
        """ Retain only data from one dose, for example control (index=(data.doses==0) ). """
        alldata=self.copy()
        data=alldata
        d=self
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
        self.alldata=alldata

class DataError(Exception):
    """ Throw an exception in case the data isn't in the correct format."""
    def __init__( self, value ):
        self.value = value

    def __str__( self ):
        return repr( self.value )
