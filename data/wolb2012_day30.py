# Data from day 30 mortality of Wolbachia flies
import numpy as np

dataLabels=(r'Wolb$^-$',r'Wolb$^+$')
dataName='wolb2012_day30'
doses=np.array([0,10**4,10**5,10**6,10**7,10**8,10**9,10**10])
response1=np.array([0,0,4,47,47,48,44,46.])
nhosts1=np.array([44,46,46,50,49,48,44,46.])
response2=np.array([2, 2, 3, 13, 29, 43, 42, 46.])
nhosts2=np.array([46,45,49,47,48,48,50,48.])
