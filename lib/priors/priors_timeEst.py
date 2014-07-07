import pymc as py

# Infection parameters
p=py.Uniform('p',0,1,10**-6)
a2=py.Uniform('a2',0.1,10,0.2)
b2=py.Uniform('b2',0.1,10,0.1)
eps=py.TruncatedNormal('eps',mu=0,tau=1/(0.00125**2),a=0,b=1/0.00125) # Truncated with values restricted between 0 and 1

# Parameters for the time to death of controls (and uninfected)
k=py.Normal('k',mu=0.0011715764701768433,tau=1/(0.0003659234910936011)**2)
meanU=py.Normal('meanU',mu=117.25340837531996,tau=1/(1.2902129894769683)**2)
sU=py.Normal('sU',mu=120.52152424700324,tau=1/(22.027083525382587)**2)

# Parameters for the time to death of infected
meanI1=py.Uniform('meanI1',0.,meanU,value=23.3)
sI1=py.Uniform('sI1',0.,100,value=12)
meanI2=py.Uniform('meanI2',0.,meanU,value=23.3)
sI2=py.Uniform('sI2',0.,100,value=12)

#Save the name of the parameters, in the order you prefer to see them in the saved results
parameters=['p','a2','b2','eps']
parameters.extend(['meanU','sU','k'])
parameters.extend(['meanI1','sI1'])
parameters.extend(['meanI2','sI2'])
