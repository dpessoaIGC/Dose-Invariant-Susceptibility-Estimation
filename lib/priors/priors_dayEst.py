import pymc as py

# Infection parameters
p=py.Uniform('p',0,1,10**-6)
a2=py.Uniform('a2',0.1,10,0.2)
b2=py.Uniform('b2',0.1,10,0.1)
eps=py.TruncatedNormal('eps',mu=0,tau=1/(0.00125**2),a=0,b=1/0.00125) # Truncated with values restricted between 0 and 1

#Save the name of the parameters, in the order you prefer to see them in the saved results
parameters=['p','a2','b2','eps']
