import pymc as py

# Parameters for the time to death of controls
meanU=py.Uniform('meanU',0,139,106.3)
sU=py.Uniform('sU',0,500.,35.)
k=py.Uniform('k',10**-6,10**-2,0.00065)

#Save the name of the parameters, in the order you prefer to see them in the saved results
parameters=['meanU','sU','k']
