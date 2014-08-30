DISE: Dose-Invariant Susceptibility Estimator
============

Estimating susceptibility distributions from dose-response curves and from survival over time of hosts challenged with escalating virus doses.

Further documentation and testing sometime later this year. In case you need help or find a problem or a bug, please email to (dpessoa@igc.gulbenkian.pt). Please note, this code has not been tested in Windows yet.

To test the homogeneous and heterogeneous models on survival data of two groups challenged with different viral doses, see ./bin/runTestHom.py

To estimate parameters of natural mortality from control survival data, see ./bin/runControlEst.py

In case the homogeneous model is suited for the first group, to estimate infection parameters, including mortality and susceptibility distribution of the second group compared to the first, see ./bin/runEst.py

To estimate infection parameters from day-mortality data, see ./bin/runDayEst.py
