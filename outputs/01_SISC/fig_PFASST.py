#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 23:41:37 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from gfm.base import GFMSolver
from gfm.util import setFig, getLastPlotCol

s = GFMSolver(lam=1j, u0=1, dt=0.2*np.pi, L=10)

fineMethod = 'COLLOCATION'
deltaMethod = 'BE'
nodesType = 'LEGENDRE'
qType = 'LOBATTO'
iType = 'RAND'
MCoarse = 2
ratio = 5

s.setFineLevel(M=ratio*(MCoarse-1)+1, method=fineMethod,
    nodes=nodesType, qType=qType)
s.setPhiDelta(deltaMethod)
s.setCoarseLevel(MCoarse, nodes=nodesType, qType=qType)
s.setPhiDeltaCoarse(deltaMethod)

nIter = 20

err = np.zeros(nIter+1)
gfmBnd = np.ones(nIter+1)
errI = np.zeros(nIter+1)
gfmBndI = np.ones(nIter+1)

fig = plt.figure()
figI = plt.figure()


# Initialization
uAlgo = s.getInitU(iType)
delta = np.max(s.getErr(uAlgo))
deltaI = np.max(s.getErr(uAlgo, interface=True))
err[0] = 1
errI[0] = 1

# Error bound computation
errEstimate = s.errBoundFunc('PFASST')
errEstimateI = s.errBoundFunc('PFASST', iOnly=True)
R, _ = s.iterationMatrix('PFASST')
gamma = np.linalg.norm(R, ord=np.inf)
eigs = np.linalg.eigvals(R)
rho = np.max(np.abs(eigs))

# Iterations
for k in range(nIter):
    s.iterate('PFASST', uAlgo)
    err[k+1] = s.getErr(uAlgo)[-1]
    errI[k+1] = s.getErr(uAlgo, interface=True)[-1]
    gfmBnd[k+1] = errEstimate(s.L, k+1)
    gfmBndI[k+1] = errEstimateI(s.L, k+1)

# Plot volume error
plt.figure(fig)
plt.semilogy(delta*err, 'o--', label='Iteration error')
plt.semilogy(delta*gfmBnd, '-', label='GFM bound', c=getLastPlotCol())
plt.semilogy([delta*gamma**k for k in range(nIter+1)],
             ':', label='Iteration matrix', c=getLastPlotCol())
plt.semilogy([delta*rho**k for k in range(nIter+1)],
             '-.', label='Spectral radius', c=getLastPlotCol())
# Plot interface error
plt.figure(figI)
plt.semilogy(deltaI*errI, 'o--', label='Iteration error')
plt.semilogy(deltaI*gfmBndI, '-', label='Interface approximation',
             c=getLastPlotCol())

for i, f in enumerate([fig, figI]):
    plt.figure(f)
    plt.ylim(1e-16, 100)
    setFig('Iteration', 'Error vs. fine solution',
           fileName=f'fig_PFASST_{i}.pdf')
