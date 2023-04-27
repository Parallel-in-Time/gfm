#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Additional figures for numerical experiments with (S)TMG
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

s.setFineLevel(
    M=ratio*(MCoarse-1)+1, method=fineMethod,
    nodes=nodesType, qType=qType)
s.setPhiDelta(deltaMethod)
s.setCoarseLevel(MCoarse, nodes=nodesType, qType=qType)
s.setPhiDeltaCoarse(deltaMethod)

nIter = 10

lOmega = [1, 0.95]
styles = ['o', '>']

err = np.ones((2, len(lOmega), nIter+1))
gfmBnd = np.ones((2, len(lOmega), nIter+1))

fig = plt.figure()
figI = plt.figure()

for i, omega in enumerate(lOmega):

    # Initialization
    uAlgo = s.getInitU(iType=iType)
    delta = np.max(s.getErr(uAlgo))
    deltaI = np.max(s.getErr(uAlgo, interface=True))

    # Error bound computation
    errEstimate = s.errBoundFunc('STMG', omega=omega)
    errEstimateI = s.errBoundFunc('STMG', iOnly=True, omega=omega)
    R, _ = s.iterationMatrix('STMG', omega=omega)
    gamma = np.linalg.norm(R, ord=np.inf)

    # Iterations
    for k in range(nIter):
        s.iterate('STMG', uAlgo, omega=omega)
        err[0, i, k+1] = s.getErr(uAlgo)[-1]
        err[1, i, k+1] = s.getErr(uAlgo, interface=True)[-1]
        gfmBnd[0, i, k+1] = errEstimate(s.L, k+1)
        gfmBnd[1, i, k+1] = errEstimateI(s.L, k+1)
    # Plot
    plt.figure(fig)
    plt.semilogy(delta*err[0, i], styles[i]+'--', label=rf'$\omega={omega}$')
    plt.semilogy(delta*gfmBnd[0, i], '-', c=getLastPlotCol())
    plt.semilogy([delta*gamma**k for k in range(nIter+1)],
                 ':', c=getLastPlotCol(), lw=1)
    plt.figure(figI)
    plt.semilogy(delta*err[1, i], styles[i]+'--', label=rf'$\omega={omega}$')
    plt.semilogy(delta*gfmBnd[1, i], '-', c=getLastPlotCol())

for i, f in enumerate([fig, figI]):
    plt.figure(f)
    plt.ylim(1e-16, 10)
    setFig('Iteration', 'Error vs. fine solution',
           fileName=f'fig_STMG_{i}.pdf')
