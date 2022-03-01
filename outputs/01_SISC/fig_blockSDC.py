#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:27:24 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from gfm.base import GFMSolver
from gfm.util import setFig, getLastPlotCol

s = GFMSolver(lam=1j, u0=1, dt=0.1*np.pi, L=10)

fineMethod = 'COLLOCATION'
deltaMethod = 'BE'
nodesType = 'LEGENDRE'
qType = 'LOBATTO'
iType = 'RAND'

s.setFineLevel(M=10, method=fineMethod,
    nodes=nodesType, qType=qType)
s.setPhiDelta(deltaMethod)

algos = ['Jacobi', 'GaussSeidel']
styles = ['s--', 'o--']
labels = ['Block Jacobi', 'Block Gauss-Seidel']
nIter = 20

err = np.zeros((len(algos), nIter+1))
gfmBnd = np.ones((len(algos), nIter+1))
errI = np.zeros((len(algos), nIter+1))
gfmBndI = np.ones((len(algos), nIter+1))

fig = plt.figure()
figI = plt.figure()

for i, algo in enumerate(algos):

    # Initialization
    uAlgo = s.getInitU(iType)
    delta = np.max(s.getErr(uAlgo))
    deltaI = np.max(s.getErr(uAlgo, interface=True))
    err[i, 0] = 1
    errI[i, 0] = 1

    # Error bound computation
    errEstimate = s.errBoundFunc(algo)
    errEstimateI = s.errBoundFunc(algo, iOnly=True)
    R, _ = s.iterationMatrix(algo)
    gamma = np.linalg.norm(R, ord=np.inf)

    # Iterations
    for k in range(nIter):
        s.iterate(algo, uAlgo)
        err[i, k+1] = s.getErr(uAlgo)[-1]
        errI[i, k+1] = s.getErr(uAlgo, interface=True)[-1]
        gfmBnd[i, k+1] = errEstimate(s.L, k+1)
        gfmBndI[i, k+1] = errEstimateI(s.L, k+1)

    # Plot volume error
    plt.figure(fig)
    plt.semilogy(delta*err[i], styles[i], label=labels[i])
    plt.semilogy(delta*gfmBnd[i], '-', c=getLastPlotCol())
    plt.semilogy([delta*gamma**k for k in range(nIter+1)],
                 ':', c=getLastPlotCol())
    # Plot interface error
    plt.figure(figI)
    plt.semilogy(deltaI*errI[i], styles[i], label=labels[i])
    plt.semilogy(deltaI*gfmBndI[i], '-', c=getLastPlotCol())

for i, f in enumerate([fig, figI]):
    plt.figure(f)
    plt.ylim(1e-16, 10)
    setFig('Iteration', 'Error vs. fine solution',
           fileName=f'fig_blockSDC_{i}.pdf')
