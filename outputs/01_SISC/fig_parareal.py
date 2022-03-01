#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 18:54:13 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from gfm.base import GFMSolver
from gfm.util import setFig

M = 1
fineMethod = 'RK4'
deltaMethod = 'FE'
nodesType = 'EQUID'
qType = 'RADAU-II'

lams = [1j, -1, 4j, -4]
nIter = 10

err = np.zeros((len(lams), nIter+1))
gfmBnd = np.ones((len(lams), nIter+1))
gfmBnd2 = np.ones((len(lams), nIter+1))

for i, lam in enumerate(lams):

    s = GFMSolver(lam, u0=1, dt=0.2*np.pi, L=10)
    s.setFineLevel(
        M=1, method=fineMethod, nodes=nodesType, qType=qType,
        nStepPerNode=10)
    s.setPhiDelta(deltaMethod, nStepPerNode=5)
    uFine = s.getU('Fine')

    algo = 'Parareal'
    # Initialization
    uAlgo = s.getInitU(iType='RAND')
    delta = np.max(s.getErr(uAlgo))
    err[i, 0] = delta
    # Error bound computation
    errEstimate = s.errBoundFunc(algo)
    errEstimate2 = s.errBoundFunc(algo, trick=True)
    R, _ = s.iterationMatrix(algo)
    gamma = np.linalg.norm(R, ord=np.inf)

    plt.figure()
    # Iterations
    for k in range(nIter):
        s.iterate(algo, uAlgo)
        err[i, k+1] = s.getErr(uAlgo)[-1]
        gfmBnd[i, k+1] = errEstimate(s.L, k+1)
        gfmBnd2[i, k+1] = errEstimate2(s.L, k+1)
    # Plot
    plt.semilogy(err[i], '--', label='Iteration error')
    plt.semilogy(delta*gfmBnd2[i], 's-', label='Original bound')
    plt.semilogy(delta*gfmBnd[i], 'o-', label='New bound')
    if i < 2:
        plt.semilogy([delta*gamma**k for k in range(nIter+1)], ':',
                     label='Norm of iteration matrix')
        plt.ylim(1e-13, 10)
    if i == 2:
        plt.ylim(1e-2, 1e4)
    if i == 3:
        plt.ylim(1e-13, 10)
    setFig('Iteration', 'Error vs. fine solution',
           fileName=f'fig_Parareal_{i}.pdf')