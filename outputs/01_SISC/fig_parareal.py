#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Figure 3 and 4
"""
import numpy as np
import matplotlib.pyplot as plt

from gfm.base import GFMSolver
from gfm.util import setFig

fineMethod = 'RK4'
deltaMethod = 'BE'
nodesType = 'EQUID'
qType = 'RADAU-II'

lams = [1j, -1, 4j, -4]
nIter = 10

err = np.zeros((len(lams), nIter+1))
gfmBnd = np.ones((len(lams), nIter+1))
gfmBnd2 = np.ones((len(lams), nIter+1))

nPtsPerWavelength = 10
plotSol = False

nPtsCoarse = int(round(nPtsPerWavelength/5))

for i, lam in enumerate(lams):

    s = GFMSolver(lam, u0=1, dt=0.2*np.pi, L=10)
    s.setFineLevel(
        M=nPtsPerWavelength if plotSol else 1,
        method=fineMethod, nodes=nodesType, qType=qType,
        nStepPerNode=1 if plotSol else nPtsPerWavelength)
    s.setPhiDelta(
        deltaMethod, nStepPerNode=nPtsCoarse)
    if plotSol:
        s.setCoarseLevel(M=nPtsCoarse, qType=qType)
        s.setPhiDeltaCoarse(deltaMethod)

    t, uExact = s.getU('Exact', times=True)
    uDelta = s.getU('Delta')
    if plotSol:
        tCoarse, uCoarse = s.getU('DeltaCoarse', times=True)
    uFine = s.getU('Fine')
    errFine = np.max(np.abs(uExact-uFine))
    errCoarse = np.max(np.abs(uExact-uDelta))
    print(f'ErrFine : {errFine:1.2e}')
    print(f'ErrCoarse : {errCoarse:1.2e}')

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
    # plt.title(f'lam={lam}, {nPtsPerWavelength} pts per wavelength')
    # Iterations
    for k in range(nIter):
        s.iterate(algo, uAlgo)
        err[i, k+1] = s.getErr(uAlgo)[-1]
        gfmBnd[i, k+1] = errEstimate(s.L, k+1)
        gfmBnd2[i, k+1] = errEstimate2(s.L, k+1)
    # Plot
    plt.semilogy(err[i], '--', label='Iteration error')
    plt.semilogy(delta*gfmBnd2[i], '^-', label='Original bound')
    plt.semilogy(delta*gfmBnd[i], 'o-', label='GFM bound')
    if True:
        plt.semilogy([delta*gamma**k for k in range(nIter+1)], ':',
                      label='Norm of iteration matrix')
    if i == 2:
        plt.ylim(1e-2, 1e4)
        # textArgs = dict(
        #     bbox=dict(boxstyle="round",
        #               ec=(0.5, 0.5, 0.5),
        #               fc=(0.8, 0.8, 0.8)))
        # plt.text(0, 0.02, 'Max. abs. err. fine: $8.35e^{-4}$', **textArgs)
    else:
        plt.ylim(1e-13, 10)
    # plt.hlines([errFine, errCoarse], 0, nIter,
    #            colors='gray', linestyles='--', linewidth=1.5)
    setFig('Iteration', 'Error vs. fine solution',
           fileName=f'fig_Parareal_{i}.pdf')

    if plotSol:
        plt.close()
        plt.figure()
        plt.title(f'lam={lam}, {nPtsPerWavelength} pts per wavelength')
        plt.plot(t, uFine.real, '-', label='fine')
        plt.plot(tCoarse, uCoarse.real, '-', label='coarse')
        plt.legend(loc='upper right')
        plt.xlabel('time')
        plt.tight_layout()
