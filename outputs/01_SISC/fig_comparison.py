#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:44:29 2022

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from gfm.base import GFMSolver
from gfm.util import setFig, getLastPlotCol

s = GFMSolver(lam=2j-0.2, u0=1, dt=0.2*np.pi, L=10)

M = 5
MCoarse = 3

nIter = 20
iApprox=True
initType='RAND'

params = [
    dict(fineMethod='COLLOCATION',
         deltaMethod='BE',
         nodesType='LEGENDRE',
         qType='LOBATTO'),
    dict(fineMethod='RK4',
         deltaMethod='RK2',
         nodesType='EQUID',
         qType='LOBATTO')]

iFig = 1
for param in params:

    # Set GFM solver
    s.setFineLevel(
        M=M, method=param['fineMethod'],
        nodes=param['nodesType'], qType=param['qType'])
    s.setPhiDelta(method=param['deltaMethod'])
    s.setCoarseLevel(MCoarse, nodes=param['nodesType'], qType=param['qType'])
    s.setPhiDeltaCoarse(method=param['deltaMethod'])

    # Comparison between one-level iterative methods
    algos = ['GaussSeidel', 'Jacobi', 'Parareal']
    symbols = ['s', '>', 'o', 'p']

    err = np.zeros((len(algos), nIter+1))
    gfmBnd = np.ones((len(algos), nIter+1))

    plt.figure()
    for i, algo in enumerate(algos):
        # Initialization
        uAlgo = s.getInitU(iType=initType)
        delta = np.max(s.getErr(uAlgo))
        err[i, 0] = delta

        # Error bound computation
        errEstimate = s.errBoundFunc(algo, iOnly=iApprox)
        R, _ = s.iterationMatrix(algo)
        gamma = np.linalg.norm(R, ord=np.inf)
        # Iterations
        for k in range(nIter):
            s.iterate(algo, uAlgo)
            err[i, k+1] = s.getErr(uAlgo)[-1]
            gfmBnd[i, k+1] = errEstimate(s.L, k+1)
        # Plot
        plt.semilogy(err[i], symbols[i]+'--', label=f'{algo}')
        plt.semilogy(delta*gfmBnd[i], symbols[i]+'-', c=getLastPlotCol(), lw=1)
    plt.ylim(1e-16, 1e3)
    setFig('Iteration', 'Error vs. fine solution',
           fileName=f'fig_comparison_{iFig}.pdf')
    iFig += 1

    # %% Comparison between two-level iterative methods

    algos = ['STMG', 'PFASST', 'ATMG', 'TFASST']
    labels = ['TMG', 'PFASST', 'TMG$_c$', 'TMG$_f$']
    styles = ['o-', '>-', 'o--', '>--']
    symbols = ['s', '>', 'o', '^']

    err = np.zeros((len(algos), nIter+1))
    gfmBnd = np.ones((len(algos), nIter+1))

    plt.figure()
    for i, algo in enumerate(algos):

        # Initialization
        uAlgo = s.getInitU(iType=initType)
        delta = np.max(s.getErr(uAlgo))
        err[i, 0] = delta

        # Error bound computation
        errEstimate = s.errBoundFunc(algo, iOnly=iApprox)

        # Iterations
        for k in range(nIter):
            s.iterate(algo, uAlgo)
            err[i, k+1] = s.getErr(uAlgo)[-1]
            gfmBnd[i, k+1] = errEstimate(s.L, k+1)
        uAlgo = s.getU(uAlgo)

        plt.semilogy(err[i], symbols[i]+'--', label=labels[i])
        plt.semilogy(delta*gfmBnd[i], symbols[i]+'-', c=getLastPlotCol(), lw=1)

    plt.ylim(1e-16, 1e3)
    setFig('Iteration', 'Error vs. fine solution',
           fileName=f'fig_comparison_{iFig}.pdf')
    iFig += 1

    t, uExact = s.getU('Exact', times=True)
    uFine = s.getU('Fine')
    uDelta = s.getU('Delta')

    tCoarse, uExactCoarse = s.getU('ExactCoarse', times=True)
    uCoarse = s.getU('Coarse')
    uDeltaCoarse = s.getU('DeltaCoarse')

    plt.figure()
    plt.semilogy(# Fine level solution
        t, np.abs(uFine-uExact), 'o-', label='Fine')
    plt.semilogy(# Delta solution on fine level
        t, np.abs(uDelta-uExact), '--', label=f'Delta, {param["deltaMethod"]}')
    plt.semilogy(# Coarse level solution
        tCoarse, np.abs(uCoarse-uExactCoarse), 's-', label='Coarse level')
    plt.semilogy(# Delta coarse level solution
        tCoarse, np.abs(uDeltaCoarse-uExactCoarse), '-.',
        label=f'Delta coarse, {param["deltaMethod"]}')
    plt.vlines(
        [0] + list(s.times[:, -1]), *plt.ylim(), colors='gray', linewidth=0.2)
    setFig('Time', 'Error vs exact solution', grid=False)

    print(f'Fine error : {np.abs(uFine-uExact).max():1.2e}')
    print(f'Fine Delta error : {np.abs(uDelta-uExact).max():1.2e}')
    print(f'Coarse error : {np.abs(uCoarse-uExactCoarse).max():1.2e}')
    print(f'Coarse Delta error : {np.abs(uDeltaCoarse-uExactCoarse).max():1.2e}')
