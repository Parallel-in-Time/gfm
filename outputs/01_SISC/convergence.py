#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 11:45:47 2021

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from gfm.base import GFMSolver
from gfm.util import setFig, getLastPlotCol

s = GFMSolver(lam=1j, u0=1, dt=0.1*np.pi, L=10)

MCoarse = 3
ratio = 2
fineMethod = 'RK4'
deltaMethod = 'FE'
nodesType = 'EQUID'
qType = 'LOBATTO'

s.setFineLevel(
    M=ratio*(MCoarse-1)+1, method=fineMethod,
    nodes=nodesType, qType=qType)
s.setPhiDelta(deltaMethod)
s.setCoarseLevel(MCoarse, nodes=nodesType, qType=qType)
s.setPhiDeltaCoarse(deltaMethod)

uFine = s.getU('Fine')
interface = False  # Look at error/bound in interface only


# %% Individual algorithm, error VS time for each iterations
if False:
    algo = 'Parareal'
    nIter = 5

    err = np.zeros((nIter+1, s.L))

    # Initialization
    uAlgo = s.getInitU(iType='ZERO')
    err[0, :] = s.getErr(uAlgo, interface=interface)
    # Iterations
    for k in range(nIter):
        s.iterate(algo, uAlgo)
        err[k+1, :] = s.getErr(uAlgo, interface=interface)
    # Plot
    plt.figure()
    for k in range(nIter+1):
        plt.semilogy(s.times[:, -1], err[k], label=f'k={k}')
    setFig('Time', 'Error', algo)

# %% Multiple algorithm, error VS iteration for each algo for last block
# 1-level algorithms
if False:
    algos = ['Jacobi', 'GaussSeidel', 'Parareal']
    nIter = 20

    err = np.zeros((len(algos), nIter+1))
    gfmBnd = np.ones((len(algos), nIter+1))

    plt.figure()
    for i, algo in enumerate(algos):
        # Initialization
        uAlgo = s.getInitU(iType='RAND')
        delta = np.max(s.getErr(uAlgo))
        err[i, 0] = delta
        # Error bound computation
        errEstimate = s.errBoundFunc(algo, iOnly=interface)
        R, _ = s.iterationMatrix(algo)
        gamma = np.linalg.norm(R, ord=np.inf)
        # Iterations
        for k in range(nIter):
            s.iterate(algo, uAlgo)
            err[i, k+1] = s.getErr(uAlgo, interface=interface)[-1]
            gfmBnd[i, k+1] = errEstimate(s.L, k+1)
        # Plot
        plt.semilogy(err[i], 'o--', label=f'{algo}')
        plt.semilogy(delta*gfmBnd[i], '-', c=getLastPlotCol())
        if not interface:
            plt.semilogy([delta*gamma**k for k in range(nIter+1)],
                         ':', c=getLastPlotCol())
    plt.ylim(1e-16, 10)
    setFig('Iteration', 'Error')

# %% Multiple algorithm, error VS iteration for each algo for last block
# 2-level algorithms
if True:
    algos = {
        'STMG' : {'omega': 0.99, 'approxCoarse': False, 'approxSmoother': False},
        'PFASST': {'approxCoarse': True, 'approxSmoother': True},
        'ATMG': {'approxCoarse': True, 'approxSmoother': False},
        'TFASST': {'approxCoarse': False, 'approxSmoother': True}}

    nIter = 10

    err = np.zeros((len(algos), nIter+1))
    gfmBnd = np.ones((len(algos), nIter+1))

    plt.figure()
    for i, (algo, params) in enumerate(algos.items()):
        # Initialization
        uAlgo = s.getInitU(iType='ZERO')
        delta = np.max(s.getErr(uAlgo, interface=interface))
        err[i, 0] = 1
        # Error bound computation
        errEstimate = s.errBoundFunc('TwoGrid', iOnly=interface, **params)
        R, _ = s.iterationMatrix('TwoGrid', **params)
        gamma = np.linalg.norm(R, ord=np.inf)
        # Iterations
        for k in range(nIter):
            s.iterate('TwoGrid', uAlgo, **params)
            err[i, k+1] = s.getErr(uAlgo, interface=interface)[-1]
            gfmBnd[i, k+1] = errEstimate(s.L, k+1)
        # Plot
        plt.semilogy(delta*err[i], 'o--', label=f'{algo}')
        plt.semilogy(delta*gfmBnd[i], '-', c=getLastPlotCol())
        if not interface:
            plt.semilogy([delta*gamma**k for k in range(nIter+1)],
                         ':', c=getLastPlotCol(), lw=1)
        break
    plt.ylim(1e-16, 10)
    setFig('Iteration', 'Error')