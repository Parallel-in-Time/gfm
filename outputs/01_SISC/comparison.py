#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:28:49 2021

@author: telu
"""
import numpy as np
import matplotlib.pyplot as plt

from qmatrix import BlockSDC

# Algorithms to be compared
algos = ['GaussSeidel', 'CoarseGaussSeidel']
algos = ['Jacobi', 'CoarseJacobi']
algos = ['Parareal', 'STMG', 'Jacobi']
algos = ['Jacobi', 'GaussSeidel']
# algos = ['STMG']
algoParams = dict(
    # STMG parameters
    nPreRelax=1, nPostRelax=0, omega=1,
    # PFASST parameters
    ideal=False,
    # Parareal & MGRIT parameters
    twoLevelCoarse=True, sdcCoarse=True)
normFCF = False

# Block parameters
args = dict(
    distr='LEGENDRE',
    quadType='RADAU-II',
    sweepType='BE',
    initCond='RAND',
    cPoints=slice(None, None, 3))
BlockSDC.NORM_TYPE = 'L_INF'
BlockSDC.APPROX_EXACT = False
BlockSDC.APPROX_EXACT_COARSE = False

dt = 0.1*np.pi
tEnd = 1*np.pi
L = int(tEnd/dt+0.5)
# tEnd = L*dt
# M = np.linspace(0, 1, num=256+1)[1:].tolist()
M = 5
nSweep = range(30)

try:
    tau = dt/len(M)
except TypeError:
    tau = dt/M

# Problem parameters
lLam = [-1]
errFine = []
errCoarse = []
u0 = 1

err = np.zeros((len(lLam), len(algos), len(nSweep), L))
res = err.copy()
bnd = err.copy()
eTh = err.copy()
for i, lam in enumerate(lLam):

    solver = BlockSDC(lam, u0, tEnd, L, M, **args)
    uExact = solver.uExact
    uExp = np.exp(lam * solver.t)
    uCoarse = solver.uCoarse

    errF = np.max(np.abs((np.exp(lam * solver.t)-uExact)))
    errFine.append(errF)
    errC = np.max(np.abs((np.exp(lam * solver.t)-uCoarse)))
    errCoarse.append(errC)

    norm = solver.normVec
    print('-'*80)
    print(f'Experiment for lam={lam}')
    if False:
        print(' -- GFM coefficients')
        for key, val in solver.gfmCoeff.items():
            print('     --', key)
            print(val)
    print(f'Fine blocks, M={len(solver.nodes)}')
    print(f' -- max fine error : {errF}')
    print(f'Coarse blocks, M={len(solver.nodesTilde)}')
    print(f' -- max coarse error : {errC}')
    ratio = len(solver.nodes)/len(solver.nodesTilde)
    print(f'Pipelined speedup (L={L}, r={ratio}): ')
    print(f' -- k=1 : {L/(L/ratio+1)} ({1/(L/ratio+1):1.2}%)')
    print(f' -- k=2 : {L/(L/ratio+2)} ({1/(L/ratio+2):1.2}%)')
    print(f' -- k=3 : {L/(L/ratio+3)} ({1/(L/ratio+3):1.2}%)')

    solver.gfmCoeff['GaussSeidel']['gamma'] = 0.02980221
    solver.gfmCoeff['GaussSeidel']['beta'] = 0.73871082
    solver.gfmCoeff['Jacobi']['gamma'] = solver.gfmCoeff['GaussSeidel']['gamma']
    solver.gfmCoeff['Jacobi']['alpha'] = solver.gfmCoeff['GaussSeidel']['beta']

    for j, sol in enumerate(algos):
        for k in nSweep:
            uNum = solver.run[sol](k, **algoParams)
            err[i, j, k, :] = [norm((uNum[l]-uExact[l])[:-1]) for l in range(L)]
            bnd[i, j, k, :] = [solver.errBound[sol](n, k) for n in range(1, L+1)]
            res[i, j, k, :] = [norm(r) for r in solver.residuum(uNum)]
            eTh[i, j, k, :] = [norm(uNum[l]-uExp[l]) for l in range(L)]

ImQDelta = solver.ImQDelta
Q, H = solver.Q, solver.H
QDelta = solver.QDelta
eM = np.zeros(M)
eM[-1] = 1
e = np.ones(M)

R = eM[None, :] @ np.linalg.solve(ImQDelta, lam*dt*(Q-QDelta))
P = eM[None, :] @ np.linalg.solve(ImQDelta, e)

# %%
var = err
sy = ['o', 's', '<', 'p', '*']
ls = ['-', '--', ':', '-.']

plt.figure()
plt.gcf().set_size_inches(10, 5)
for i, lam in enumerate(lLam):
    plt.subplot(1, len(lLam), i+1)
    for j, sol in enumerate(algos):
        if normFCF and sol == 'MGRIT':
            nIterMax = len(nSweep)
            y = var[i, j, :nIterMax//2, -1]
            x = np.arange(y.size)*2
        else:
            x = nSweep
            y = var[i, j, :, -1]
            b = bnd[i, j, :, -1]
        plt.semilogy(x, y, sy[j]+'-', label=sol)
        c = plt.gca().get_lines()[-1].get_color()
        plt.semilogy(x, b, '--', c=c)
    plt.hlines(errFine[i], x[0], x[-1], colors='gray', linestyles=':')
    plt.hlines(errCoarse[i], x[0], x[-1], colors='brown', linestyles=':')

    plt.title(rf'$\lambda={lam}$')
    plt.legend()
    plt.grid(True)
    plt.xlabel('$k$')
    plt.ylabel('$e^k_{L}$')
    # plt.ylabel('$r^k_{L}$')
    plt.ylim(ymin=1e-17,ymax=10)
    plt.tight_layout()

# %%
if False:
    var = eTh
    plt.figure()
    plt.gcf().set_size_inches(10, 5)
    for i, lam in enumerate(lLam):
        plt.subplot(1, len(lLam), i+1)
        for j, sol in enumerate(algos):
            if normFCF and sol == 'MGRIT':
                nIterMax = len(nSweep)
                y = var[i, j, :nIterMax//2, -1]
                x = np.arange(y.size)*2
            else:
                x = nSweep
                y = var[i, j, :, -1]
                b = bnd[i, j, :, -1]
            plt.semilogy(x, y, sy[j]+'-', label=sol)
        plt.hlines(errFine[i], x[0], x[-1], colors='gray', linestyles='--')
        plt.hlines(errCoarse[i], x[0], x[-1], colors='brown', linestyles='--')

        plt.title(rf'$\lambda={lam}$')
        plt.legend()
        plt.grid(True)
        plt.xlabel('$k$')
        plt.ylabel('Err. with exact')
        plt.tight_layout()