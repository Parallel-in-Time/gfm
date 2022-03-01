#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:50:20 2022

@author: telu
"""
import numpy as np
from scipy.special import binom, factorial

def pbi(n, k, alpha=None, beta=None, gamma=None, trick=False, reTol=1e-4,
        **kwargs):

    def neglect(one, *others):
        others = [val for val in others if val is not None]
        if one is None:
            return True
        elif one == 0:
            return True
        elif one < reTol*min(others):
            return True
        else:
            return False

    if n == 0:
        return 0
    if k == 0:
        return 1
    if neglect(beta, alpha, gamma) and neglect(alpha, beta, gamma):
        return gamma**k
    if neglect(gamma, alpha, beta):
        # PBI-1 (Parareal, Simplified STMG with 1 smoothing cycle)
        if trick:
            beta = max(1, beta)
            prod = np.prod([n-1-l for l in range(k)])
            return alpha**k / factorial(k) * beta**(n-1-k) * prod
        else:
            return alpha**k * sum(binom(i+k-1, i)*beta**i
                                  for i in range(n-k))
    elif neglect(beta, alpha, gamma):
        # PBI-2 (Block Jacobi SDC / Jacobi smoothing)
        r = alpha/gamma
        return gamma**k * sum(binom(k, i)*r**i
                              for i in range(min(n-1, k)+1))
    elif neglect(alpha, beta, gamma):
        # PBI-3 (Block Gauss-Seidel SDC)
        if trick:
            beta = max(1, beta)
            prod = np.prod([n+l for l in range(k)])
            return gamma**k / factorial(k) * beta**(n-1) * prod
        else:
            return gamma**k * sum(binom(i+k-1, i)*beta**i
                                  for i in range(n))
    else:
        # Full PBI (PFASST, General STMG with 1 smoothing cycle)
        r = alpha/gamma
        s = sum(sum(binom(k, i) * binom(l+k-1, l) * r**i * beta**l
                    for l in range(n-i))
                for i in range(min(n-1, k)+1))
        return gamma**k * s

def pbiRec(n, k, alpha=None, beta=None, gamma=None):
    if n == 0:
        return 0
    if k == 0:
        return 1
    if gamma in [None, 0]:
        return alpha * pbiRec(n-1, k-1, alpha, beta, gamma) + \
            beta * pbiRec(n-1, k, alpha, beta, gamma)
    elif beta in [None, 0]:
        return alpha * pbiRec(n-1, k-1, alpha, beta, gamma) + \
            gamma * pbiRec(n, k-1, alpha, beta, gamma)
    elif alpha in [None, 0]:
        return beta * pbiRec(n-1, k, alpha, beta, gamma) + \
            gamma * pbiRec(n, k-1, alpha, beta, gamma)
    else:
        return alpha * pbiRec(n-1, k-1, alpha, beta, gamma) + \
            beta * pbiRec(n-1, k, alpha, beta, gamma) + \
            gamma * pbiRec(n, k-1, alpha, beta, gamma)

def sdc(n, k, gamma, beta):
    Sk = beta * sum(gamma**l for l in range(k))
    return gamma**k * sum(Sk**i for i in range(n+1))


if __name__ == '__main__':
    # Testing PBI-1
    t1 = np.array(
        [pbi(k, 10, alpha=0.1, beta=1) for k in range(10)])
    t2 = np.array(
        [pbi(k, 10, alpha=0.1, beta=1, trick=True) for k in range(10)])
    assert np.linalg.norm(t1-t2, np.inf) < 1e-15
    t1Rec = np.array(
        [pbiRec(k, 10, alpha=0.1, beta=1) for k in range(10)])
    assert np.linalg.norm(t1-t1Rec, np.inf) < 1e-15

    # Testing PBI-2
    t1 = np.array([pbi(k, 10, gamma=0.1, alpha=1) for k in range(10)])
    t1Rec = np.array([pbiRec(k, 10, gamma=0.1, alpha=1) for k in range(10)])
    assert np.linalg.norm(t1-t1Rec, np.inf) < 1e-15

    # Testing PBI-3
    t1 = np.array(
        [pbi(k, 10, gamma=0.1, beta=1) for k in range(10)])
    t2 = np.array(
        [pbi(k, 10, gamma=0.1, beta=1, trick=True) for k in range(10)])
    assert np.linalg.norm(t1-t2, np.inf) < 1e-15
    t1Rec = np.array(
        [pbiRec(k, 10, gamma=0.1, beta=1) for k in range(10)])
    assert np.linalg.norm(t1-t1Rec, np.inf) < 1e-15

    # Testing PBI-Full
    t1 = np.array(
        [pbi(k, 10, gamma=0.1, alpha=1, beta=0.1) for k in range(10)])
    t1Rec = np.array(
        [pbiRec(k, 10, gamma=0.1, alpha=1, beta=0.1) for k in range(10)])
    assert np.linalg.norm(t1-t1Rec, np.inf) < 1e-15