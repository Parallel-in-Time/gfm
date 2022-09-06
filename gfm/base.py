#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:04:06 2021

@author: cpf5546
"""
import numpy as np

# Pythag library can be found at https://gitlab.com/tlunet/pythag
from pythag.applications import LagrangeApproximation

# Internal import
import gfm.analytic as gfma

STABILITY_FUNCTION_RK = {
    'BE': lambda z: (1-z)**(-1),
    'FE': lambda z: 1+z,
    'TRAP': lambda z: (1+z/2)/(1-z/2),
    'RK2': lambda z: 1+z+z**2/2,
    'RK4': lambda z: 1+z+z**2/2+z**3/6+z**4/24,
    'EXACT': lambda z: np.exp(z)}
RK_METHODS = STABILITY_FUNCTION_RK.keys()

class GFMSolver(object):

    def __init__(self, lam, u0, dt, L):
        self.lam = lam
        self.u0 = u0
        self.dt = dt
        self.L = L

        self.numType = np.array(1.*lam*u0).dtype

        self.phi = None
        self.chi = None
        self.f0 = None
        self.nodes = None
        self.method = None
        self.form = None

        self.phiDelta = None
        self.methodDelta = None

        self.phiCoarse = None
        self.chiCoarse = None
        self.nodesCoarse = None
        self.TFtoC = None
        self.TCtoF = None
        self.deltaChi = None

        self.phiDeltaCoarse = None
        self.methodDeltaCoarse = None

        # Storage variable for sequential solutions
        self._uFine = None
        self._uDelta = None
        self._uCoarse = None
        self._uDeltaCoarse = None

    def copy(self):
        """Bof bof ..."""
        return GFMSolver(self.lam, self.u0, self.dt, self.L)

    @property
    def tEnd(self):
        return self.dt*self.L

    # -------------------------------------------------------------------------
    # Methods building block operators on fine and coarse level
    # -------------------------------------------------------------------------

    def setFineLevel(self, M, method, **kwargs):
        # Build phi and chi operators
        self.phi, self.chi, self.nodes, self.form = getBlockOperators(
            self.lam*self.dt, M, method, **kwargs)
        self.method = method
        # Build first component of the global system right hand side
        # ! without the chi multiplication (done during computations) !
        self.f0 = np.ones(M) * self.u0
        # Reset solution storage variables
        self._uFine = None

    def setPhiDelta(self, method, **kwargs):
        # Build phiDelta
        self.phiDelta = getBlockOperators(
            self.lam*self.dt, self.M, method,
            nodes=self.nodes, form=self.form, **kwargs)[0]
        # Store used method
        self.methodDelta = method
        # Reset solution storage variables
        self._uDelta = None

    def setCoarseLevel(self, M, **kwargs):
        # Build phi and chi operators for coarse level
        self.phiCoarse, self.chiCoarse, self.nodesCoarse = \
            getBlockOperators(
            self.lam*self.dt, M, self.method, form=self.form, **kwargs)[:3]
        # Build transfer operators
        self.TFtoC, self.TCtoF = getTransferOperators(
            self.nodes, self.nodesCoarse)
        # Compute deltaChi operator
        self.deltaChi = self.TFtoC @ self.chi - self.chiCoarse @ self.TFtoC
        # Reset solution storage variables
        self._uCoarse = None

    def setPhiDeltaCoarse(self, method, **kwargs):
        # Build phiDelta on coarse level
        self.phiDeltaCoarse = getBlockOperators(
            self.lam*self.dt, self.MCoarse, method,
            nodes=self.nodesCoarse, form=self.form, **kwargs)[0]
        # Store used method
        self.methodDelta = method
        # Reset solution storage variables
        self._uDeltaCoarse = None

    # -------------------------------------------------------------------------
    # Utilitary attributes and methods
    # -------------------------------------------------------------------------

    @property
    def M(self):
        try:
            return len(self.nodes)
        except TypeError:
            return 0

    @property
    def MCoarse(self):
        try:
            return len(self.nodesCoarse)
        except TypeError:
            return 0

    @property
    def times(self):
        return np.array([[(i+tau)*self.dt for tau in self.nodes]
                         for i in range(self.L)])

    @property
    def timesCoarse(self):
        return np.array([[(i+tau)*self.dt for tau in self.nodesCoarse]
                         for i in range(self.L)])

    @property
    def noDeltaChi(self):
        return np.linalg.norm(self.deltaChi, ord=np.inf) < 1e-14

    def getInitU(self, iType='ZERO'):
        """
        Generate an initial iterate for the block soluton

        Parameters
        ----------
        iType : str, optional
            Chosen initialization type, choices are :

            - 'ZERO' : null vector
            - 'U0' : the initial solution copied along all times
            - 'RAND' : a random solution with given seed=1990
            - 'U_DELTA' : the solution computed with delta phi operator.
            - 'U_COARSE' : propagate the coarse operator, and interpolate.
            - 'U_COARSE_DELTA' : propagate the coarse delta operator, and interpolate.

            The default is 'ZERO'.

        Returns
        -------
        u : np.ndarray
            The initial solution vector, shape (L, M).
        """
        u = np.zeros((self.L, self.M), dtype=self.numType)
        if iType == 'ZERO':
            pass
        elif iType== 'U0':
            u[:] = self.u0
        elif iType == 'RAND':
            np.random.seed(1990)
            u[:] = np.random.rand(*u.shape)
            if u.dtype == complex:
                u[:] += np.random.rand(*u.shape)*1j
        elif iType == 'U_DELTA':
            u[:] = self.uDelta
        elif iType == 'U_COARSE':
            u[:] = (self.TCtoF @ self.uCoarse.T).T
        elif iType == 'U_DELTA_COARSE':
            u[:] = (self.TCtoF @ self.uDeltaCoarse.T).T
        else:
            raise ValueError(f'wrong initialization type iType={self.iType}')
        return u

    def getU(self, u, times=False, coarse=False, includeInitSol=True):
        if isinstance(u, str):
            if 'Coarse' in u:
                coarse = True
            u = self.__getattribute__(f'u{u}')
        uInit = [self.u0] if includeInitSol else []
        u = np.array(uInit + list(u.ravel()))
        if times:
            tInit = [0] if includeInitSol else []
            if coarse:
                t = np.array(tInit + list(self.timesCoarse.ravel()))
            else:
                t = np.array(tInit + list(self.times.ravel()))
            u = t, u
        return u

    def getErr(self, u, ref='Fine', interface=False):
        # Set norm
        norm = lambda x: np.linalg.norm(x, ord=np.inf, axis=1)
        # Set reference solution (can also be given as vector in ref argument)
        if ref == 'Fine':
            ref = self.uFine
        elif ref == 'Exact':
            ref = self.uExact
        # Compute and return error
        diff = u - ref
        if interface:
            diff = diff[:, -1:]
        return norm(diff)

    # -------------------------------------------------------------------------
    # Sequential solutions as properties
    # -------------------------------------------------------------------------

    @property
    def uExact(self):
        return np.exp(self.lam*self.times)

    @property
    def uExactCoarse(self):
        return np.exp(self.lam*self.timesCoarse)

    def _uNumSeq(self, sType):
        name = f'_u{sType}'
        if getattr(self, name) is None:

            # Select block size and operators
            if 'Coarse' in name:
                M = self.MCoarse
                chi = self.chiCoarse
                rhs0 = self.TFtoC @ self.chi @ self.f0
            else:
                M = self.M
                chi = self.chi
                rhs0 = self.chi @ self.f0
            phi = getattr(self, f"phi{'' if sType == 'Fine' else sType}")

            # Compute solution sequentially
            u = np.zeros((self.L, M), dtype=self.numType)
            u[0] = np.linalg.solve(phi, rhs0)
            for l in range(self.L-1):
                u[l+1] = np.linalg.solve(phi, chi @ u[l])

            # Store solution
            setattr(self, name, u)

        return getattr(self, name)

    @property
    def uFine(self):
        return self._uNumSeq('Fine')

    @property
    def uDelta(self):
        return self._uNumSeq('Delta')

    @property
    def uCoarse(self):
        return self._uNumSeq('Coarse')

    @property
    def uDeltaCoarse(self):
        return self._uNumSeq('DeltaCoarse')

    # -------------------------------------------------------------------------
    # Iteration method for each algorithm
    # -------------------------------------------------------------------------
    def _preset(self, algo, kwargs):
        if algo == 'PFASST':
            algo = 'TwoGrid'
            kwargs.update({
                'approxCoarse': True,
                'approxSmoother': True,
                'omega': 1,
                'nPreRelax': 1,
                'nPostRelax': 0})
        elif algo == 'STMG':
            algo = 'TwoGrid'
            kwargs.update({
                'approxCoarse': False,
                'approxSmoother': False,
                'nPreRelax': 1,
                'nPostRelax': 0})
        elif algo == 'TFASST':
            algo = 'TwoGrid'
            kwargs.update({
                'approxCoarse': False,
                'approxSmoother': True,
                'nPreRelax': 1,
                'nPostRelax': 0})
        elif algo == 'ATMG':
            algo = 'TwoGrid'
            kwargs.update({
                'approxCoarse': True,
                'approxSmoother': False,
                'nPreRelax': 1,
                'nPostRelax': 0})
        return algo, kwargs


    def iterate(self, algo, u, **kwargs):
        L, phi, chi, f0 = self.L, self.phi, self.chi, self.f0
        phiDelta = self.phiDelta
        phiCoarse, phiDeltaCoarse = self.phiCoarse, self.phiDeltaCoarse
        chiCoarse, TFtoC, TCtoF = self.chiCoarse, self.TFtoC, self.TCtoF

        algo, kwargs = self._preset(algo, kwargs)

        if algo == 'GaussSeidel':

            uPrev = f0
            for l in range(L):
                rhs = chi @ uPrev - phi @ u[l]
                u[l] += np.linalg.solve(phiDelta, rhs)
                uPrev = u[l]

        elif algo == 'Jacobi':

            uPrev = f0
            for l in range(L):
                rhs = chi @ uPrev - phi @ u[l]
                uPrev = u[l].copy()
                u[l] += np.linalg.solve(phiDelta, rhs)

        elif algo == 'Parareal':

            uk = f0
            ukp1 = f0
            for l in range(L):
                uF = np.linalg.solve(phi, chi @ uk)
                uGk = np.linalg.solve(phiDelta, chi @ uk)
                uGkp1 = np.linalg.solve(phiDelta, chi @ ukp1)
                uk = u[l].copy()
                u[l] = uF + uGkp1 - uGk
                ukp1 = u[l]

        elif algo == 'TwoGrid':

            # Algorithm parameters
            nPreRelax = kwargs.get('nPreRelax', 1)
            nPostRelax = kwargs.get('nPostRelax', 0)
            omega = kwargs.get('omega', 1)
            approxCoarse = kwargs.get('approxCoarse', False)
            approxSmoother = kwargs.get('approxSmoother', False)

            # Smoother
            def relax(nRelax):
                phiSmoother = phiDelta if approxSmoother else phi
                for i in range(nRelax):
                    uPrev = f0
                    for l in range(L):
                        rhs = chi @ uPrev - phi @ u[l]
                        uPrev = u[l].copy()
                        u[l] += omega*np.linalg.solve(phiSmoother, rhs)

            # Coarse operator
            if approxCoarse:
                phiCoarse = phiDeltaCoarse

            # Pre-relaxation step
            relax(nPreRelax)

            # Coarse correction
            uPrev = f0
            for l in range(L):
                rhs = chiCoarse @ TFtoC @ uPrev - TFtoC @ phi @ u[l]
                u[l] += TCtoF @ np.linalg.solve(phiCoarse, rhs)
                uPrev = u[l]

            # Post-relaxation step
            relax(nPostRelax)

        else:
            raise NotImplementedError(f'algo = {algo}')

    # -------------------------------------------------------------------------
    # Global matrix form and associated methods
    # -------------------------------------------------------------------------
    def _genBlockBiDiag(self, diagBlock, lowerBlock):
        L = self.L
        null = diagBlock*0
        mat = [
            [null]*(i-1) + [lowerBlock]*(i>0) + [diagBlock] + [null]*(L-1-i)
            for i in range(L)]
        return np.block(mat)

    @property
    def A(self):
        return self._genBlockBiDiag(self.phi, -self.chi)

    @property
    def f(self):
        return np.array([self.chi @ self.f0]+[self.f0*0]*(self.L-1)).ravel()

    def iterationMatrix(self, algo, **kwargs):

        A = self.A
        I = np.eye(self.L*self.M)

        phi, chi, phiDelta = self.phi, self.chi, self.phiDelta
        phiCoarse, phiDeltaCoarse = self.phiCoarse, self.phiDeltaCoarse
        chiCoarse, TFtoC, TCtoF = self.chiCoarse, self.TFtoC, self.TCtoF

        # List of (preconditionner, inverseTag)
        precond = []

        algo, kwargs = self._preset(algo, kwargs)

        if algo == 'GaussSeidel':
            precond += [(self._genBlockBiDiag(phiDelta, -chi), True)]

        elif algo == 'Jacobi':
            precond += [(self._genBlockBiDiag(phiDelta, 0*chi), True)]

        elif algo == 'Parareal':
            precond += [(self._genBlockBiDiag(
                phi, -phi @ np.linalg.solve(phiDelta, chi)), True)]

        elif algo == 'TwoGrid':
            # Algorithm parameters
            nPreRelax = kwargs.get('nPreRelax', 1)
            nPostRelax = kwargs.get('nPostRelax', 0)
            omega = kwargs.get('omega', 1)
            approxCoarse = kwargs.get('approxCoarse', False)
            approxSmoother = kwargs.get('approxSmoother', False)

            # Take into account eventual approximations
            phiSmoother = phiDelta if approxSmoother else phi
            if approxCoarse:
                phiCoarse = phiDeltaCoarse

            # Pre-relaxation
            S = self._genBlockBiDiag(phiSmoother, 0*chi)/omega
            precond += [(S, True)]*nPreRelax

            # Two-grid correction
            TFtoCBar = self._genBlockBiDiag(TFtoC, 0*TFtoC)
            TCtoFBar = self._genBlockBiDiag(TCtoF, 0*TCtoF)
            ACoarse = self._genBlockBiDiag(phiCoarse, -chiCoarse)
            TG = TCtoFBar @ np.linalg.solve(ACoarse, TFtoCBar)
            precond += [(TG, False)]*nPreRelax

            # Post-relaxation
            precond += [(S, True)]*nPostRelax

        else:
            raise NotImplementedError(f'algo = {algo}')

        # Assemble iteration matrix and b vector
        iterMat = I
        bVect = 0*I[:, 0]
        for M, inv in precond:
            R = (I - np.linalg.solve(M, A)) if inv else I - M @ A
            iterMat = R @ iterMat
            bVect = R @ bVect
            bVect += np.linalg.solve(M, self.f) if inv else M @ self.f

        return iterMat, bVect

    # -------------------------------------------------------------------------
    # Method for bounds using Generating Functions
    # -------------------------------------------------------------------------
    def errBoundFunc(self, algo, iOnly=False, **kwargs):

        I = np.eye(self.M)
        phi, chi, phiDelta = self.phi, self.chi, self.phiDelta
        phiCoarse, phiDeltaCoarse = self.phiCoarse, self.phiDeltaCoarse
        TFtoC, TCtoF = self.TFtoC, self.TCtoF

        algo, kwargs = self._preset(algo, kwargs)

        # Compute block operators
        if algo == 'GaussSeidel':
            B01 = I-np.linalg.solve(phiDelta, phi)
            B10 = np.linalg.solve(phiDelta, chi)
            B11 = 0*chi

        elif algo == 'Jacobi':
            B01 = I-np.linalg.solve(phiDelta, phi)
            B10 = 0*chi
            B11 = np.linalg.solve(phiDelta, chi)

        elif algo == 'Parareal':
            B01 = 0*chi
            B10 = np.linalg.solve(phiDelta, chi)
            B11 = np.linalg.solve(phi, chi) - B10

        elif algo == 'TwoGrid':
            # Algorithm parameters (only one pre-relaxation step)
            omega = kwargs.get('omega', 1)
            approxCoarse = kwargs.get('approxCoarse', False)
            approxSmoother = kwargs.get('approxSmoother', False)

            # Phi operators on coarse and fine level, two-grid operator
            phiTG = phiDeltaCoarse if approxCoarse else phiCoarse
            phiS = phiDelta if approxSmoother else phi
            TG = (I - TCtoF @ np.linalg.solve(phiTG, TFtoC) @ phi)

            # Block operators
            B01 = TG @ (I - omega*np.linalg.solve(phiS, phi))
            B10 = TCtoF @ np.linalg.solve(phiTG, TFtoC) @ chi
            B11 = omega * TG @ np.linalg.solve(phiS, chi)

        else:
            raise NotImplementedError(f'algo = {algo}')

        # Wether or not using the interface approximation
        if iOnly:
            eM = np.array([[0.]*(self.M-1) + [1.]])
            B01, B10, B11 = eM @ B01, eM @ B10, eM @ B11
            B01, B10, B11 = B01[:, -1], B10[:, -1], B11[:, -1]
            norm = abs
        else:
            norm = lambda x: np.linalg.norm(x, ord=np.inf)

        # Generate and return error bound function
        alpha, beta, gamma = norm(B11), norm(B10), norm(B01)
        print(alpha, beta, gamma)
        return lambda n, k: gfma.pbi(n, k, alpha, beta, gamma, **kwargs)



def getBlockOperators(lamDt, M, method, form=None, **kwargs):

    # Reduce M for collocation with exact end-point prolongation
    exactProlong = kwargs.get('exactProlong', False)
    if exactProlong and method == 'COLLOCATION':
        M -= 1

    # Set nodes and associated polynomial approximation
    nodes = kwargs.get('nodes',
                       'LEGENDRE' if method=='COLLOCATION' else 'EQUID')
    qType = kwargs.get('qType', 'LOBATTO')
    if qType not in ['LOBATTO', 'RADAU-II', 'RADAU-I', 'GAUSS']:
        raise ValueError(f'qType={qType}')
    if isinstance(nodes, str):
        if nodes == 'EQUID' or M < 2:
            nodes = np.linspace(0, 1, M+1)[1:] if qType == 'RADAU-II' else \
                np.linspace(0, 1, M+1)[:-1] if qType == 'RADAU-I' else \
                np.linspace(0, 1, M+2)[1:-1] if qType == 'GAUSS' else \
                np.linspace(1, 0, M)[-1::-1]  # LOBATTO, with tau=1 for M=1
        elif nodes in ['LEGENDRE', 'CHEBY-1', 'CHEBY-2']:
            nodes = LagrangeApproximation(('LEGENDRE', M), qType=qType).points
            # nodes = np.abs((nodes + 1)/2)
            nodes = (nodes + 1)/2
        else:
            raise NotImplementedError(f'nodes={nodes}')
    nodes = np.round(np.ravel(nodes), 14)
    if not ((min(nodes) >= 0) and (max(nodes) <= 1)):
        raise ValueError(f'inconsistent nodes : {nodes}')
    M = len(nodes)
    deltas = np.array(
        [tauR-tauL for tauL, tauR in zip([0]+list(nodes)[:-1], list(nodes))])

    # Node formulation
    if form is None:
        form = 'Z2N' if method == 'COLLOCATION' else 'N2N'
    if form not in ['Z2N', 'N2N']:
        raise ValueError('form argument can only be '
                         'N2N (node-to-node) or Z2N (zero-to-node), '
                         f'got {form}')

    # Runge-Kutta types methods
    if method in RK_METHODS:

        # Default node-to-node formulation
        nStepPerNode = kwargs.get('nStepPerNode', 1)

        # Compute amplification factor
        z = lamDt*deltas/nStepPerNode
        R = STABILITY_FUNCTION_RK[method](z)**(-nStepPerNode)

        # Build phi and chi matrices
        phi = np.diag(R)
        phi[1:,:-1][np.diag_indices(M-1)] = -1
        chi = np.zeros((M, M))
        chi[0, -1] = 1

        # Eventually switch to zero-to-node formulation
        if form == 'Z2N':
            T = np.tril(np.ones((M, M)))
            phi = T @ phi
            chi = T @ chi

    # Collocation methods
    elif method == 'COLLOCATION':

        # Default zero-to-node formulation
        polyApprox = LagrangeApproximation(nodes)
        Q = polyApprox.getIntegrationMatrix([(0, tau) for tau in nodes])

        if exactProlong:
            # Using exact prolongation
            nodes = np.array(nodes.tolist()+[1])
            weights = polyApprox.getIntegrationMatrix([(0, 1)]).ravel()
            phi = np.zeros((M+1, M+1))*lamDt
            phi[:-1, :-1] = np.eye(M) - lamDt*Q
            phi[-1, :-1] = -lamDt*weights
            phi[-1, -1] = 1
            chi = np.zeros((M+1, M+1))
            chi[:, -1] = 1
        else:
            phi = np.eye(M) - lamDt*Q
            chi = polyApprox.getInterpolationMatrix([1]).repeat(M, axis=0)

        # Eventually switch to node-to-node formulation
        if form == 'N2N':
            T = np.eye(M)
            T[1:,:-1][np.diag_indices(M-1)] = -1
            phi = T @ phi
            chi = T @ chi

    elif method == 'MULTISTEP':  # Adams-Bashforth method

        # Default node-to-node formulation
        a = (1+3/2*lamDt*deltas)
        b = -lamDt/2*deltas

        phi = np.eye(M) + 0*lamDt
        phi[1:,:-1][np.diag_indices(M-1)] = -a[1:]
        phi[2:,:-2][np.diag_indices(M-2)] = -b[2:]

        chi = np.zeros((M, M)) + 0*lamDt
        chi[0, -1] = a[0]
        chi[0, -2] = b[0]
        chi[1, -1] = b[1]

        # Eventually switch to zero-to-node formulation
        if form == 'Z2N':
            T = np.tril(np.ones((M, M)))
            phi = T @ phi
            chi = T @ chi

    else:
        raise NotImplementedError(f'method = {method}')

    return phi, chi, nodes, form


def getTransferOperators(nodesFine, nodesCoarse):
    # Build polynomial approximations
    polyApproxFine = LagrangeApproximation(nodesFine)
    polyApproxCoarse = LagrangeApproximation(nodesCoarse)
    # Compute interpolation matrix
    TFtoC = polyApproxFine.getInterpolationMatrix(nodesCoarse)
    TCtoF = polyApproxCoarse.getInterpolationMatrix(nodesFine)
    return TFtoC, TCtoF

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from gfm.util import setFig

    s = GFMSolver(lam=-1, u0=1, dt=0.5*np.pi, L=20)

    MCoarse = 3
    ratio = 3
    M = ratio+1 if MCoarse == 1 else ratio*(MCoarse-1)+1
    fineMethod = 'COLLOCATION'
    deltaMethod = 'TRAP'
    nodesType = 'LEGENDRE'
    qType = 'LOBATTO'

    iType = 'RAND'
    nIter = 3
    nPreRelax = 1

    s.setFineLevel(M=M, method=fineMethod, nodes=nodesType, qType=qType)
    s.setPhiDelta(deltaMethod)
    s.setCoarseLevel(MCoarse, nodes=nodesType, qType=qType)
    s.setPhiDeltaCoarse(deltaMethod)

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
        t, np.abs(uDelta-uExact), '--', label=f'Delta, {deltaMethod}')
    plt.semilogy(# Coarse level solution
        tCoarse, np.abs(uCoarse-uExactCoarse), 's-', label='Coarse level')
    plt.semilogy(# Delta coarse level solution
        tCoarse, np.abs(uDeltaCoarse-uExactCoarse), '-.',
        label=f'Delta coarse, {deltaMethod}')
    # plt.ylim(1e-11, 10)
    plt.vlines(
        [0] + list(s.times[:, -1]), *plt.ylim(), colors='gray', linewidth=0.2)
    setFig('Time', 'Error vs exact solution', grid=False)

    if True:

        uBGS = s.getInitU(iType)
        for k in range(nIter):
            s.iterate('GaussSeidel', uBGS)
        uBGS = s.getU(uBGS)

        uBJ = s.getInitU(iType)
        for k in range(nIter):
            s.iterate('Jacobi', uBJ)
        uBJ = s.getU(uBJ)

        uParareal = s.getInitU(iType)
        for k in range(nIter):
            s.iterate('Parareal', uParareal)
        uParareal = s.getU(uParareal)

        uTG = s.getInitU(iType)
        for k in range(nIter):
            s.iterate(
                'TwoGrid', uTG, approxCoarse=False, approxSmoother=True,
                nPreRelax=nPreRelax)
        uTG = s.getU(uTG)

        uTG2 = s.getInitU(iType)
        for k in range(nIter):
            s.iterate(
                'TwoGrid', uTG2, approxCoarse=True, approxSmoother=False,
                nPreRelax=nPreRelax)
        uTG2 = s.getU(uTG2)

        uPFASST = s.getInitU(iType)
        for k in range(nIter):
            s.iterate(
                'TwoGrid', uPFASST, approxCoarse=True, approxSmoother=True,
                nPreRelax=nPreRelax)
        uPFASST = s.getU(uPFASST)

        uSTMG = s.getInitU(iType)
        for k in range(nIter):
            s.iterate(
                'TwoGrid', uSTMG, approxCoarse=False, approxSmoother=False,
                nPreRelax=nPreRelax)
        uSTMG = s.getU(uSTMG)

        plt.figure()
        # plt.semilogy(t, np.abs(uBGS-uFine), label=f'BGS, {deltaMethod}')
        # plt.semilogy(t, np.abs(uBJ-uFine), label=f'BJ, {deltaMethod}')
        # plt.semilogy(t, np.abs(uParareal-uFine), label=f'Parareal, {deltaMethod}')
        plt.semilogy(t, np.abs(uSTMG-uFine), label='STMG')
        plt.semilogy(t, np.abs(uPFASST-uFine), label='PFASST')
        plt.semilogy(t, np.abs(uTG2-uFine), '--', label='ATMG')
        plt.semilogy(t, np.abs(uTG-uFine), '-.', label='TFASST')

        plt.vlines(s.times[:, -1], *plt.ylim(), colors='gray', linewidth=0.2)
        setFig('Time', 'Error vs fine solution', f'After {nIter} iterations',
               grid=False)
