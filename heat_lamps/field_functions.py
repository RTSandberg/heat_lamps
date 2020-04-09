# heat_lamps/heat_lamps/field_functions.py
"""Contains functionality for evaluating field in 1d periodic Vlasov-Poisson simulations

Routines
--------

calc_E_tree
calc_E_tree_gpu
calc_E_exact
calc_E_sort
calc_E_RK
calc_E_atan
calc_U
"""
import numpy as np
import numba
from numba import jitclass
from numba import float32, float64
from numba import int32
from numba import njit

import os
import sys
import resource
import mpi4py.MPI as MPI
from heat_lamps.barytree import BaryTreeInterface as BT

# treecode parameters
maxPerSourceLeaf=50
maxPerTargetLeaf=10
GPUpresent=False
theta=0.7
treecodeOrder=5
# gaussianAlpha=1.0
approximation = BT.Approximation.LAGRANGE
singularity = BT.Singularity.SKIPPING
computeType = BT.ComputeType.PARTICLE_CLUSTER

verbosity=0




def calc_E_tree(targets,sources,weights,L,delta,kernelName='mq'):
    """calculate E using BaryTree treecode
    
    Calculate E at `targets` from `sources` with weights `weights`.  System size is `L`. Softening parameter is `delta` 
    
    Parameters
    ----------
    targets : array-like, where to calculate E
    sources : array-like, locations of source charges
    weights : array-like, charges of source charges
    L : float, size of computational domain
    delta : float, degree of softening

    Returns
    -------
    E : ndarray, electric field at targets

    Notes
    -----
    Uses atan or mq kernel
    """
    if kernelName =='atan':
        kernel = BT.Kernel.ATAN
    else:
        kernel = BT.Kernel.MQ
    numberOfKernelParameters=2
    kernelParameters=np.array([L, delta])

    Nt = targets.size
    Ns = sources.size
    
    # W = np.ones(num_per_proc)
    # Q = -1. * L / N * W

    return BT.callTreedriver(  Nt, Ns, 
                               np.zeros(Nt), np.zeros(Nt), np.copy(targets), np.ones(Nt), 
                               np.zeros(Ns), np.zeros(Ns), sources, weights, np.ones(Ns),
                               kernel, numberOfKernelParameters, kernelParameters, singularity, approximation,
                               computeType, treecodeOrder, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
                               GPUpresent, verbosity, sizeCheck = 1.0)

def calc_E_tree_gpu(targets,sources,weights,L,delta,kernelName='mq'):
    """calculate E using BaryTree treecode on gpu
    
    Calculate E at `targets` from `sources` with weights `weights`.  System size is `L`. Softening parameter is `delta` 
    
    Parameters
    ----------
    targets : array-like, where to calculate E
    sources : array-like, locations of source charges
    weights : array-like, charges of source charges
    L : float, size of computational domain
    delta : float, degree of softening

    Returns
    -------
    E : ndarray, electric field at targets
    """
    if kernelName =='atan':
        kernel = BT.Kernel.ATAN
    else:
        kernel = BT.Kernel.MQ

    maxPerSourceLeaf=500
    maxPerTargetLeaf=500
    GPUpresent=True
    numberOfKernelParameters=2
    kernelParameters=np.array([L, delta])

    Nt = targets.size
    Ns = sources.size

    return BT.callTreedriver(  Nt, Ns,
                               np.zeros(Nt), np.zeros(Nt), np.copy(targets), np.zeros(Nt),
                               np.zeros(Ns), np.zeros(Ns), sources, weights, np.ones_like(weights),
                               kernel, numberOfKernelParameters, kernelParameters, singularity, approximation,
                               computeType, treecodeOrder, theta, maxPerSourceLeaf, maxPerTargetLeaf, 
                               GPUpresent, verbosity, sizeCheck = 1.0)




@njit(parallel=True)
def calc_E_exact(targets,sources,weights,L,delta):
    """calculate E 
    
    Calculate E at `targets` from `sources` with weights `weights`.  System size is `L`. Softening parameter is `delta` 
    
    Parameters
    ----------
    targets : array-like, where to calculate E
    sources : array-like, locations of source charges
    weights : array-like, charges of source charges
    L : float, size of computational domain
    delta : float, degree of softening

    Returns
    -------
    E : ndarray, electric field at targets
    """
#     rhobar = -np.sum(weights)/L
    rhobar = 0
    for w in weights:
        rhobar -= w
    rhobar /= L
    
    sum_ywj = 1/L * np.dot(sources,weights)
    sum_ywj = 0
    for ii, y in enumerate(sources):
        sum_ywj += y*weights[ii]
    sum_ywj = sum_ywj/L
    
    E = np.zeros_like(targets)
#     cumE = 0
    for ii in numba.prange(targets.size):
        for jj, y in enumerate(sources):
            E[ii] += .5*np.sign(targets[ii] - y)*weights[jj]
        E[ii] += targets[ii]*rhobar + sum_ywj
        
    # E -= np.mean(E)
    return E

def calc_E_sort(targets, sources, weights, L,delta):
    """Calculate E using exact kernel
    
    Calculate E at `targets` from `sources` with weights `weights`.  System size is `L`. Softening parameter is `delta` 
    
    Parameters
    ----------
    targets : array-like, where to calculate E
    sources : array-like, locations of source charges
    weights : array-like, charges of source charges
    L : float, size of computational domain
    delta : float, degree of softening


    Returns
    -------
    E : ndarray, electric field at targets
    """
    
    # for now
#     targets = sources
    
    rhobar = -np.sum(weights)/L
    
    pos = np.hstack([targets,sources])
    tot_weights = np.hstack([np.zeros_like(targets),weights])
    
    
    
    [sortpos,inds] = np.unique(pos,return_inverse=True)
    sortweights = np.zeros_like(sortpos)
    for ii, ind in enumerate(inds):
    #     print(ind, ii)
        sortweights[ind] += tot_weights[ii]
    sorted_field = np.zeros_like(sortpos)
    sorted_field[0] = -np.sum(sortweights[1:])
    for ii in range(1,len(sortpos)):
        sorted_field[ii] = sorted_field[ii-1] + sortweights[ii-1] + sortweights[ii]
    E_stored = sorted_field[inds]
    E_stored = .5 * E_stored[:len(targets)]

    E_stored +=  targets * rhobar +np.dot(sources,weights)/L
    
    return E_stored


# alternate E RK
@numba.njit(parallel=True)
def calc_E_RK(targets,sources,q_weights,L,epsilon):
    """Calculate E
    
    Calculate E at `targets` from `sources` with weights `weights`.  System size is `L`. Softening parameter is `delta`
    Uses a multiquadric kernel  
    
    Parameters
    ----------
    targets : array-like, where to calculate E
    sources : array-like, locations of source charges
    weights : array-like, charges of source charges
    L : float, size of computational domain
    delta : float, degree of softening

    Returns
    -------
    E : ndarray, electric field at targets
    
    Notes
    -----
    """
    # a = 1./L * np.dot(sources, q_weights)
    # rhobar = -1./L * np.sum(q_weights)
    epsLsq = epsilon**2 / L**2
    norm_epsL = np.sqrt(1 + 4*epsLsq)
    
    E = np.zeros_like(targets)
    for ii in numba.prange(targets.size):
        xt = targets[ii]
        for jj in numba.prange(sources.size):
            xs = sources[jj]
            z = xt - xs
            modz = (z + L*(z < -L/2.) - L*(z > L/2.))/L
            E[ii] += q_weights[jj] * (.5 * modz * norm_epsL \
            / np.sqrt(modz**2 + epsLsq) - modz )
    return E

def calc_E_gauss(targets,sources,q_weights,L,epsilon):
    """
    Parameters
    ----------
    
    Notes
    -----
    E = 
    rhobar = -1/L * sum_i [sources[i] * q_wi]
    """
    s2e = 1./np.sqrt(2.)/epsilon
    norm_fac = 1./spe.erf(L/2/np.sqrt(2)/epsilon)
    
    
    E = np.zeros_like(targets)
    for ii, xt in enumerate(targets):
        for jj, xs in enumerate(sources):
            z = xt - xs
            modz = (z + L*(z < -L/2.) - L*(z > L/2.))/L
            E[ii] += q_weights[jj] * (.5 * spe.erf(modz*s2e) \
            * norm_fac - modz )
    return E


@njit(fastmath=True,parallel=True)
def calc_E_atan(targets, sources, weights, L, delta):
    """Calculate E
    
    Calculate E at `targets` from `sources` with weights `weights`.  System size is `L`. Softening parameter is `delta` 
    
    Parameters
    ----------
    targets : array-like, where to calculate E
    sources : array-like, locations of source charges
    weights : array-like, charges of source charges
    L : float, size of computational domain
    delta : float, degree of softening
    
    Returns
    -------
    E : ndarray, electric field at targets
    """
    wadj = 1/(1-delta/np.sqrt(1+delta**2))
    TOL = 1e-15
    E = np.zeros_like(targets)
    for i in numba.prange(targets.size):
        for j, source in enumerate(sources):
            z = (targets[i] - source)/L
            if (abs(z-.5)>TOL and abs(z + .5) > TOL) :
                E[i] += weights[j] * ( 1/np.pi * \
                    np.arctan( np.sqrt( 1 + 1./delta**2)* np.tan(np.pi * z)) - \
                    np.mod(z-.5,1.) + .5)
    E = wadj * E
    return E

@njit
def calc_U(sources, weights, L):
    """
    
    K(z) = .5*sign(z) - 1/L *z
    -d G(z)/dz = K(z):
    G(z) = -.5*|z| + 1/2/L * z**2
    """
    U_sum = 0
    for ii,xi in enumerate(sources):
        for jj,xj in enumerate(sources):
            U_sum += -weights[ii] * weights[jj] * (.5*abs(xi-xj) - 1./2./L * (xi-xj)**2)
    return  .5 * U_sum
