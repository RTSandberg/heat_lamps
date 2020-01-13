
import numpy as np
from numba import jitclass
from numba import float32, float64
from numba import int32
from numba import njit


@jitclass([('delta',float64)])
class exact_field:
    """
    """

    def __init__(self,delta):
        """
        """
        self.delta = 0.

    def calc_E(self,targets,sources,weights, L):
        rhobar = -np.sum(weights)/L
        
        E_stored = np.zeros_like(targets)
        for i, target in enumerate(targets):
            E_stored[i] = .5 * np.dot(np.sign(target - sources), weights)
            E_stored[i] += np.dot(sources,weights)/L + target * rhobar
        
        return E_stored

@njit
def calc_E_direct(targets, sources, weights, L):
    """
    """
    rhobar = -np.sum(weights)/L
    
    E_stored = np.zeros_like(targets)
    for i, target in enumerate(targets):
        E_stored[i] = .5 * np.dot(np.sign(target - sources), weights)
        E_stored[i] += np.dot(sources,weights)/L + target * rhobar
    
    return E_stored
calc_E_direct(np.linspace(0,1),np.array([.5]),np.array([1.]),1.);

class sort_field:
    def __init__(self,delta):
        self.delta = 0

    def calc_E(self,targets, sources, weights, L):
    
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

def calc_E_sort(targets, sources, weights, L):
    """
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


@jitclass([('delta',float64)])
class atan_field:
    """
    """
    def __init__(self, delta):
        self.delta = delta

    def calc_E(self,targets,sources,weights, L):
#         zs = targets - sources
        wadj = 1/(1-self.delta/np.sqrt(1+self.delta**2))

        E = np.zeros_like(targets)
        for i, target in enumerate(targets):
            for j, source in enumerate(sources):
                z = (target - source)/L
                if (abs(z - .5) > 1e-12 and abs(z+.5) > 1e-12):
                    E[i] += weights[j] *( 1/np.pi *                         np.arctan( np.sqrt( 1 + 1./self.delta**2)*                        np.tan(np.pi * z)) - np.mod(z-.5,1.) + .5)
        return wadj * E


@njit(fastmath=True)
def calc_E_atan(targets, sources, weights, L, delta):
#         zs = targets - sources
    wadj = 1/(1-delta/np.sqrt(1+delta**2))
    TOL = 1e-12
    E = np.zeros_like(targets)
    for i, target in enumerate(targets):
        for j, source in enumerate(sources):
            z = (target - source)/L
            if (abs(z-.5)>TOL and abs(z + .5) > TOL) :
                E[i] += weights[j] * ( 1/np.pi *                     np.arctan( np.sqrt( 1 + 1./delta**2)*                    np.tan(np.pi * z)) - np.mod(z-.5,1.) + .5)
    return wadj * E

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