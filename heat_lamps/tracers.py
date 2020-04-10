# tracers.py
"""An extension of panels class for passively advected tracers

Only implemented with leapfrog time stepping

Routines
--------
add_tracers
update
diag_dump

"""

import numpy as np

def add_tracers(self, xs_tracers, vs_tracers, dt):
    """add tracer particles to passively advect and get them staggered for leapfrog time stepping
    
    Parameters
    ----------
    xs_tracers : array like,
        initial positions of tracers
    vs_tracers : array like,
        initial velocity of tracers
    """

    self.have_tracers = True
    self.x0s_tracers = 1. * xs_tracers # ensure a copy, not reference
    self.v0s_tracers = 1. * xs_tracers
    
    self.xs_tracers = 1. * xs_tracers
    self.vs_new_tracers = 1. * vs_tracers
    self.vs_old_tracers = 1. * vs_tracers

    # if sources_on_uniform_grid:
    self.Es_tracers = self.calc_E_uniform_grid(self.xs_tracers)
    # else:
    #     self.Es_tracers = self.calc_E(self.xs_tracers, self.x0s,self.weights, self.L, self.delta)
    
    self.vs_old_tracers -= .5 * dt * self.qm * self.Es_tracers
    self.vs_new_tracers += .5 * dt * self.qm * self.Es_tracers

def update(self, dt, sources_on_uniform_grid=False):
    """advect tracers 1 step

    Parameters
    ----------
    dt : float, size of time step to take
    """
    # xn+1 = xn + dt * v n+.5
    # En+1 = E(xn+1)
    # vn+1.5 = vn+.5 + dt * q/m * En+1
    self.xs_tracers += dt * self.vs_new_tracers
    self.xs_tracers = np.mod(self.xs_tracers, self.L)
    # self.Es = calc_E_atan(self.xs, self.xs, self.weights, self.L, delta)
    if sources_on_uniform_grid:
        self.Es_tracers = self.calc_E_uniform_grid(self.xs_tracers)
    else:
        self.Es_tracers = self.calc_E(self.xs_tracers, self.xs, self.weights, self.L,self.delta)
    # print(Es_tracers)
    self.vs_old_tracers = 1.*self.vs_new_tracers
    self.vs_new_tracers += dt * self.qm * self.Es_tracers

def diag_dump_leapfrog(self,iter_num):
    """
    write diagnostics to file

    Parameters
    ----------
    iter_num : float, which iteration simulation is at
    """
    np.ndarray.tofile(self.xs_tracers, self.output_dir + 'xs/xs_tracers_%i'%iter_num)
    np.ndarray.tofile(.5 * (self.vs_old_tracers + self.vs_new_tracers), self.output_dir + 'vs/vs_tracers_%i'%iter_num)
    np.ndarray.tofile(self.Es_tracers, self.output_dir + 'Es/Es_tracers_%i'%iter_num)
