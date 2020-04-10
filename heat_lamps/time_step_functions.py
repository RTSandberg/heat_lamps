# Copyright 2019, 
# Authors: Ryan Sandberg
# License
"""functions for time stepping

.. TODO::
* take modulus of positions somewhere!  Either in time stepping or in field calculation. Be consistent!
* Choice for now: in time stepping.


"""
import numpy as np

def initialize_leapfrog(self, dt):
    """
    Calculate initial E and stagger velocities so vs_new = v^{n+1.5}, vs_old = v^{n+.5}

    Parameters
    ----------
    dt : float
    """
    self.xs = np.mod(self.xs, self.L)
    # self.Es = self.calc_E(self.xs, self.xs[:self.npanels], self.weights[:self.npanels], self.L,self.delta)
    
    # for initialization field calculation, coming from points on a uniform grid:
    # f0T = np.reshape(self.f0s[:self.npanels],(self.npanels_x,self.npanels_v))
    # reduced_qws = self.q * np.sum(f0T,1)*self.dv*self.dx

    Es_reduced = self.calc_E_uniform_grid(np.hstack([self.x_mids, self.x_verts]))

    # Es_reduced = self.calc_E(np.hstack([self.x_mids, self.x_verts]), self.x_mids, reduced_qws, self.L, self.delta)    
    Es_mids = np.repeat(Es_reduced[:self.npanels_x], self.npanels_v)
    Es_verts = np.repeat(Es_reduced[self.npanels_x:], self.npanels_v+1)
    self.Es = np.hstack([Es_mids, Es_verts])

    self.vs_old -= .5 * dt * self.qm * self.Es
    self.vs_new += .5 * dt * self.qm * self.Es

def update_leapfrog(self, dt):
    """
    """
    # xn+1 = xn + dt * v n+.5
    # En+1 = E(xn+1)
    # vn+1.5 = vn+.5 + dt * q/m * En+1
    self.xs += dt * self.vs_new
    self.xs = np.mod(self.xs, self.L)
    self.Es = self.calc_E(self.xs, self.xs[:self.npanels], self.weights[:self.npanels], self.L,self.delta)
    # self.Es = calc_E_atan(self.xs, self.xs, self.weights, self.L, delta)
    self.vs_old[:] = self.vs_new[:]
    self.vs_new += dt * self.qm * self.Es

def diag_dump_leapfrog(self,iter_num):
    """
    write diagnostics to file

    Parameters
    ----------
    iter_num : float, which iteration simulation is at
    """
    np.ndarray.tofile(self.xs, self.output_dir + 'xs/xs_%i'%iter_num)
    np.ndarray.tofile(.5 * (self.vs_old + self.vs_new), self.output_dir + 'vs/vs_%i'%iter_num)
    np.ndarray.tofile(self.f0s, self.output_dir + 'fs/fs_%i'%iter_num)
    np.ndarray.tofile(self.weights, self.output_dir + 'weights/weights_%i'%iter_num)
    np.ndarray.tofile(self.Es, self.output_dir + 'Es/Es_%i'%iter_num)


# In[17]:


def initialize_single_stage(self, dt):
    """
    """
    self.xs = np.mod(self.xs, self.L)
    self.Es = self.field_obj.calc_E(self.xs, self.xs, self.weights, self.L)

def update_RK4(self,dt):
    """
    // F(x,v) = (v, q/mE(x) )
  // un = (xn, vn)
  // k1 = (v1,a1) = F(un) = F(xn,vn) = (vn, q/mE(xn) )
  // k2 = (v2,a2) = F(un + h/2 k1) = F(xn + delt/2 v1, vn + delt/2 a1)
  //              = ( (vn + delt f1 / 2), q E(xn + delt v1 /2) )
  // k3 = (v3,a3) = F(un + h/2 k2) = F(xn + delt/2 v2, vn + delt/2 a2)
  //              = ( (vn + delt f2 / 2), q E(xn + delt v2 /2) )
  // k4 = (v4,a4) = F(un + h k3) = F(xn + delt v3, vn + delt a3)
  //              = ( (vn + delt f3), q E(xn + delt v3) )
  // un+1 = (xn+1,vn+1) = un + h/6 (k1 + 2k2 + 2k3 + k4)
  //                    = (xn + delt/6 (v1 + 2 v2 + 2 v3 + v4),
  //                    = vn + delt/6 (a1 + 2 a2 + 2 a3 + a4) )
  //
  
    """
    #k1 = (v1,a1) = F(un) = F(xn,vn) = (vn, q/mE(xn) )
    self.Es = self.qm * self.field_obj.calc_E(self.xs , self.xs, self.weights, self.L)
    v1 = self.vs_new[:]
    a1 = self.qm * self.Es
    
    x2 = self.xs + dt/2 * v1
    x2 = np.mod(x2, self.L)
    v2 = v1 + dt/2 * a1
    a2 = self.qm * self.field_obj.calc_E(x2 , x2, self.weights, self.L)
    
    x3 = self.xs + dt/2 * v2
    x3 = np.mod(x3, self.L)
    v3 = v2 + dt/2 * a2
    a3 = self.qm * self.field_obj.calc_E(x3 , x3, self.weights, self.L)
    
    x4 = self.xs + dt * v3
    x4 = np.mod(x4, self.L)
    v4 = v3 + dt * a3
    a4 = self.qm * self.field_obj.calc_E(x4 , x4, self.weights, self.L)
    
    self.xs += dt/6 * (v1 + 2*v2 + 2*v3 + v4)
    self.xs = np.mod(self.xs, self.L)
    self.vs_new += dt/6 * (a1 + 2*a2 + 2*a3 + a4)
    self.Es = self.field_obj.calc_E(self.xs, self.xs, self.weights, self.L)
    
def diag_dump_single_stage(self,iter_num):
    """
    write diagnostics to file
    """
    np.ndarray.tofile(self.xs, self.output_dir + 'xs/xs_%i'%iter_num)
    np.ndarray.tofile(self.vs_new, self.output_dir + 'vs/vs_%i'%iter_num)
    np.ndarray.tofile(self.f0s, self.output_dir + 'fs/fs_%i'%iter_num)
    np.ndarray.tofile(self.weights, self.output_dir + 'weights/weights_%i'%iter_num)
    np.ndarray.tofile(self.Es, self.output_dir + 'Es/Es_%i'%iter_num)
