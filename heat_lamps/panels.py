import numpy as np
import os


# what's in a panel?  list of midpoint and 4 vertices, weight,
panel = np.dtype([('midpoint',np.uint),('vertices',np.uint,4),('weight',np.float)])
phase_point_leapfrog = np.dtype([('x',np.float),('v_new',np.float),                        ('v_old',np.float),('f0',np.float),                                 ('weight',np.float)])


# In[15]:


class Panels:
    q = -1.
    m = 1.
    qm = -1.
    
    L = 1.
    vth = 0.
    
    delta = 0.
    
    x0s = np.array([])
    v0s = np.array([])
    
    xs = np.array([])
    vs_old = np.array([])
    vs_new = np.array([])
    f0s = np.array([])
    weights = np.array([])
    
    
    npanels_x = 0
    npanels_v = 0
    npanels = 0
    dx = 0
    dv = 0
    
    phase_panels = np.ndarray(shape=(),dtype=panel)
    
    field_obj = 0. # later define this to be a field object
    # must have member function calc_E(xs,ys,weights,L)
    
    ouput_dir = '' # location of diagnostic output
    
    
    def set_xs_vs(self,npanels_x, xlim, npanels_v, vlim):
        """
        """
        
        self.npanels_x = npanels_x
        self.npanels_v = npanels_v
        self.npanels = npanels_x * npanels_v
        nverts = npanels_x * (npanels_v + 1)

        extents_x = np.linspace(xlim[0],xlim[1],npanels_x+1)
        extents_v = np.linspace(vlim[0],vlim[1],npanels_v+1)
        self.dx = extents_x[1]-extents_x[0]
        self.dv = extents_v[1]-extents_v[0]
        
        [v_verts, x_verts] = np.meshgrid(extents_v,extents_x)
        x_verts = x_verts.reshape(x_verts.size)
        v_verts = v_verts.reshape(v_verts.size)

#         dv = (vlim[1] - vlim[0])/npanels_v
        mid_v_list = np.linspace(vlim[0],vlim[1],npanels_v,endpoint=False) + .5*self.dv
        # print(mid_v)

        # mid_v = mid_v + .5 * (mi)
        [v_mids, x_mids] = np.meshgrid(mid_v_list,extents_x[:-1]+.5*self.dx)
        x_mids = x_mids.reshape(x_mids.size)
        v_mids = v_mids.reshape(v_mids.size)

        self.x0s = np.hstack([ x_mids, x_verts])
        self.xs = np.hstack([ x_mids, x_verts])
        
        self.v0s = np.hstack([v_mids, v_verts])
        self.vs_old = np.hstack([v_mids, v_verts])
        self.vs_new = np.hstack([v_mids, v_verts])
        
    def set_panels(self):
        
        self.phase_panels = np.ndarray(shape=(self.npanels_v,self.npanels_x),dtype=panel)
        for jj in range(self.npanels_x):
            for ii in range(self.npanels_v):
                panel_ind = ii*self.npanels_x + jj
                panel_base = self.npanels + panel_ind + ii 
                # add npanels because vertices are stored after midpoints in x0s list
#                 if ii < self.npanels_x-1:
                # stored in lexicographical order
#                     self.phase_panels[ii][jj]['vertices'] = (panel_base, panel_base+1,\
#                           panel_base + npanels_v +1, panel_base + npanels_v + 2)
            #stored in plotting order, clockwise from lower left-hand
                self.phase_panels[ii][jj]['vertices'] = (panel_base, panel_base+1,\
                  panel_base + self.npanels_v + 2, panel_base + self.npanels_v +1)

#                 else:
#                 # stored in lexicographical order
# #                     self.phase_panels[ii][jj]['vertices'] = (panel_base, panel_base+1,jj,jj+1)
#                 # stored in plotting order
#                     self.phase_panels[ii][jj]['vertices'] = (panel_base, panel_base+1,jj+1,jj)
                self.phase_panels[ii][jj]['midpoint'] = panel_ind
                
    def set_f0_weights(self,f0):
        """Set weights 

        Uses function `f0` to set weights of simulation points
        
        Parameters
        ----------
        f0, function object taking arguments x,v
        """
        
        self.f0s = f0(self.xs,self.vs_old)
        self.weights = np.zeros_like(self.xs)
        self.weights[:self.npanels] = self.q * self.f0s[:self.npanels] * self.dx * self.dv
        # have some redundancy right now, as panel weight is carried by the midpoints
        for jj in range(self.npanels_x):
            for ii in range(self.npanels_v):
        #         print(mypts[myarr[ii][jj]['midpoint']]['weight'])
                self.phase_panels[ii][jj]['weight'] =             self.weights[self.phase_panels[ii][jj]['midpoint']]
            
    def setup_diagnostic_dir(self, sim_dir):
        """
        """
        self.output_dir = sim_dir + 'simulation_output/'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.output_dir + 'xs/'):
            os.makedirs(self.output_dir + 'xs/')
        if not os.path.exists(self.output_dir + 'vs/'):
            os.makedirs(self.output_dir + 'vs/')
        if not os.path.exists(self.output_dir + 'Es/'):
            os.makedirs(self.output_dir + 'Es/')
        if not os.path.exists(self.output_dir + 'fs/'):
            os.makedirs(self.output_dir + 'fs/')
        if not os.path.exists(self.output_dir + 'weights/'):
            os.makedirs(self.output_dir + 'weights/')