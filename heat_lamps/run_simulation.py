# Copyright 2020, HeatLAMPS
# Authors: Ryan Sandberg
# License:
"""This file is part of the heat_lamps package.  It contains the run_sim function for the Panels class
"""

import numpy as np
from scipy.interpolate import griddata

def run_sim(self, dt, num_steps,dump_freq = 1):
	"""Run panels simulation

	Performs `Panels.Nt` iterations of the following loop
	* 

	Parameters
	----------
	dt: float, size of time step
	num_steps: int, number of steps to take

    """
	
	self.initialize(dt)
	self.diag_dump(0)
	if self.have_tracers == True:
		self.diag_dump_tracers(0)

	plotvs = np.zeros_like(self.xs)
	fs = np.zeros_like(plotvs)

	print_update_frequency = int((num_steps+1)/10)
	print_update_counter = 0

	for iter_num in range(1,num_steps+1):
		self.update(dt)
		if self.have_tracers == True:
			self.update_tracers(dt)
		
		# re-mesh
		if self.do_remesh:

			if np.mod(iter_num, self.remesh_freq) == 0:
	#             print('at step %i, remeshing'%iter_num)
				# xs = x0s
				modxs = np.mod(self.xs, self.L)
				np.copyto(plotvs,.5*(self.vs_new + self.vs_old))
				np.copyto(fs, self.f0s)
				np.copyto(self.xs, self.x0s)
				# vs = v0s
				np.copyto(self.vs_new, self.v0s)
				np.copyto(self.vs_old, self.v0s)
				# f0s = [remesh midpoints, re-mesh vertices]
				self.f0s = griddata((np.hstack([modxs - self.L, modxs, modxs+self.L]), \
							np.hstack([plotvs, plotvs, plotvs])), \
							np.hstack([fs,fs,fs]), \
							(self.x0s, self.v0s),\
							method=self.interpolation_type)
				# f0_nonpos = np.where(self.f0s <= self.limiter)
				# self.f0s[f0_nonpos] = self.limiter
				
				self.weights[:self.npanels] = self.f0s[:self.npanels] * self.dx * self.dv * self.q 

				# need to calculate E and re-stagger v
				self.initialize(dt)
				
			
		if np.mod(iter_num,dump_freq) == 0:
	#         print('dumping at step %i'%iter_num)
			self.diag_dump(iter_num)
			if self.have_tracers:
				self.diag_dump_tracers(iter_num)
				
		if print_update_counter == print_update_frequency:
			print(f'Iteration number {iter_num}, simulation is about {iter_num/(num_steps+1)*100 :0.0f}% complete')
			print_update_counter = 0
		print_update_counter += 1
			
