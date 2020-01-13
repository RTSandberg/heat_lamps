from scipy.interpolate import griddata


from __main__ import *

species_data.diag_dump(0)

plotvs = np.zeros_like(species_data.xs)
fs = np.zeros_like(plotvs)


for iter_num in range(1,num_steps+1):
    # clear_output(wait=True)
    simtime = iter_num * dt
    species_data.update(dt)
#     update_leapfrog(species_data.xs, species_data.vs_old, \
#                     species_data.vs_new, species_data.Es, \
#                     species_data.weights, dt, delta)
    
    
    # re-mesh
    if do_remesh:
        if np.mod(iter_num, remesh_freq) == 0:
#             print('at step %i, remeshing'%iter_num)
            # xs = x0s
            modxs = np.mod(species_data.xs, L)
            np.copyto(plotvs,species_data.vs_new)
            np.copyto(fs, species_data.f0s)
            np.copyto(species_data.xs, species_data.x0s)
            # vs = v0s
            np.copyto(species_data.vs_new, species_data.v0s)
            # f0s = [remesh midpoints, re-mesh vertices]
            f0mids = griddata((np.hstack([modxs - L, modxs, modxs+L]), \
                           np.hstack([plotvs, plotvs, plotvs])), \
                          np.hstack([fs,fs,fs]), \
                         (species_data.x0s[:npanels], species_data.v0s[:npanels]),\
                         method=interpolation)
            f0verts = griddata((np.hstack([modxs - L, modxs, modxs+L]), \
                           np.hstack([plotvs, plotvs, plotvs])), \
                          np.hstack([fs,fs,fs]), \
                         (species_data.x0s[npanels:], species_data.v0s[npanels:]),\
                         method=interpolation)
            species_data.f0s = np.hstack([f0mids, f0verts])
            # weights[:npanels] = f0s[:npanels] * dx * dv * q
            species_data.weights[:npanels] = f0mids * species_data.dx * species_data.dv * species_data.q 

            species_data.Es = species_data.field_obj.calc_E(species_data.xs, species_data.xs,\
                                                            species_data.weights, L)
            # variable re-meshing
#             remesh_freq = np.random.randint(7,14)
        
    if np.mod(iter_num,dump_freq) == 0:
#         print('dumping at step %i'%iter_num)
        species_data.diag_dump(iter_num)
        