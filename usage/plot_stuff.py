import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.animation as manimation
from scipy.interpolate import griddata


from __main__ import *

do_plot_panels = False

extents_x = np.linspace(xlim[0],xlim[1],npanels_x, endpoint=False)
extents_v = np.linspace(vlim[0],vlim[1],npanels_v+1)
dx = extents_x[1]-extents_x[0]
dv = extents_v[1]-extents_v[0]

#         dv = (vlim[1] - vlim[0])/npanels_v
mid_v_list = np.linspace(vlim[0],vlim[1],npanels_v,endpoint=False) + .5*dv
# print(mid_v)

# mid_v = mid_v + .5 * (mi)
[x_mids, v_mids] = np.meshgrid(extents_x+.5*dx,mid_v_list)
# print(v_mids, 


# phase points | phase
# ----------------------
# E            | delta f

# plot using precomputed data
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title=sim_group, artist='Matplotlib',
                comment='')
writer = FFMpegWriter(fps=5, metadata=metadata)

tlist = dt * (np.arange(0,num_steps,diag_freq))
num_frames = len(tlist)

Esq_list = np.zeros(num_frames)
num_grid = 100
gridE = np.zeros([num_grid, num_frames])
gridx = np.linspace(0,L,num_grid)
maxE = np.zeros(num_frames)


fig,ax = plt.subplots(2,2,figsize=(8,6))
# dt = .1
# mydist.calc_E(L)

with writer.saving(fig, sim_dir+ sim_name + ".mp4", dpi=100):

    ax_ph = ax[0][0]

    fxs = np.fromfile(species_data.output_dir + 'xs/xs_0')
    plotvs = np.fromfile(species_data.output_dir + 'vs/vs_0')
    fEs = np.fromfile(species_data.output_dir + 'Es/Es_0')
    ffs = np.fromfile(species_data.output_dir + 'fs/fs_0')
    fws = np.fromfile(species_data.output_dir + 'weights/weights_0')
    modx = np.mod(fxs,L)
    # ax_ph.plot(modx, plotvs, '.')
    sc_ph = ax_ph.scatter(modx,plotvs, c = np.log10(ffs), marker='.')

    cb_ph = plt.colorbar(sc_ph, ax=ax_ph)
    cb_ph.set_label(r'$\log_{10}(f)$')

    ax_ph.set_xlim([0,L])
    ax_ph.set_xlabel(r'x[$v_{0}/\omega_p$]')
    ax_ph.set_title('phase space')
    ax_ph.set_ylabel(r'v[$v_0$]')

    ax_interp_ph = ax[0][1]
    plot_fs = np.reshape(ffs[:npanels_x*npanels_v], [npanels_x, npanels_v])
    #     i_ph = ax_interp_ph.imshow(np.log10(np.reshape(plot_fs[:n, []),\
    i_ph = ax_interp_ph.imshow(np.log10(plot_fs.transpose()),                        origin='lower',extent=[0, L, vlim[0],vlim[-1]],aspect='auto')
    cb_i_ph = plt.colorbar(i_ph, ax=ax_interp_ph)
    cb_i_ph.set_label(r'$\log_{10}(f)$')
    ax_interp_ph.set_xlim([0,L])
    ax_interp_ph.set_xlabel(r'x[$v_{0}/\omega_p$]')
    #     ax_interp_ph.set_title('phase space interpolated with scipy interpolate.griddata %s'%interpolation)
    ax_interp_ph.set_title('interpolated')
    ax_interp_ph.set_ylabel(r'v[$v_0$]')

    ax_E = ax[1][0]
    # ax_E.plot(modx, species_data.Es, '.')
    sc_E = ax_E.scatter(modx, fEs,  c = np.log10(ffs), marker='.')
    cb_E = plt.colorbar(sc_E, ax=ax_E)
    cb_E.set_label(r'$\log_{10}(f)$')

    ax_E.set_xlim([0,L])
    ax_E.set_ylim(Elim)
    ax_E.set_xlabel(r'x[$v_{th}/\omega_p$]')
    ax_E.set_title('E')
    ax_E.set_ylabel(r'E[$mv_0\omega_p/e$]')
    ax_E.grid()

    ax_delf = ax[1][1]
    plot_fM = f_M(x_mids,v_mids,L,vth)
    delf = ax_delf.imshow(plot_fs.transpose()-plot_fM,                        origin='lower',cmap='coolwarm',extent=[0, L, vlim[0],vlim[-1]],aspect='auto')
    cb_delf = plt.colorbar(delf, ax=ax_delf)
    cb_delf.set_label(r'$f-f_M$')
    ax_delf.set_xlim([0,L])
    ax_delf.set_xlabel(r'x[$v_{0}/\omega_p$]')
    ax_delf.set_title('f - Maxwellian')
    ax_delf.set_ylabel(r'v[$v_0$]')


    fig.suptitle('time 0.000, ' + sim_title)
    fig.canvas.draw()
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)
    writer.grab_frame()

    diag_iter = 0

    for iter_num in range(1,num_steps+1):
        simtime = iter_num * dt
    #         species_data.update(dt)

        # re-mesh after the diagnostics

        # diagnostic/ movie block
        if np.mod(iter_num,movie_freq) == 0:

            xs = np.fromfile(species_data.output_dir + 'xs/xs_%i'%iter_num)
            modx = np.mod(xs, L)
            plotvs = np.fromfile(species_data.output_dir + 'vs/vs_%i'%iter_num)
            fEs = np.fromfile(species_data.output_dir + 'Es/Es_%i'%iter_num)
            fs = np.fromfile(species_data.output_dir + 'fs/fs_%i'%iter_num)
            fws = np.fromfile(species_data.output_dir + 'weights/weights_%i'%iter_num)

    #             modx = np.mod(species_data.xs, L)
    #             plotvs = .5 * (species_data.vs_old + species_data.vs_new)
    #             fs = species_data.f0s
            sc_ph.set_offsets(np.c_[modx,plotvs])
            sc_ph.set_array(np.log10(fs))

    #             for ii, column in enumerate(species_data.phase_panels[:-1]):
    #                 for jj, panel in enumerate(column):
    #                     panel_inds = panel[1]
    #             #         panel_inds = np.hstack([panel_inds,panel_inds])
    #                     # print(species_data.xs[panel_inds], species_data.vs_new[panel_inds])
    #                     ax_panels.lines[3*(ii*npanels_v + jj)].set_data(xs[panel_inds], plotvs[panel_inds],'k')
    #                     ax_panels.lines[3*(ii*npanels_v + jj)+1].set_data(xs[panel_inds]+L, plotvs[panel_inds],'k')
    #                     ax_panels.lines[3*(ii*npanels_v + jj)+2].set_data(xs[panel_inds]-L, plotvs[panel_inds],'k')

            b = i_ph.get_array()

            b = griddata((np.hstack([modx - L, modx, modx+L]),                            np.hstack([plotvs, plotvs, plotvs])),                           np.hstack([fs,fs,fs]),                          (x_mids, v_mids),                         method=interpolation)

    #             b = b * (b > 0) + 1e-13 * (b <= 0)
    #             b = np.log(np.sign(driver_mq)*phase_driver[:])
            i_ph.set_array(np.log10(b))

            delf.set_array(b - f_M(x_mids,v_mids,L,vth))

    #         ax_ph.lines[0].set_data(modx,plotvs)
            sc_E.set_offsets(np.c_[modx,fEs])
    #         ax_E.lines[0].set_data(modx,species_data.Es)

            fig.suptitle('time %.03f, '%simtime + sim_title)
            fig.canvas.draw()
            plt.tight_layout()
            fig.subplots_adjust(top=0.82)
            writer.grab_frame()

plt.close()
# In[33]:


# %%time
# plot panels using precomputed data
if do_plot_panels:
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title=sim_group, artist='Matplotlib',
                        comment='')
        writer = FFMpegWriter(fps=5, metadata=metadata)

        fig,ax = plt.subplots(1,1,figsize=(8,6))
        # dt = .1
        # mydist.calc_E(L)

        with writer.saving(fig, sim_dir+ 'show_panels_' + sim_name + ".mp4", dpi=100):

        #     ax_ph = ax[0]
        
                fxs = np.fromfile(species_data.output_dir + 'xs/xs_0')
                plotvs = np.fromfile(species_data.output_dir + 'vs/vs_0')
                fEs = np.fromfile(species_data.output_dir + 'Es/Es_0')
                ffs = np.fromfile(species_data.output_dir + 'fs/fs_0')
                fws = np.fromfile(species_data.output_dir + 'weights/weights_0')
                modx = np.mod(fxs,L)
                
                ax_panels = ax #[2]
                ax_panels.set_xlim([0,L])
                ax_panels.set_xlabel(r'x[$v_{0}/\omega_p$]')
                ax_panels.set_title('phase space panels')
                ax_panels.set_ylabel(r'v[$v_0$]')
                for jj, column in enumerate(species_data.phase_panels):
                        for panel in column:
                                panel_inds = panel[1]
                        #         panel_inds = np.hstack([panel_inds,panel_inds])
                                # print(species_data.xs[panel_inds], species_data.vs_new[panel_inds])
                                ax_panels.plot(species_data.xs[panel_inds], species_data.vs_new[panel_inds],'k')
                                ax_panels.plot(species_data.xs[panel_inds]+L, species_data.vs_new[panel_inds],'k')
                        #             ax_panels.plot(species_data.xs[panel_inds]+2*L, species_data.vs_new[panel_inds],'k')
                        #             ax_panels.plot(species_data.xs[panel_inds]+3*L, species_data.vs_new[panel_inds],'k')
                                ax_panels.plot(species_data.xs[panel_inds]-L, species_data.vs_new[panel_inds],'k')
                        #             ax_panels.plot(species_data.xs[panel_inds]-2*L, species_data.vs_new[panel_inds],'k')
                        #             ax_panels.plot(species_data.xs[panel_inds]-3*L, species_data.vs_new[panel_inds],'k')
                n_panel_lines = 3
                
                sc_ph = ax_panels.scatter(modx,plotvs, c = np.log10(species_data.f0s), marker='.')
                cb_ph = plt.colorbar(sc_ph, ax=ax_ph)
                cb_ph.set_label(r'$\log_{10}(f)$')
                
                fig.suptitle('time 0.000, ' + sim_title)
                fig.canvas.draw()
                plt.tight_layout()
                fig.subplots_adjust(top=0.82)
                writer.grab_frame()
                

                diag_iter = 0
                
                for iter_num in range(1,num_steps+1):
                        simtime = (iter_num) * dt
                #         species_data.update(dt)
                        
                        # re-mesh after the diagnostics

                        # diagnostic/ movie block
                        if np.mod(iter_num,movie_freq) == 0:
                        
                                xs = np.fromfile(species_data.output_dir + 'xs/xs_%i'%iter_num)
                                modx = np.mod(xs, L)
                                plotvs = np.fromfile(species_data.output_dir + 'vs/vs_%i'%iter_num)
                                fEs = np.fromfile(species_data.output_dir + 'Es/Es_%i'%iter_num)
                                fs = np.fromfile(species_data.output_dir + 'fs/fs_%i'%iter_num)
                                fws = np.fromfile(species_data.output_dir + 'weights/weights_%i'%iter_num)

                                sc_ph.set_offsets(np.c_[modx,plotvs])
                        
                                for ii, column in enumerate(species_data.phase_panels):
                                        for jj, panel in enumerate(column):
                                                panel_inds = panel[1]
                                        #         panel_inds = np.hstack([panel_inds,panel_inds])
                                                # print(species_data.xs[panel_inds], species_data.vs_new[panel_inds])
                                                ax_panels.lines[n_panel_lines*(ii*npanels_v + jj)].set_data(xs[panel_inds], plotvs[panel_inds])
                                                ax_panels.lines[n_panel_lines*(ii*npanels_v + jj)+1].set_data(xs[panel_inds]+L, plotvs[panel_inds])
                                #                     ax_panels.lines[n_panel_lines*(ii*npanels_v + jj)+2].set_data(xs[panel_inds]+2*L, plotvs[panel_inds])
                                #                     ax_panels.lines[n_panel_lines*(ii*npanels_v + jj)+3].set_data(xs[panel_inds]+3*L, plotvs[panel_inds])
                                                ax_panels.lines[n_panel_lines*(ii*npanels_v + jj)+2].set_data(xs[panel_inds]-L, plotvs[panel_inds])
                                #                     ax_panels.lines[n_panel_lines*(ii*npanels_v + jj)+5].set_data(xs[panel_inds]-2*L, plotvs[panel_inds])
                                #                     ax_panels.lines[n_panel_lines*(ii*npanels_v + jj)+6].set_data(xs[panel_inds]-3*L, plotvs[panel_inds])
                                

                                fig.suptitle('time %.03f, '%simtime + sim_title)
                                fig.canvas.draw()
                                plt.tight_layout()
                                fig.subplots_adjust(top=0.82)
                                writer.grab_frame()
                        

        plt.close()

# In[39]:


## other phase movie
# phase   | f(v)
# ------------
# density | E
# plot using precomputed data
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title=sim_group, artist='Matplotlib',
                comment='')
writer = FFMpegWriter(fps=5, metadata=metadata)

mid_x_list = extents_x + .5*dx


fig,ax = plt.subplots(2,2,figsize=(8,6))
# dt = .1
# mydist.calc_E(L)

with writer.saving(fig, sim_dir+ 'densities' + sim_name + ".mp4", dpi=100):


    fxs = np.fromfile(species_data.output_dir + 'xs/xs_0')
    plotvs = np.fromfile(species_data.output_dir + 'vs/vs_0')
    fEs = np.fromfile(species_data.output_dir + 'Es/Es_0')
    ffs = np.fromfile(species_data.output_dir + 'fs/fs_0')
    fws = np.fromfile(species_data.output_dir + 'weights/weights_0')
    modx = np.mod(fxs,L)
    # ax_ph.plot(modx, plotvs, '.')

    ax_interp_ph = ax[0][0]
    plot_fs = np.reshape(ffs[:npanels_x*npanels_v], [npanels_x, npanels_v])
    #     i_ph = ax_interp_ph.imshow(np.log10(np.reshape(plot_fs[:n, []),\
    i_ph = ax_interp_ph.imshow(np.log10(plot_fs.transpose()),                        origin='lower',extent=[0, L, vlim[0],vlim[-1]],aspect='auto')
    cb_i_ph = plt.colorbar(i_ph, ax=ax_interp_ph)
    cb_i_ph.set_label(r'$\log_{10}(f)$')
    ax_interp_ph.set_xlim([0,L])
    ax_interp_ph.set_xlabel(r'x[$v_{0}/\omega_p$]')
    #     ax_interp_ph.set_title('phase space interpolated with scipy interpolate.griddata %s'%interpolation)
    ax_interp_ph.set_title('interpolated')
    ax_interp_ph.set_ylabel(r'v[$v_0$]')
    
    ax_fv = ax[0][1]
    fvs = np.sum(plot_fs,axis=0)*dx
    p_fv = ax_fv.plot(fvs, mid_v_list)
    ax_fv.set_xlabel('f(v)')
    ax_fv.set_ylabel('v')
    ax_fv.set_title('f(v)')
    ax_fv.set_xlim([0,L])
    ax_fv.set_ylim(vlim)
    ax_fv.grid()
    
    ax_dens = ax[1][0]
    dens = np.sum(plot_fs,axis=1)*dv
    pdens = ax_dens.plot(mid_x_list,dens)
    ax_dens.set_xlabel('x')
    ax_dens.set_ylabel('n')
    ax_dens.set_title('density n(x)')
    ax_dens.set_xlim([0,L])
    ax_dens.grid()

    ax_E = ax[1][1]
    # ax_E.plot(modx, species_data.Es, '.')
    sc_E = ax_E.scatter(modx, fEs,  c = np.log10(ffs), marker='.')
    cb_E = plt.colorbar(sc_E, ax=ax_E)
    cb_E.set_label(r'$\log_{10}(f)$')

    ax_E.set_xlim([0,L])
    ax_E.set_ylim(Elim)
    ax_E.set_xlabel(r'x[$v_{th}/\omega_p$]')
    ax_E.set_title('E')
    ax_E.set_ylabel(r'E[$mv_0\omega_p/e$]')
    ax_E.grid()

    fig.suptitle('time 0.000, ' + sim_title)
    fig.canvas.draw()
    plt.tight_layout()
    fig.subplots_adjust(top=0.82)
    writer.grab_frame()

    diag_iter = 0

    for iter_num in range(1,num_steps+1):
        simtime = iter_num * dt
    #         species_data.update(dt)

        # re-mesh after the diagnostics

        # diagnostic/ movie block
        if np.mod(iter_num,movie_freq) == 0:

            xs = np.fromfile(species_data.output_dir + 'xs/xs_%i'%iter_num)
            modx = np.mod(xs, L)
            plotvs = np.fromfile(species_data.output_dir + 'vs/vs_%i'%iter_num)
            fEs = np.fromfile(species_data.output_dir + 'Es/Es_%i'%iter_num)
            fs = np.fromfile(species_data.output_dir + 'fs/fs_%i'%iter_num)
            fws = np.fromfile(species_data.output_dir + 'weights/weights_%i'%iter_num)


            b = i_ph.get_array()

            b = griddata((np.hstack([modx - L, modx, modx+L]),                            np.hstack([plotvs, plotvs, plotvs])),                           np.hstack([fs,fs,fs]),                          (x_mids, v_mids),                         method=interpolation)

    #             b = b * (b > 0) + 1e-13 * (b <= 0)
    #             b = np.log(np.sign(driver_mq)*phase_driver[:])
            i_ph.set_array(np.log10(b))
        
            plot_fs = b.transpose()
            ax_fv.lines[0].set_data(np.sum(plot_fs,axis=0)*dx,mid_v_list)
            ax_dens.lines[0].set_data(mid_x_list,np.sum(plot_fs,axis=1)*dv)
        
            sc_E.set_offsets(np.c_[modx,fEs])
    #         ax_E.lines[0].set_data(modx,species_data.Es)

            fig.suptitle('time %.03f, '%simtime + sim_title)
            fig.canvas.draw()
            plt.tight_layout()
            fig.subplots_adjust(top=0.82)
            writer.grab_frame()

plt.close()

# In[40]:


## diagnostics over run    

tlist = dt * (np.arange(0,num_steps+1,diag_freq))
num_frames = len(tlist)

Esq_list = np.zeros(num_frames)
num_grid = 100
gridE = np.zeros([num_grid, num_frames])
gridx = np.linspace(0,L,num_grid)
maxE = np.zeros(num_frames)

f1 = np.zeros(num_frames)
kinetic = np.zeros(num_frames)
potential = np.zeros(num_frames)
total = np.zeros(num_frames)
momentum = np.zeros(num_frames)
drift = np.zeros(num_frames)
thermal = np.zeros(num_frames)

# thermal
# drift energy
# Fourier modes of E

diag_iter = 0
for iter_num in range(0,num_steps+1, diag_freq):
    simtime = iter_num * dt


    modx = np.mod(np.fromfile(species_data.output_dir + 'xs/xs_%i'%iter_num), L)
    plotvs = np.fromfile(species_data.output_dir + 'vs/vs_%i'%iter_num)
    total_E = np.fromfile(species_data.output_dir + 'Es/Es_%i'%iter_num)
    fs = np.fromfile(species_data.output_dir + 'fs/fs_%i'%iter_num)
    fws = np.fromfile(species_data.output_dir + 'weights/weights_%i'%iter_num)
#             modx = np.mod(species_data.xs, L)
    # max E
#             total_E = fEs
    maxE[diag_iter] = max(abs(total_E))

    sortpos, sortind = np.unique(modx,return_index = True)

    unpad = np.hstack([sortpos[-1] - L, sortpos, sortpos[0] + L])
    Esort = total_E[sortind]
    Esortpad = np.hstack([Esort[-1], Esort, Esort[0]])

    indj = 0
    for ii in range(num_grid):
        xi = gridx[ii]
        while unpad[indj+1] < xi:
            indj += 1
        theta = (xi - unpad[indj]) / (unpad[indj+1] - unpad[indj])
        gridE[ii,diag_iter] = Esortpad[indj] * (1.-theta) + Esortpad[indj+1]*theta


    xwidths = np.zeros(Esort.size)
    xwidths[1:-1] = sortpos[2:] - sortpos[:-2]
    xwidths[0] = sortpos[1]+L - sortpos[-1]
    xwidths[-1] = sortpos[0] + L - sortpos[-2]

    Esq_list[diag_iter] = .25 * sum(Esort**2 * xwidths)
    
    ## ||f||_1
    f1[diag_iter] = sum(fws)
    momentum[diag_iter] = np.dot(fws, plotvs)
    drift[diag_iter] = momentum[diag_iter] / f1[diag_iter]

    
    kinetic[diag_iter] = np.dot(fws, plotvs**2)
#     potential[diag_iter] = .5 * sum(total_E**2)

    
    diag_iter +=1
    #end diagnostics
f1 /= q
momentum *= m/q
kinetic *= m/q
thermal = np.sqrt(2 * (kinetic/m/f1 - drift**2))
total = kinetic + Esq_list


# In[41]:


fig, ax = plt.subplots(2,1)
ax_f1 = ax[0]
ax_f1.plot(tlist, f1)
ax_f1.set_title(r'$\sum_i f(x_i)\Delta x_i \Delta v_i \approx |f_1|$')
ax_f1.grid()

ax_mom = ax[1]
ax_mom.plot(tlist, momentum)
ax_mom.set_title('Momentum ' + r'$\approx \sum_i v_i f(x_i)\Delta x_i\Delta v_i$')
ax_mom.set_xlabel(r't[$1/\omega_p$]')
ax_mom.grid()

fig.tight_layout()
plt.savefig(sim_dir + '/mass_momentum_'+ sim_name + '.png')

plt.close()

# In[42]:


fig, ax = plt.subplots(3,1)
ax_U = ax[0]
ax_U.plot(tlist, Esq_list)
ax_U.set_title(r'$\sum_i E(x_i)^2 \Delta x_i$')
ax_U.set_xlabel(r't[$1/\omega_p$]')
# ax_U.plot(tlist, potential)
ax_U.grid()

ax_K = ax[1]
ax_K.plot(tlist, kinetic)
ax_K.set_title(r'$\sum w_i v_i^2$')
ax_K.grid()

ax_T = ax[2]
ax_T.plot(tlist, total)
ax_T.set_title('$\sum_i E(x_i)^2 \Delta x_i$ +$\sum w_i v_i^2$' )
ax_T.grid()

# ax_T.plot(tlist, kinetic + potential)
fig.suptitle('Energy')
fig.tight_layout()
fig.subplots_adjust(top=0.86)
plt.savefig(sim_dir + '/energy_'+ sim_name + '.png')

plt.close()

# In[43]:


fig, ax = plt.subplots(2,1)
ax_drift = ax[0]
ax_drift.plot(tlist, drift)
ax_drift.set_ylabel(r'v[$v_0$]')
ax_drift.set_title('drift velocity ' + r'$ (\sum w_i v_i)/\sum w_i$')
ax_drift.grid()

ax_th = ax[1]
ax_th.plot(tlist, thermal)
ax_th.set_xlabel(r't[$1/\omega_p$]')
ax_th.set_ylabel(r'v[$v_0$]')
ax_th.set_title('thermal velocity, ' + r'$\sqrt{2}\sqrt{\sum w_i v_i^2 / \sum w_i - drift^2 }$')
ax_th.grid()
fig.tight_layout()
plt.savefig(sim_dir + '/drift_thermal_'+ sim_name + '.png')

plt.close()

# In[44]:


fig, ax = plt.subplots(figsize=(8,6))
extent = [0, tf, 0, L]

pltET = plt.imshow(gridE[1:-1,:], extent = extent,aspect='auto')
cb1 = plt.colorbar(pltET, ax=ax)
cb1.set_label(r'$E[mc\omega_p/e]$')

plt.xlabel('t')
plt.ylabel('x')
plt.title('E linearly interpolated to grid,\n' + sim_title)
fig.subplots_adjust(top=0.86)
plt.savefig(sim_dir + '/gridEvtime_'+ sim_name + '.png')

plt.close()

# In[62]:



l2E = .5*Esq_list
max_inds = np.where((l2E[2:] - l2E[1:-1] <= 0)* (l2E[1:-1] - l2E[:-2] >= 0))
max_inds = max_inds[0]
max_inds = max_inds[1:-1]
# max_inds = np.hstack([max_inds[1:2],max_inds[4:7:2]])
# max_inds = max_inds[1:6:2]
logE = np.log(l2E[max_inds])
tmax = tlist[max_inds]
slopes = (logE[:-1]-logE[1:])/2/(tmax[:-1]-tmax[1:])
print('slopes')
print(slopes)
gamma_m = np.mean(slopes)
print('average slope')
print(gamma_m)
plt.figure()
plt.plot(tmax,logE,'.')
plt.plot(tmax,logE[0] + 2*gamma_m*(tmax-tmax[0]))


# In[64]:


plt.figure(figsize=(7,6))
l2E = .5*Esq_list
plt.semilogy(tlist,l2E,label=r'$\frac{1}{2}\sum E_i^2 \Delta x_i$')
# plt.semilogy(tlist[max_inds+1],l2E[max_inds+1],'*')
# gamma1 = np.sqrt(np.pi)/(k*vth)**3 *(1+1.5*(k*vth)**2)**1.5 *  np.exp(-1.5 - 1/(k*vth)**2)
# gamma2 = np.sqrt(np.pi)/(k*vth)**3  *  np.exp(-1.5 - 1/(k*vth)**2)
gamma3 = .1533
# print(gamma1)
# print(gamma2)
# plt.semilogy(tlist,l2E[0]*np.exp(-gamma1*tlist))
ind1 = max_inds[0]+1
E1 = l2E[ind1]
t1 = tlist[ind1]
plt.semilogy(tlist,E1*np.exp(-2*gamma3*(tlist-t1)),label=r'$E_0e^{-2\cdot0.1533 t}$')
plt.semilogy(tlist,E1*np.exp(2*gamma_m*(tlist-t1)),label=r'$E_0e^{-2\cdot %.04f t}$'%abs(gamma_m))
plt.ylim([1e-14,1e-2])
plt.xlabel('t')
plt.title('L2 norm of E v time\n' + sim_title)
plt.grid()
plt.legend()
plt.savefig(sim_dir + 'L2Evtime_'+sim_name + '.png')

plt.close()
