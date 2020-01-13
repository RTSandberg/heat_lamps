
# from warm_lpm5 import panels
# from warm_lpm5 import field_functions as field
# from warm_lpm5 import time_step_functions as step

import panels
import field_functions as field
import time_step_functions as step

import numpy as np
import os
from numba import jit
from numba import njit
# import time



@njit
def f0(x,v,L,vth, amp):
    """
    """
    return 1./ np.sqrt(2.*np.pi) /vth * np.exp(-v**2/2./vth**2) * (1+amp *np.cos(2*np.pi/L*x))

@njit
def f_M(x,v,L,vth):
    """
    """
    return 1./ np.sqrt(2.*np.pi) /vth * np.exp(-v**2/2./vth**2)


if __name__== '__main__':
    # command line options
    # npanels_x
    # delta
    # field
    # re-mesh
    # re-mesh frequency
    # num_steps

    L = 2 * 2*np.pi
    k = 2*np.pi / L
    vth = 1.
    amp = .01
    q = -1.
    m = 1.
    qm = q/m
    xlim = (0,L)
    npanels_x = 32
    vlim = (-5,5)
    npanels_v = 32
    npanels = npanels_x * npanels_v

    Elim = (-1.5*amp/k, 1.5*amp/k)

    delta = .1
    interpolation = 'cubic'
    do_remesh = False
    remesh_freq = 0
    if do_remesh:
        remesh_string = ''
    else:
        remesh_string = 'no'

    dt = .1
    num_steps = 200
    tf = dt * num_steps


    movie_freq = 5
    diag_freq = 1
    dump_freq = min(movie_freq,diag_freq)


    sim_group = 'Landau_damping_modular_remeshing_delta_%.03f'%delta

    ## the following should occur for each parameter combination

    sim_name = 'LD_remeshing_%i_steps_interp_%s_linear_standardcase_'%(remesh_freq,interpolation) +        'L_%.03f_vth_%.03f_amp_%.03f_'%(L,vth,amp) +        'npanels_x_%i_npanels_v_%i_'%(npanels_x,npanels_v) +        'vmax_%.03f_delta_%.03f_'%(vlim[1],delta) +        'dt_%.03f_num_steps_%i_'%(dt,num_steps) +        'tf_%.03f'%(tf)
    sim_title = 'LD with %s re-meshing every %i steps, interpolation: %s,\n'%(remesh_string, remesh_freq,interpolation) +        'L %.03f, vth %.03f, amp %.03f,\n'%(L,vth,amp) +        'npanels_x %i, npanels_v %i, '%(npanels_x,npanels_v) +        'vmax %.03f, delta %.03f,\n'%(vlim[1],delta) +        'dt %.03f, nt %i, '%(dt,num_steps) +        'tf %.03f'%(tf)

    
    sim_dir = sim_name + '/'
    # analysis_dir = sim_dir + 'simulation_analysis/'
    analysis_dir = sim_dir

    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    # if not os.path.exists(analysis_dir):
    #     os.makedirs(analysis_dir)

    species_data = panels.Panels()
    species_data.L = L
    species_data.vth = vth
    species_data.q = q
    species_data.m = m
    species_data.qm = qm
    species_data.set_xs_vs(npanels_x,xlim,npanels_v,vlim)
    species_data.set_panels()
    species_data.set_f0_weights(lambda x,v: f0(x,v,L,vth,amp))
    species_data.field_obj = field.atan_field(delta)
    species_data.field_obj = field.exact_field()

    panels.Panels.initialize = step.initialize_leapfrog
    panels.Panels.update = step.update_leapfrog
    panels.Panels.diag_dump = step.diag_dump_leapfrog

    species_data.setup_diagnostic_dir(sim_dir)



    species_data.initialize(dt)

    import run_sim
    import plot_stuff
    # clear output data
    import shutil
    shutil.rmtree(species_data.output_dir)