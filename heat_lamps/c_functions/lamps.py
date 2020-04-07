import ctypes
import numpy as np

_lamps = ctypes.CDLL('lamps_c.so')
c_double_p = ctypes.POINTER(ctypes.c_double)

def initialize(xs, vs_old, vs_new, q_ws, Es, qm, L, delta, dt):
    global _lamps
    
    nx_c = ctypes.c_int(xs.size)
    qm_c = ctypes.c_double(qm)
    L_c = ctypes.c_double(L)
    delta_c = ctypes.c_double(delta)
    dt_c = ctypes.c_double(dt)
    
    xs_p = xs.ctypes.data_as(c_double_p)
    vs_new_p = vs_new.ctypes.data_as(c_double_p)
    vs_old_p = vs_old.ctypes.data_as(c_double_p)
    qws_p = q_ws.ctypes.data_as(c_double_p)
    Es_p = Es.ctypes.data_as(c_double_p)
    
    result = _lamps.initialize(nx_c, xs_p, vs_old_p, vs_new_p, qws_p, Es_p, qm_c, L_c, delta_c, dt_c)
    return result


    
def step(xs, vs_old, vs_new, q_ws, Es, qm, L, delta, dt):
    global _lamps
    
    nx_c = ctypes.c_int(xs.size)
    qm_c = ctypes.c_double(qm)
    L_c = ctypes.c_double(L)
    delta_c = ctypes.c_double(delta)
    dt_c = ctypes.c_double(dt)
    
    xs_p = xs.ctypes.data_as(c_double_p)
    vs_new_p = vs_new.ctypes.data_as(c_double_p)
    vs_old_p = vs_old.ctypes.data_as(c_double_p)
    qws_p = q_ws.ctypes.data_as(c_double_p)
    Es_p = Es.ctypes.data_as(c_double_p)
    
    result = _lamps.step(nx_c, xs_p, vs_old_p, vs_new_p, qws_p, Es_p, qm_c, L_c, delta_c, dt_c)
    return result
    
def gather(vs_old, vs_new):
    global _lamps
    
    nx_c = ctypes.c_int(vs_new.size)
    vs_new_p = vs_new.ctypes.data_as(c_double_p)
    vs_old_p = vs_old.ctypes.data_as(c_double_p)
    
    result = _lamps.gather(nx_c, vs_old_p, vs_new_p)
    return result

def run_leapfrog_fns(xs, vs_new, q_ws, Es, qm, L, delta, Nt, dt):
    global _lamps
    
    nx_c = ctypes.c_int(xs.size)
    qm_c = ctypes.c_double(qm)
    L_c = ctypes.c_double(L)
    delta_c = ctypes.c_double(delta)
    dt_c = ctypes.c_double(dt)
    Nt_c = ctypes.c_int(Nt)
    
    xs_p = xs.ctypes.data_as(c_double_p)
#     vs_old_p = vs_old.ctypes.data_as(c_double_p)
    vs_new_p = vs_new.ctypes.data_as(c_double_p)
    qws_p = q_ws.ctypes.data_as(c_double_p)
    Es_p = Es.ctypes.data_as(c_double_p)
    result = _lamps.run_leapfrog_fns(nx_c, xs_p, vs_new_p, qws_p, Es_p, qm_c, L_c, delta_c, Nt_c, dt_c)
    return result
    
def run_sim_leapfrog(xs, vs_new, q_ws, Es, qm, L, delta, Nt, dt):
    global _lamps
    
    nx_c = ctypes.c_int(xs.size)
    qm_c = ctypes.c_double(qm)
    L_c = ctypes.c_double(L)
    delta_c = ctypes.c_double(delta)
    dt_c = ctypes.c_double(dt)
    Nt_c = ctypes.c_int(Nt)
    
    xs_p = xs.ctypes.data_as(c_double_p)
    vs_new_p = vs_new.ctypes.data_as(c_double_p)
    qws_p = q_ws.ctypes.data_as(c_double_p)
    Es_p = Es.ctypes.data_as(c_double_p)
    
    result = _lamps.run_sim_leapfrog(nx_c, xs_p, vs_new_p, qws_p, Es_p, qm_c, L_c, delta_c, Nt_c, dt_c)
    return result
    
    