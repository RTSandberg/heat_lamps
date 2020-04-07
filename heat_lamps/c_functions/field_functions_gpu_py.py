import ctypes
import numpy as np

_field_gpu = ctypes.CDLL('field_functions_gpu.so')
c_double_p = ctypes.POINTER(ctypes.c_double)


#_sum.do_sum.argtypes = (ctypes.c_int,ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_double)

def calc_E_exact(targets, sources, ws, L, delta):
    global _field_gpu
    nt = targets.size
    ns = sources.size
 #   array_type = ctypes.c_double * num_numbers
    double_type = ctypes.c_double
    int_type = ctypes.c_int
    Es = np.zeros(nt)

    targets_p = targets.ctypes.data_as(c_double_p)
    sources_p = sources.ctypes.data_as(c_double_p)
    ws_p = ws.ctypes.data_as(c_double_p)
    Es_p = Es.ctypes.data_as(c_double_p)

    _field_gpu.calc_E_exact(int_type(nt), Es_p, targets_p, int_type(ns), sources_p, ws_p, double_type(L))
    return Es

def calc_E_atan(targets, sources, ws, L, delta):
    global _field_gpu
    nt = targets.size
    ns = sources.size
    double_type = ctypes.c_double
    int_type = ctypes.c_int

    Es = np.zeros_like(targets)

    targets_p = targets.ctypes.data_as(c_double_p)
    sources_p = sources.ctypes.data_as(c_double_p)
    ws_p = ws.ctypes.data_as(c_double_p)
    Es_p = Es.ctypes.data_as(c_double_p)

    _field_gpu.calc_E_atan(int_type(nt), Es_p, targets_p, int_type(ns), sources_p, ws_p, double_type(L), double_type(delta))
    return Es

def calc_E_mq(targets, sources, ws, L, delta):
    global _field_gpu
    nt = targets.size
    ns = sources.size
    double_type = ctypes.c_double
    int_type = ctypes.c_int

    Es = np.zeros_like(targets)

    targets_p = targets.ctypes.data_as(c_double_p)
    sources_p = sources.ctypes.data_as(c_double_p)
    ws_p = ws.ctypes.data_as(c_double_p)
    Es_p = Es.ctypes.data_as(c_double_p)

    c_val = _field_gpu.calc_E_mq(int_type(nt), Es_p, targets_p, int_type(ns), sources_p, ws_p, double_type(L), double_type(delta))
    return  Es

def calc_E_gauss(targets, sources, ws, L, delta):
    global _field_gpu
    nt = targets.size
    ns = sources.size
    double_type = ctypes.c_double
    int_type = ctypes.c_int

    Es = np.zeros_like(targets)

    targets_p = targets.ctypes.data_as(c_double_p)
    sources_p = sources.ctypes.data_as(c_double_p)
    ws_p = ws.ctypes.data_as(c_double_p)
    Es_p = Es.ctypes.data_as(c_double_p)

    c_val = _field_gpu.calc_E_gauss(int_type(nt), Es_p, targets_p, int_type(ns), sources_p, ws_p, double_type(L), double_type(delta))
    return  Es
