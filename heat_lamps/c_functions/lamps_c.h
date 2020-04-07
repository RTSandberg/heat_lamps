// lamps_c.h
#include "field_functions_gpu.h"

int initialize(int nx, double *xs, double *vs_old, double *vs_new,  double *q_ws, double *Es, double qm, double L, double delta, double dt);

int step(int nx, double *xs, double *vs_old, double *vs_new, double *q_ws, double *Es, double qm, double L, double delta, double dt);

int gather(int nx, double *vs_old, double *vs_new);

int run_leapfrog_fns(int nx, double *xs, double *vs_new, double *q_ws, double *Es, double qm, double L, double delta, int Nt, double dt);
int run_sim_leapfrog(int nx, double *xs, double *vs_new, double *charges, double *Es, double qm, double L, double delta, int Nt, double dt);