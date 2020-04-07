// lamps_c.c
//
// c routine for performing Lagrangian particle simulation on GPU
// 
// Use leapfrog time stepping, periodic boundary conditions
// evaluate electric field using Green's functions, softened
//
// take in list of xs, vs_new, vs_old, and charges
//
// assumes xs are in [0,L]

#include "lamps_c.h"

int initialize(int nx, double *xs, double *vs_old, double *vs_new, double *q_ws, double *Es, double qm, double L, double delta, double dt) {
    int ii;
    
    calc_E_atan(nx, Es, xs, nx, xs, q_ws, L, delta);
    for (ii = 0; ii < nx; ++ii) {
        vs_old[ii] = vs_new[ii] - .5 * dt * qm * Es[ii];
        vs_new[ii] += .5 * dt * qm * Es[ii];
    }
    return 0;
}


inline int step(int nx, double *xs, double *vs_old, double *vs_new, double *q_ws, double *Es, double qm, double L, double delta, double dt) {
    int ii;
    for (ii = 0; ii < nx; ++ii) {
        xs[ii] += dt * vs_new[ii];
    }
    calc_E_atan(nx, Es, xs, nx, xs, q_ws, L, delta);
    for (ii = 0; ii < nx; ++ii) {
        vs_old[ii] = vs_new[ii];
        vs_new[ii] += dt * qm * Es[ii];
    }
    return 0;
}

int gather(int nx, double *vs_old, double *vs_new) {
    int ii;
    for (ii = 0; ii < nx; ++ii) {
        vs_new[ii] = .5*(vs_new[ii] + vs_old[ii]);
    }
    return 0;
}

int run_leapfrog_fns(int nx, double *xs, double *vs_new, double *q_ws, double *Es, double qm, double L, double delta, int Nt, double dt) {
    double vs_old[nx];
    int ii, iter_num;
    
    initialize(nx, xs, vs_old, vs_new, q_ws, Es, 
               qm, L, delta, dt);
    
    for (iter_num = 0; iter_num < Nt; ++iter_num) {
        step(nx, xs, vs_old, vs_new, q_ws, Es, 
             qm, L, delta, dt);
    }
    
    gather(nx, vs_old, vs_new);
    return 0;
}
    

int run_sim_leapfrog(int nx, double *xs, double *vs_new, double *q_ws, double *Es, double qm, double L, double delta, int Nt, double dt)
{
    double vs_old[nx];
    int ii, iter_num;
    // initialize
    calc_E_atan(nx, Es, xs, nx, xs, q_ws, L, delta);
    
    for (ii = 0; ii < nx; ++ii) {
        vs_old[ii] = vs_new[ii] - .5 * dt * qm * Es[ii];
        vs_new[ii] += .5 * dt * qm * Es[ii];
    }

    // loop and update
    for (iter_num = 1; iter_num < Nt; ++iter_num) {
        for (ii = 0; ii < nx; ++ii) {
            xs[ii] += dt * vs_new[ii];
        }
        calc_E_atan(nx, Es, xs, nx, xs, q_ws, L, delta);
        for (ii = 0; ii < nx; ++ii) {
            vs_old[ii] = vs_new[ii];
            vs_new[ii] += dt * qm * Es[ii];
        } 
    }
    
    // unstagger
    for (ii = 0; ii < nx; ++ii) {
        vs_new[ii] = .5*(vs_new[ii] + vs_old[ii]);
    }
    return 0;
}