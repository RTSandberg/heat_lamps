#include <math.h>

void calc_E_gauss(int nt, double *Es, double *targets, int ns, double *sources,  double *ws, double L, double delta);
void calc_E_atan(int nt, double *Es,double *targets, int ns, double *sources,  double *ws, double L, double delta);
void calc_E_mq(int nt, double *Es, double *targets, int ns, double *sources,  double *ws, double L, double delta);
void calc_E_exact(int nt, double *Es, double *targets, int ns,  double *sources,  double *ws, double L);
