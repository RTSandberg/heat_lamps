#include "field_functions_gpu.h"


void calc_E_atan(int nt, double *Es, double *targets, int ns, double *sources,  double *q_ws, double L, double delta) {
    int i;
    int j;
    double wadj = 1/(1-delta/sqrt(1+delta*delta));
    double delta_factor = sqrt( 1 + 1.0/(delta*delta));

//     #pragma omp parallel for
    #pragma acc parallel loop
    for (i = 0; i < nt; i++)
    {
        double Ei = 0;
        double xi = targets[i];

//         #pragma omp for reduction(+:Ei)
        #pragma acc loop reduction(+:Ei)
        for (j = 0; j < ns; j++)
        {
            double xj = sources[j];
            double nz = (xi - xj) / L;

            if (nz < -0.5) {
                nz += 1.0;
            }
            if (nz > 0.5) {
                nz -= 1.0;
            }
            Ei += wadj * q_ws[j] * (1.0/M_PI * atan (delta_factor * tan(M_PI * nz)) - nz);
        }
        Es[i] = Ei;
    }

}

void calc_E_mq(int nt, double *Es, double *targets, int ns, double *sources,  double *q_ws, double L, double delta) {
    int i;
    int j;
    double deltaLsq = delta * delta / L / L;
    double norm_delta_L = sqrt(1 + 4 * deltaLsq);


    #pragma acc parallel loop
    for (i = 0; i < nt; i++)
    {
        double Ei = 0;
        double xi = targets[i];

        #pragma acc loop reduction(+:Ei)
        for (j = 0; j < ns; j++)
        {
            double xj = sources[j];
            double nz = (xi - xj) / L;

            if (nz < -0.5) {
                nz += 1.0;
            }
            if (nz > 0.5) {
                nz -= 1.0;
            }
            Ei += q_ws[j] * (.5 * nz * norm_delta_L / sqrt(nz * nz + deltaLsq) - nz);
        }
        Es[i] = Ei;
    }
}

double sgn(const double num)
{
  return (num < 0) ? -1.0 : (num > 0 ? 1.0 : 0.0);
}


void calc_E_exact(int nt, double *Es, double *targets, int ns, double *sources, double *ws, double L) {
    int i;
    int j;
    double sumxiwi = 0, rhobar = 0;

    for (int ii = 0; ii < ns; ++ii)
    {
        sumxiwi += sources[ii] * ws[ii];
        rhobar += ws[ii];
    }
    sumxiwi /= L;
    rhobar /= -(L);


//     #pragma acc parallel loop
    for (i = 0; i < nt; i++)
    {
        double Ei = 0;
        double xi = targets[i];

//         #pragma acc loop reduction(+:Ei)
        for (j = 0; j < ns; j++)
        {
            double xj = sources[j];
            if (xi < xj)
            {
                Ei -=  .5 * ws[j];
            } else if (xi > xj) {
                Ei += .5 * ws[j];
            }
        }
        Es[i] = Ei + sumxiwi + rhobar * xi;;
    }
}

void calc_E_gauss(int nt, double *Es, double *targets, int ns, double *sources,  double *q_ws, double L, double delta) {
    int i;
    int j;
    double s2e = 1./sqrt(2.)/delta;
    double norm_fac = 1./erf(L/2./sqrt(2.)/delta);

    #pragma acc parallel loop
    for (i = 0; i < nt; i++)
    {
        double Ei = 0;
        double xi = targets[i];

        #pragma acc loop reduction(+:Ei)
        for (j = 0; j < ns; j++)
        {
            double xj = sources[j];
            double nz = (xi - xj) / L;

            if (nz < -0.5) {
                nz += 1.0;
            }
            if (nz > 0.5) {
                nz -= 1.0;
            }
            Ei += q_ws[j] * (.5 * erf(nz*s2e) * norm_fac - nz);
        }
        Es[i] = Ei;
    }
}

