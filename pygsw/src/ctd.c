#include <math.h>
#include <gswteos-10.h>

double 
ctd_density(double SP, double t, double p, double lat, double lon) {
    double SA = gsw_sa_from_sp(SP,p,lon,lat);
    double CT = gsw_ct_from_t(SA,t,p);
    double rho = gsw_rho(SA,CT,p);
}


