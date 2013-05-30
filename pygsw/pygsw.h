#ifndef __PYGSW_H__
#define __PYGSW_H__


double gsw_sp_from_c(double C, double t, double p);
double ctd_density(double SP, double t, double p, double lat, double lon);

#endif /* __PYGSW_H__ */
