import numpy as np
cimport numpy as np

cimport cython

np.import_array()

cdef extern from "gswteos-10.h":
     double gsw_rho(double sa, double ct, double p)
     void   gsw_add_barrier(double *input_data, double lon, double lat,
			double long_grid, double lat_grid, double dlong_grid,
			double dlat_grid, double *output_data)
     void   gsw_add_mean(double *data_in, double lon, double lat,
			double *data_out)
     double gsw_alpha(double sa, double ct, double p)
     double gsw_alpha_wrt_t_exact(double sa, double t, double p)

     double gsw_beta_const_t_exact(double sa, double t, double p)
     double gsw_beta(double sa, double ct, double p)
     double gsw_cp_t_exact(double sa, double t, double p)
     double gsw_ct_freezing(double sa, double p, double saturation_fraction)
     double gsw_ct_from_pt(double sa, double pt)
     double gsw_ct_from_t(double sa, double t, double p)
     double gsw_deltasa_from_sp(double sp, double p, double lon, double lat)
     double gsw_delta_sa_ref(double p, double lon, double lat)
     double gsw_dynamic_enthalpy(double sa, double ct, double p)
     double gsw_enthalpy(double sa, double ct, double p)
     double gsw_enthalpy_t_exact(double sa, double t, double p)
     double gsw_entropy_part(double sa, double t, double p)
     double gsw_entropy_part_zerop(double sa, double pt0)
     double gsw_entropy_t_exact(double sa, double t, double p)
     double gsw_fdelta(double p, double lon, double lat)
     double gsw_gibbs(int ns, int nt, int np, double sa, double t, double p)
     double gsw_gibbs_pt0_pt0(double sa, double pt0)
     double gsw_hill_ratio_at_sp2(double t)
     int    gsw_indx(double *x, int n, double z)
     double gsw_internal_energy(double sa, double ct, double p)
     double gsw_kappa_t_exact(double sa, double t, double p)
     double gsw_latentheat_evap_ct(double sa, double ct)
     double gsw_latentheat_evap_t(double sa, double t)
     double gsw_latentheat_melting(double sa, double p)
     double gsw_pot_rho_t_exact(double sa, double t, double p, double p_ref)
     double gsw_pt0_from_t(double sa, double t, double p)
     double gsw_pt_from_ct(double sa, double ct)
     double gsw_pt_from_t(double sa, double t, double p, double p_ref)
     double gsw_rho(double sa, double ct, double p)
     double gsw_rho_t_exact(double sa, double t, double p)
     double gsw_saar(double p, double lon, double lat)
     double gsw_sa_from_sp_baltic(double sp, double lon, double lat)
     double gsw_sa_from_sp(double sp, double p, double lon, double lat)
     double gsw_sa_from_sstar(double sstar, double p,double lon,double lat)
     double gsw_sound_speed(double sa, double ct, double p)
     double gsw_sound_speed_t_exact(double sa, double t, double p)
     double gsw_specvol_anom(double sa, double ct, double p)
     double gsw_specvol(double sa, double ct, double p)
     double gsw_specvol_sso_0_p(double p)
     double gsw_specvol_t_exact(double sa, double t, double p)
     double gsw_sp_from_sa_baltic(double sa, double lon, double lat)
     double gsw_sp_from_sa(double sa, double p, double lon, double lat)
     double gsw_sp_from_sr(double sr)
     double gsw_sp_from_sstar(double sstar, double p,double lon,double lat)
     double gsw_sr_from_sp(double sp)
     double gsw_sstar_from_sa(double sa, double p, double lon, double lat)
     double gsw_sstar_from_sp(double sp, double p, double lon, double lat)
     double gsw_t_freezing(double sa, double p, double saturation_fraction)
     double gsw_t_from_ct(double sa, double ct, double p)
     double gsw_xinterp1(double *x, double *y, int n, double x0)



@cython.boundscheck(False)
@cython.wraparound(False)
def add_barrier(np.ndarray input_data, double lon, double lat, double long_grid, double lat_grid, double dlong_grid, double dlat_grid):
    '''
!  Adds a barrier through Central America (Panama) and then averages
!  over the appropriate side of the barrier
! 
!  input_data   :  data                                         [unitless]
!  lon          :  Longitudes of data degrees east              [0 ... +360]
!  lat          :  Latitudes of data degrees north              [-90 ... +90]
!  longs_grid   :  Longitudes of regular grid degrees east      [0 ... +360]
!  lats_grid    :  Latitudes of regular grid degrees north      [-90 ... +90]
!  dlongs_grid  :  Longitude difference of regular grid degrees [deg longitude]
!  dlats_grid   :  Latitude difference of regular grid degrees  [deg latitude]
!
! gsw_add_barrier  : average of data depending on which side of the 
!                    Panama cannal it is on                                 [unitless]
    '''

    if input_data.ndim != 1:
        raise TypeError('Incorrect number of dimensions. (1!=%s)'%input_data.ndim)
    if input_data.shape[0] != 4:
        raise TypeError('Invalid shape: expecting (4,) got (%s)' % input_data.shape[0])

    cdef np.ndarray[np.double_t, ndim=1, mode='c'] input = np.ascontiguousarray(input_data, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] retval = np.empty((4,))

    gsw_add_barrier(&input[0], lon, lat, long_grid, lat_grid, dlong_grid, dlat_grid, &retval[0])
    return retval

@cython.boundscheck(False)
@cython.wraparound(False)
def add_mean(np.ndarray input_data, double lon, double lat):
    if input_data.ndim != 1:
        raise TypeError('Incorrect number of dimensions. (1!=%s)'%input_data.ndim)
    if input_data.shape[0] != 4:
        raise TypeError('Invalid shape: expecting (4,) got (%s)' % input_data.shape[0])
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] input = np.ascontiguousarray(input_data, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1, mode='c'] retval = np.empty((4,))
    gsw_add_mean(&input[0], lon, lat, &retval[0])
    return retval



@cython.boundscheck(False)
@cython.wraparound(False)
def alpha(double sa, double ct, double p):
    return gsw_alpha(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def alpha_wrt_t_exact(double sa, double t, double p):
    return gsw_alpha_wrt_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def beta_const_t_exact(double sa, double t, double p):
    return gsw_beta_const_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def beta(double sa, double ct, double p):
    return gsw_beta(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def cp_t_exact(double sa, double t, double p):
    return gsw_cp_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def ct_freezing(double sa, double p, double saturation_fraction):
    return gsw_ct_freezing(sa,p,saturation_fraction)

@cython.boundscheck(False)
@cython.wraparound(False)
def ct_from_pt(double sa, double pt):
    return gsw_ct_from_pt(sa,pt)

@cython.boundscheck(False)
@cython.wraparound(False)
def ct_from_t(double sa, double t, double p):
    return gsw_ct_from_t(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def deltasa_from_sp(double sp, double p, double lon, double lat):
    return gsw_deltasa_from_sp(sp,p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def delta_sa_ref(double p, double lon, double lat):
    return gsw_delta_sa_ref(p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def dynamic_enthalpy(double sa, double ct, double p):
    return gsw_dynamic_enthalpy(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def enthalpy(double sa, double ct, double p):
    return gsw_enthalpy(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def enthalpy_t_exact(double sa, double t, double p):
    return gsw_enthalpy_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_part(double sa, double t, double p):
    return gsw_entropy_part(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_part_zerop(double sa, double pt0):
    return gsw_entropy_part_zerop(sa,pt0)

@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_t_exact(double sa, double t, double p):
    return gsw_entropy_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def fdelta(double p, double lon, double lat):
    return gsw_fdelta(p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs(int ns, int nt, int np_, double sa, double t, double p):
    return gsw_gibbs(ns,nt,np_,sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs_pt0_pt0(double sa, double pt0):
    return gsw_gibbs_pt0_pt0(sa,pt0)

@cython.boundscheck(False)
@cython.wraparound(False)
def hill_ratio_at_sp2(double t):
    return gsw_hill_ratio_at_sp2(t)

@cython.boundscheck(False)
@cython.wraparound(False)
def internal_energy(double sa, double ct, double p):
    return gsw_internal_energy(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def kappa_t_exact(double sa, double t, double p):
    return gsw_kappa_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def latentheat_evap_ct(double sa, double ct):
    return gsw_latentheat_evap_ct(sa,ct)

@cython.boundscheck(False)
@cython.wraparound(False)
def latentheat_evap_t(double sa, double t):
    return gsw_latentheat_evap_t(sa,t)

@cython.boundscheck(False)
@cython.wraparound(False)
def latentheat_melting(double sa, double p):
    return gsw_latentheat_melting(sa,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def pot_rho_t_exact(double sa, double t, double p, double p_ref):
    return gsw_pot_rho_t_exact(sa,t,p,p_ref)

@cython.boundscheck(False)
@cython.wraparound(False)
def pt0_from_t(double sa, double t, double p):
    return gsw_pt0_from_t(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def pt_from_ct(double sa, double ct):
    return gsw_pt_from_ct(sa,ct)

@cython.boundscheck(False)
@cython.wraparound(False)
def pt_from_t(double sa, double t, double p, double p_ref):
    return gsw_pt_from_t(sa,t,p,p_ref)

@cython.boundscheck(False)
@cython.wraparound(False)
def rho(double sa, double ct, double p):
    return gsw_rho(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def rho_t_exact(double sa, double t, double p):
    return gsw_rho_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def saar(double p, double lon, double lat):
    return gsw_saar(p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sa_from_sp_baltic(double sp, double lon, double lat):
    return gsw_sa_from_sp_baltic(sp,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sa_from_sp(double sp, double p, double lon, double lat):
    return gsw_sa_from_sp(sp,p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sa_from_sstar(double sstar, double p,double lon,double lat):
    return gsw_sa_from_sstar(sstar, p, lon, lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sound_speed(double sa, double ct, double p):
    return gsw_sound_speed(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def sound_speed_t_exact(double sa, double t, double p):
    return gsw_sound_speed_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def specvol_anom(double sa, double ct, double p):
    return gsw_specvol_anom(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def specvol(double sa, double ct, double p):
    return gsw_specvol(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def specvol_sso_0_p(double p):
    return gsw_specvol_sso_0_p(p)

@cython.boundscheck(False)
@cython.wraparound(False)
def specvol_t_exact(double sa, double t, double p):
    return gsw_specvol_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def sp_from_sa_baltic(double sa, double lon, double lat):
    return gsw_sp_from_sa_baltic(sa,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sp_from_sa(double sa, double p, double lon, double lat):
    return gsw_sp_from_sa(sa,p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sp_from_sr(double sr):
    return gsw_sp_from_sr(sr)

@cython.boundscheck(False)
@cython.wraparound(False)
def sp_from_sstar(double sstar, double p,double lon,double lat):
    return gsw_sp_from_sstar(sstar,p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sr_from_sp(double sp):
    return gsw_sr_from_sp(sp)

@cython.boundscheck(False)
@cython.wraparound(False)
def sstar_from_sa(double sa, double p, double lon, double lat):
    return gsw_sstar_from_sa(sa,p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sstar_from_sp(double sp, double p, double lon, double lat):
    return gsw_sstar_from_sp(sp,p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def t_freezing(double sa, double p, double saturation_fraction):
    return gsw_t_freezing(sa,p,saturation_fraction)

@cython.boundscheck(False)
@cython.wraparound(False)
def t_from_ct(double sa, double ct, double p):
    return gsw_t_from_ct(sa,ct,p)


