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
    '''
    ! Replaces NaN's with non-nan mean of the 4 adjacent neighbours
    !
    ! data_in   : data set of the 4 adjacent neighbours   
    ! lon      : longitude
    ! lat       : latitude
    !
    ! data_out : non-nan mean of the 4 adjacent neighbours     [unitless]
    '''
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
    '''
    !  Calculates the thermal expansion coefficient of seawater with respect to 
    !  Conservative Temperature using the computationally-efficient 48-term 
    !  expression for density in terms of SA, CT and p (McDougall et al., 2011)
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! ct     : Conservative Temperature                        [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_alpha : thermal expansion coefficient of seawater (48 term equation)
    '''    
    return gsw_alpha(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def alpha_wrt_t_exact(double sa, double t, double p):
    '''
    ! Calculates thermal expansion coefficient of seawater with respect to 
    ! in-situ temperature
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : insitu temperature                              [deg C]
    ! p      : sea pressure                                    [dbar]
    !
    ! gsw_alpha_wrt_t_exact : thermal expansion coefficient    [1/K]
    !                         wrt (in-situ) temperature
    '''    
    return gsw_alpha_wrt_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def beta_const_t_exact(double sa, double t, double p):
    '''
    ! Calculates saline (haline) contraction coefficient of seawater at 
    ! constant in-situ temperature.
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_beta_const_t_exact : haline contraction coefficient  [kg/g]
    '''    
    return gsw_beta_const_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def beta(double sa, double ct, double p):
    '''
    ! Calculates saline (haline) contraction coefficient of seawater at 
    ! constant in-situ temperature.
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_beta_const_t_exact : haline contraction coefficient  [kg/g]
    '''    
    return gsw_beta(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def cp_t_exact(double sa, double t, double p):
    '''
    ! Calculates isobaric heat capacity of seawater
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_cp_t_exact : heat capacity                           [J/(kg K)]
    '''    
    return gsw_cp_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def ct_freezing(double sa, double p, double saturation_fraction):
    '''
    ! Calculates the Conservative Temperature at which of seawater freezes 
    ! from Absolute Salinity and pressure.
    !
    ! sa     : Absolute Salinity                                 [g/kg]
    ! p      : sea pressure                                      [dbar]
    ! saturation_fraction : saturation fraction
    !
    ! gsw_ct_freezing : Conservative Temperature freezing point  [deg C]
    '''    
    return gsw_ct_freezing(sa,p,saturation_fraction)

@cython.boundscheck(False)
@cython.wraparound(False)
def ct_from_pt(double sa, double pt):
    '''
    ! Calculates Conservative Temperature from potential temperature of seawater  
    !
    ! sa      : Absolute Salinity                              [g/kg]
    ! pt      : potential temperature with                     [deg C]
    !           reference pressure of 0 dbar
    !
    ! gsw_ct_from_pt : Conservative Temperature                [deg C]
    '''    
    return gsw_ct_from_pt(sa,pt)

@cython.boundscheck(False)
@cython.wraparound(False)
def ct_from_t(double sa, double t, double p):
    '''
    ! Calculates Conservative Temperature from in-situ temperature
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    !
    ! gsw_ct_from_t : Conservative Temperature                 [deg C]
    '''    
    return gsw_ct_from_t(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def deltasa_from_sp(double sp, double p, double lon, double lat):
    '''
    ! Calculates Absolute Salinity Anomaly, deltaSA, from Practical Salinity, SP. 
    !
    ! sp     : Practical Salinity                              [unitless]
    ! p      : sea pressure                                    [dbar]
    ! lon   : longitude                                       [deg E]     
    ! lat    : latitude                                        [deg N]
    !
    ! gsw_deltasa_from_sp : Absolute Salinty Anomaly           [g/kg]
    '''    
    return gsw_deltasa_from_sp(sp,p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def delta_sa_ref(double p, double lon, double lat):
    '''
    ! Calculates the Absolute Salinity Anomaly reference value, delta_SA_ref.
    !
    ! p      : sea pressure                                    [dbar]
    ! lon   : longiture                                       [deg E]     
    ! lat    : latitude                                        [deg N]
    !
    ! gsw_delta_sa_ref : Absolute Salinity Anomaly reference value    [g/kg]
    '''    
    return gsw_delta_sa_ref(p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def dynamic_enthalpy(double sa, double ct, double p):
    '''
    !  Calculates dynamic enthalpy of seawater using the computationally-
    !  efficient 48-term expression for density in terms of SA, CT and p
    !  (McDougall et al., 2011)
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! ct     : Conservative Temperature                        [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_dynamic_enthalpy  :  dynamic enthalpy of seawater (48 term equation)
    '''    
    return gsw_dynamic_enthalpy(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def enthalpy(double sa, double ct, double p):
    '''
    !  Calculates specific enthalpy of seawater using the computationally-
    !  efficient 48-term expression for density in terms of SA, CT and p
    !  (McDougall et al., 2011)
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! ct     : Conservative Temperature                        [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_enthalpy  :  specific enthalpy of seawater (48 term equation)
    '''    
    return gsw_enthalpy(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def enthalpy_t_exact(double sa, double t, double p):
    '''
    ! Calculates the specific enthalpy of seawater
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_enthalpy_t_exact : specific enthalpy                 [J/kg]
    '''    
    return gsw_enthalpy_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_part(double sa, double t, double p):
    '''
    ! entropy minus the terms that are a function of only SA
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_entropy_part : entropy part
    '''    
    return gsw_entropy_part(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_part_zerop(double sa, double pt0):
    '''
    ! entropy part evaluated at the sea surface
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! pt0    : insitu temperature                              [deg C]
    ! 
    ! gsw_entropy_part_zerop : entropy part at the sea surface
    '''    
    return gsw_entropy_part_zerop(sa,pt0)

@cython.boundscheck(False)
@cython.wraparound(False)
def entropy_t_exact(double sa, double t, double p):
    '''
    ! Calculates the specific entropy of seawater
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_entropy_t_exact : specific entropy                   [J/(kg K)]
    '''    
    return gsw_entropy_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def fdelta(double p, double lon, double lat):
    '''
    ! Calculates fdelta. 
    !
    ! sp     : Practical Salinity                              [unitless]
    ! p      : sea pressure                                    [dbar]
    ! lon   : longitude                                       [deg E]     
    ! lat    : latitude                                        [deg N]
    !
    ! gsw_fdelta : Absolute Salinty Anomaly                    [unitless]
    '''    
    return gsw_fdelta(p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs(int ns, int nt, int np_, double sa, double t, double p):
    '''
    ! seawater specific Gibbs free energy and derivatives up to order 2
    !
    ! ns     : order of s derivative
    ! nt     : order of t derivative
    ! np     : order of p derivative
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : temperature                                     [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 								-1
    ! gsw_gibbs  : specific Gibbs energy or its derivative	   [J kg  ]
    '''    
    return gsw_gibbs(ns,nt,np_,sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs_pt0_pt0(double sa, double pt0):
    '''
    ! gibbs_tt at (sa,pt,0)
    !
    ! sa     : Absolute Salinity                            [g/kg]
    ! pt0    : potential temperature                        [deg C]
    ! 
    ! gsw_gibbs_pt0_pt0 : gibbs_tt at (sa,pt,0)
    '''    
    return gsw_gibbs_pt0_pt0(sa,pt0)

@cython.boundscheck(False)
@cython.wraparound(False)
def hill_ratio_at_sp2(double t):
    '''
    !  Calculates the Hill ratio, which is the adjustment needed to apply for
    !  Practical Salinities smaller than 2.  This ratio is defined at a 
    !  Practical Salinity = 2 and in-situ temperature, t using PSS-78. The Hill
    !  ratio is the ratio of 2 to the output of the Hill et al. (1986) formula
    !  for Practical Salinity at the conductivity ratio, Rt, at which Practical
    !  Salinity on the PSS-78 scale is exactly 2.
    '''    
    return gsw_hill_ratio_at_sp2(t)

@cython.boundscheck(False)
@cython.wraparound(False)
def internal_energy(double sa, double ct, double p):
    '''
    !  Calculates internal energy of seawater using the computationally
    !  efficient 48-term expression for density in terms of SA, CT and p
    !  (McDougall et al., 2011)
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! ct     : Conservative Temperature                        [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_internal_energy  :  internal_energy of seawater (48 term equation)
    '''    
    return gsw_internal_energy(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def kappa_t_exact(double sa, double t, double p):
    '''
    ! isentropic compressibility of seawater
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    !
    ! gsw_kappa_t_exact : isentropic compressibility           [1/Pa]
    '''    
    return gsw_kappa_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def latentheat_evap_ct(double sa, double ct):
    '''
    ! Calculates latent heat, or enthalpy, of evaporation.
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! ct     : Conservative Temperature                        [deg C]
    ! 
    ! gsw_latentheat_evaporation : latent heat of evaporation  [J/kg]
    '''    
    return gsw_latentheat_evap_ct(sa,ct)

@cython.boundscheck(False)
@cython.wraparound(False)
def latentheat_evap_t(double sa, double t):
    '''
    ! Calculates latent heat, or enthalpy, of evaporation.
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! 
    ! gsw_latentheat_evap_t : latent heat of evaporation       [J/kg]
    '''    
    return gsw_latentheat_evap_t(sa,t)

@cython.boundscheck(False)
@cython.wraparound(False)
def latentheat_melting(double sa, double p):
    '''
    ! Calculates latent heat, or enthalpy, of melting.
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_latentheat_melting : latent heat of melting          [kg/m^3]
    '''    
    return gsw_latentheat_melting(sa,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def pot_rho_t_exact(double sa, double t, double p, double p_ref):
    '''
    ! Calculates the potential density of seawater
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! p_ref  : reference sea pressure                          [dbar]
    ! 
    ! gsw_pot_rho_t_exact : potential density                  [kg/m^3]
    '''    
    return gsw_pot_rho_t_exact(sa,t,p,p_ref)

@cython.boundscheck(False)
@cython.wraparound(False)
def pt0_from_t(double sa, double t, double p):
    '''
    ! Calculates potential temperature with reference pressure, p_ref = 0 dbar. 
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    !
    ! gsw_pt0_from_t : potential temperature, p_ref = 0        [deg C]
    '''    
    return gsw_pt0_from_t(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def pt_from_ct(double sa, double ct):
    '''
    ! potential temperature of seawater from conservative temperature
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! ct     : Conservative Temperature                        [deg C]
    ! p      : sea pressure                                    [dbar]
    !
    ! gsw_pt_from_ct : potential temperature with              [deg C]
    !                  reference pressure of  0 dbar
    '''    
    return gsw_pt_from_ct(sa,ct)

@cython.boundscheck(False)
@cython.wraparound(False)
def pt_from_t(double sa, double t, double p, double p_ref):
    '''
    ! Calculates potential temperature of seawater from in-situ temperature 
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! p_ref  : reference sea pressure                          [dbar]
    !
    ! gsw_pt_from_t : potential temperature                    [deg C]
    '''    
    return gsw_pt_from_t(sa,t,p,p_ref)

@cython.boundscheck(False)
@cython.wraparound(False)
def rho(double sa, double ct, double p):
    '''
    !  Calculates in-situ density from Absolute Salinity and Conservative 
    !  Temperature, using the computationally-efficient 48-term expression for
    !  density in terms of SA, CT and p (McDougall et al., 2011).
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! ct     : Conservative Temperature                        [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_rho  : in-situ density (48 term equation)
    '''    
    return gsw_rho(sa,ct,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def rho_t_exact(double sa, double t, double p):
    '''
    ! Calculates in-situ density of seawater from Absolute Salinity and 
    ! in-situ temperature.
    !
    ! sa     : Absolute Salinity                               [g/kg]
    ! t      : in-situ temperature                             [deg C]
    ! p      : sea pressure                                    [dbar]
    ! 
    ! gsw_rho_t_exact : in-situ density                        [kg/m^3]
    '''    
    return gsw_rho_t_exact(sa,t,p)

@cython.boundscheck(False)
@cython.wraparound(False)
def saar(double p, double lon, double lat):
    '''
    ! Calculates the Absolute Salinity Anomaly Ratio, SAAR.
    !
    ! p      : sea pressure                                    [dbar]
    ! long   : longitude                                       [deg E]     
    ! lat    : latitude                                        [deg N]
    !
    ! gsw_saar : Absolute Salinity Anomaly Ratio               [unitless]
    '''    
    return gsw_saar(p,lon,lat)

@cython.boundscheck(False)
@cython.wraparound(False)
def sa_from_sp_baltic(double sp, double lon, double lat):
    '''
    ! For the Baltic Sea, calculates Absolute Salinity with a value
    ! computed analytically from Practical Salinity
    !
    ! sp     : Practical Salinity                              [unitless]
    ! lon    : longitude                                       [deg E]     
    ! lat    : latitude                                        [deg N]
    ! p      : sea pressure                                    [dbar]
    !
    ! gsw_sa_from_sp_baltic : Absolute Salinity                [g/kg]
    '''    
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


