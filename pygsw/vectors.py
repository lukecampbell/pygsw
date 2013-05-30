
import seawater as gsw
import ctd 
import practical_salinity as pracsal
import numpy as np

alpha = np.vectorize(lambda sa, ct, p : gsw.alpha(sa,ct,p))

alpha_wrt_t_exact = np.vectorize(lambda sa,t,p : gsw.alpha_wrt_t_exact(sa,t,p))

beta_const_t_exact = np.vectorize(lambda sa,t,p : gsw.beta_const_t_exact(sa,t,p))

beta = np.vectorize(lambda sa,ct,p : gsw.beta(sa,ct,p))

cp_t_exact = np.vectorize(lambda sa,t,p : gsw.cp_t_exact(sa,t,p))

ct_freezing = np.vectorize(lambda sa,p,saturation_fraction : gsw.ct_freezing(sa,p,saturation_fraction))

ct_from_pt = np.vectorize(lambda sa,pt : gsw.ct_from_pt(sa,pt))

ct_from_t = np.vectorize(lambda sa,t,p : gsw.ct_from_t(sa,t,p))

deltasa_from_sp = np.vectorize(lambda sp,p,lon,lat: gsw.deltasa_from_sp(sp,p,lon,lat))

delta_sa_ref = np.vectorize(lambda p,lon,lat : gsw.delta_sa_ref(p,lon,lat))

dynamic_enthalpy = np.vectorize(lambda sa,ct,p : gsw.dynamic_enthalpy(sa,ct,p))

enthalpy = np.vectorize(lambda sa,ct,p : gsw.enthalpy(sa,ct,p))

enthalpy_t_exact = np.vectorize(lambda sa,t,p : gsw.enthalpy_t_exact(sa,t,p))

entropy_part = np.vectorize(lambda sa,t,p : gsw.entropy_part(sa,t,p))

entropy_part_zerop = np.vectorize(lambda sa,pt0 : gsw.entropy_part_zerop(sa,pt0))

entropy_t_exact = np.vectorize(lambda sa,t,p : gsw.entropy_t_exact(sa,t,p))

fdelta = np.vectorize(lambda p,lon,lat : gsw.fdelta(p,lon,lat))

gibbs = np.vectorize(lambda ns,nt,np_,sa,t,p : gsw.gibbs(ns,nt,np_,sa,t,p))

gibbs_pt0_pt0 = np.vectorize(lambda sa,pt0 : gsw.gibbs_pt0_pt0(sa,pt0))

hill_ratio_at_sp2 = np.vectorize(lambda t : gsw.hill_ratio_at_sp2(t))

internal_energy = np.vectorize(lambda sa,ct,p : gsw.internal_energy(sa,ct,p))

kappa_t_exact = np.vectorize(lambda sa,t,p : gsw.kappa_t_exact(sa,t,p))

latentheat_evap_ct = np.vectorize(lambda sa,ct : gsw.latentheat_evap_ct(sa,ct))

latentheat_evap_t = np.vectorize(lambda sa,t : gsw.latentheat_evap_t(sa,t))

latentheat_melting = np.vectorize(lambda sa,p : gsw.latentheat_melting(sa,p))

pot_rho_t_exact = np.vectorize(lambda sa,t,p,p_ref : gsw.pot_rho_t_exact(sa,t,p,p_ref))

pt0_from_t = np.vectorize(lambda sa,t,p : gsw.pt0_from_t(sa,t,p))

pt_from_ct = np.vectorize(lambda sa,ct : gsw.pt_from_ct(sa,ct))

pt_from_t = np.vectorize(lambda sa,t,p,p_ref : gsw.pt_from_t(sa,t,p,p_ref))

rho = np.vectorize(lambda sa,ct,p : gsw.rho(sa,ct,p))

rho_t_exact = np.vectorize(lambda sa,t,p : gsw.rho_t_exact(sa,t,p))

saar = np.vectorize(lambda p,lon,lat : gsw.saar(p,lon,lat))

sa_from_sp_baltic = np.vectorize(lambda sp,lon,lat : gsw.sa_from_sp_baltic(sp,lon,lat))

sa_from_sp = np.vectorize(lambda sp,p,lon,lat : gsw.sa_from_sp(sp,p,lon,lat))

sa_from_sstar = np.vectorize(lambda sstar, p, lon, lat : gsw.sa_from_sstar(sstar, p, lon, lat))

sound_speed = np.vectorize(lambda sa,ct,p : gsw.sound_speed(sa,ct,p))

sound_speed_t_exact = np.vectorize(lambda sa,t,p : gsw.sound_speed_t_exact(sa,t,p))

specvol_anom = np.vectorize(lambda sa,ct,p : gsw.specvol_anom(sa,ct,p))

specvol = np.vectorize(lambda sa,ct,p : gsw.specvol(sa,ct,p))

specvol_sso_0_p = np.vectorize(lambda p : gsw.specvol_sso_0_p(p))

specvol_t_exact = np.vectorize(lambda sa,t,p : gsw.specvol_t_exact(sa,t,p))

sp_from_c = np.vectorize(lambda C,t,p : gsw.sp_from_c(C,t,p))

sp_from_sa_baltic = np.vectorize(lambda sa,lon,lat : gsw.sp_from_sa_baltic(sa,lon,lat))

sp_from_sa = np.vectorize(lambda sa,p,lon,lat : gsw.sp_from_sa(sa,p,lon,lat))

sp_from_sr = np.vectorize(lambda sr : gsw.sp_from_sr(sr))

sp_from_sstar = np.vectorize(lambda sstar,p,lon,lat : gsw.sp_from_sstar(sstar,p,lon,lat))

sr_from_sp = np.vectorize(lambda sp : gsw.sr_from_sp(sp))

sstar_from_sa = np.vectorize(lambda sa,p,lon,lat : gsw.sstar_from_sa(sa,p,lon,lat))

sstar_from_sp = np.vectorize(lambda sp,p,lon,lat : gsw.sstar_from_sp(sp,p,lon,lat))

t_freezing = np.vectorize(lambda sa,p,saturation_fraction : gsw.t_freezing(sa,p,saturation_fraction))

t_from_ct = np.vectorize(lambda sa,ct,p : gsw.t_from_ct(sa,ct,p))

ctd_density = np.vectorize(lambda sp, t, p, lat, lon : ctd.density(sp,t,p,lat,lon))
