import numpy as np
cimport numpy as np

cimport cython

np.import_array()

cdef extern from "pygsw.h":
    double ctd_density(double SP, double t, double p, double lat, double lon)

@cython.boundscheck(False)
@cython.wraparound(False)
def density(double SP, double t, double p, double lat, double lon):
    """
    Description:
    
        OOI Level 2 Density core data product, which is calculated using the
        Thermodynamic Equations of Seawater - 2010 (TEOS-10) Version 3.0, with
        data from the conductivity, temperature and depth (CTD) family of
        instruments. This calculation is defined in the Data Product
        Specification for Density - DCN 1341-00050.
        
    Implemented by:
    
        2013-03-11: Christopher Mueller. Initial code.
        2013-03-13: Christopher Wingard. Added commenting and moved to
        2013-05-30: Luke Campbell. Ported to C
    Usage:
    
        import pygsw.vectors as gsw

        rho = gsw.ctd_density(SP, t, p, lat, lon)
        
            where
    
        rho = Density (seawater density) [kg m^-3]
        SP = Practical Salinity (PSS-78) [unitless], (see
            1341-00040_Data_Product_Spec_PRACSAL)
        t = temperature (seawater temperature) [deg_C], (see
            1341-00020_Data_Product_Spec_TEMPWAT)
        p = pressure (sea pressure) [dbar], (see
            1341-00020_Data_Product_Spec_PRESWAT)
        lat = latitude where input data was collected [decimal degree]
        lon = longitude where input data was collected [decimal degree]
    
    Example:
        
        SP = 33.4952
        t = 28
        p = 0
        lat = 15.00
        lon = -55.00
        
        rho = ctd_density(SP, t, p, lat, lon)
        print rho
        1021.26508777
        
    References:
    
        OOI (2012). Data Product Specification for Density. Document Control
            Number 1341-00050. https://alfresco.oceanobservatories.org/ (See:
            Company Home >> OOI >> Controlled >> 1000 System Level >>
            1341-00050_Data_Product_SPEC_DENSITY_OOI.pdf)
    """
    
    return ctd_density(SP,t,p,lat,lon)
