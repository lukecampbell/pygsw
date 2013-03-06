import numpy as np
cimport numpy as np

np.import_array()
a = (0.0080, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081)

b = (0.0005, -0.0056, -0.0066, -0.0375, 0.0636, -0.0144)

c = (0.6766097, 2.00564e-2, 1.104259e-4, -6.9698e-7, 1.0031e-9)

d = (3.426e-2, 4.464e-4, 4.215e-1, -3.107e-3)

e = (2.070e-5, -6.370e-10, 3.989e-15)

P = (4.577801212923119e-3, 1.924049429136640e-1, 2.183871685127932e-5,
    -7.292156330457999e-3, 1.568129536470258e-4, -1.478995271680869e-6,
    9.086442524716395e-4, -1.949560839540487e-5, -3.223058111118377e-6,
    1.175871639741131e-7, -7.522895856600089e-5, -2.254458513439107e-6,
    6.179992190192848e-7, 1.005054226996868e-8, -1.923745566122602e-9,
    2.259550611212616e-6, 1.631749165091437e-7, -5.931857989915256e-9,
    -4.693392029005252e-9, 2.571854839274148e-10, 4.198786822861038e-12)

q = (5.540896868127855e-5, 2.015419291097848e-1, -1.445310045430192e-5,
    -1.567047628411722e-2, 2.464756294660119e-4, -2.575458304732166e-7,
    5.071449842454419e-3, -9.081985795339206e-5, -3.635420818812898e-6,
    2.249490528450555e-8, -1.143810377431888e-3, 2.066112484281530e-5,
    7.482907137737503e-7, 4.019321577844724e-8, -5.755568141370501e-10,
    1.120748754429459e-4, -2.420274029674485e-6, -4.774829347564670e-8,
    -4.279037686797859e-9, -2.045829202713288e-10, 5.025109163112005e-12)

r = (3.432285006604888e-3, 1.672940491817403e-1, 2.640304401023995e-5,
    1.082267090441036e-1, -6.296778883666940e-5, -4.542775152303671e-7,
    -1.859711038699727e-1, 7.659006320303959e-4, -4.794661268817618e-7,
    8.093368602891911e-9, 1.001140606840692e-1, -1.038712945546608e-3,
    -6.227915160991074e-6, 2.798564479737090e-8, -1.343623657549961e-10,
    1.024345179842964e-2, 4.981135430579384e-4, 4.466087528793912e-6,
    1.960872795577774e-8, -2.723159418888634e-10, 1.122200786423241e-12)

u = (5.180529787390576e-3, 1.052097167201052e-3, 3.666193708310848e-5,
    7.112223828976632, -3.631366777096209e-4, -7.336295318742821e-7,
    -1.576886793288888e+2, -1.840239113483083e-3, 8.624279120240952e-6,
    1.233529799729501e-8, 1.826482800939545e+3, 1.633903983457674e-1,
    -9.201096427222349e-5, -9.187900959754842e-8, -1.442010369809705e-10,
    -8.542357182595853e+3, -1.408635241899082, 1.660164829963661e-4,
    6.797409608973845e-7, 3.345074990451475e-10, 8.285687652694768e-13)

k = 0.0162
def Hill_ratio_at_SP2(double t):
    r"""TODO: Write docstring
    Hill ratio at SP = 2
    """

    # USAGE:
    #  Hill_ratio = Hill_ratio_at_SP2(t)
    #
    # DESCRIPTION:
    #  Calculates the Hill ratio, which is the adjustment needed to apply for
    #  Practical Salinities smaller than 2.  This ratio is defined at a
    #  Practical Salinity = 2 and in-situ temperature, t using PSS-78. The Hill
    #  ratio is the ratio of 2 to the output of the Hill et al. (1986) formula
    #  for Practical Salinity at the conductivity ratio, Rt, at which Practical
    #  Salinity on the PSS-78 scale is exactly 2.
    #
    # INPUT:
    #  t  =  in-situ temperature (ITS-90)                  [ deg C ]
    #
    # OUTPUT:
    #  Hill_ratio  =  Hill ratio at SP of 2                [ unitless ]
    #
    # AUTHOR:
    #  Trevor McDougall and Paul Barker
    #
    # VERSION NUMBER: 3.0 (26th March, 2011)

    SP2 = 2 * np.ones_like(t)
    SP2 = 2

    #------------------------------
    # Start of the calculation
    #------------------------------

    a0 = 0.0080
    a1 = -0.1692
    a2 = 25.3851
    a3 = 14.0941
    a4 = -7.0261
    a5 = 2.7081

    b0 = 0.0005
    b1 = -0.0056
    b2 = -0.0066
    b3 = -0.0375
    b4 = 0.0636
    b5 = -0.0144

    g0 = 2.641463563366498e-1
    g1 = 2.007883247811176e-4
    g2 = -4.107694432853053e-6
    g3 = 8.401670882091225e-8
    g4 = -1.711392021989210e-9
    g5 = 3.374193893377380e-11
    g6 = -5.923731174730784e-13
    g7 = 8.057771569962299e-15
    g8 = -7.054313817447962e-17
    g9 = 2.859992717347235e-19

    k = 0.0162

    t68 = t * 1.00024
    ft68 = (t68 - 15) / (1 + k * (t68 - 15))

    #--------------------------------------------------------------------------
    # Find the initial estimates of Rtx (Rtx0) and of the derivative dSP_dRtx
    # at SP = 2.
    #--------------------------------------------------------------------------

    Rtx0 = g0 + t68 * (g1 + t68 * (g2 + t68 * (g3 + t68 * (g4 + t68 * (g5
              + t68 * (g6 + t68 * (g7 + t68 * (g8 + t68 * g9))))))))

    dSP_dRtx = (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * Rtx0) * Rtx0) *
    Rtx0) * Rtx0 + ft68 * (b1 + (2 * b2 + (3 * b3 + (4 * b4 + 5 * b5 * Rtx0) *
    Rtx0) * Rtx0) * Rtx0))

    #--------------------------------------------------------------------------
    # Begin a single modified Newton-Raphson iteration to find Rt at SP = 2.
    #--------------------------------------------------------------------------

    SP_est = (a0 + (a1 + (a2 + (a3 + (a4 + a5 * Rtx0) * Rtx0) * Rtx0) * Rtx0) *
    Rtx0 + ft68 * (b0 + (b1 + (b2 + (b3 + (b4 + b5 * Rtx0) * Rtx0) * Rtx0) *
    Rtx0) * Rtx0))

    Rtx = Rtx0 - (SP_est - SP2) / dSP_dRtx
    Rtxm = 0.5 * (Rtx + Rtx0)
    dSP_dRtx = (a1 + (2 * a2 + (3 * a3 + (4 * a4 + 5 * a5 * Rtxm) * Rtxm) *
    Rtxm) * Rtxm + ft68 * (b1 + (2 * b2 + (3 * b3 + (4 * b4 + 5 * b5 * Rtxm) *
    Rtxm) * Rtxm) * Rtxm))

    Rtx = Rtx0 - (SP_est - SP2) / dSP_dRtx

    # This is the end of one full iteration of the modified Newton-Raphson
    # iterative equation solver.  The error in Rtx at this point is equivalent
    # to an error in SP of 9e-16 psu.

    x = 400 * Rtx * Rtx
    sqrty = 10 * Rtx
    part1 = 1 + x * (1.5 + x)
    part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
    SP_Hill_raw_at_SP2 = SP2 - a0 / part1 - b0 * ft68 / part2

    return 2. / SP_Hill_raw_at_SP2



def SP_from_C(double C,double t,double p):
    r"""Calculates Practical Salinity, SP, from conductivity, C, primarily
    using the PSS-78 algorithm.  Note that the PSS-78 algorithm for Practical
    Salinity is only valid in the range 2 < SP < 42.  If the PSS-78 algorithm
    produces a Practical Salinity that is less than 2 then the Practical
    Salinity is recalculated with a modified form of the Hill et al. (1986)
    formula. The modification of the Hill et al. (1986) expression is to ensure
    that it is exactly consistent with PSS-78 at SP = 2.  Note that the input
    values of conductivity need to be in units of mS/cm (not S/m).

    Parameters
    ----------
    C : array
        conductivity [mS cm :sup:`-1`]
    t : array
        in-situ temperature [:math:`^\circ` C (ITS-90)]
    p : array
        sea pressure [dbar]
        (i.e. absolute pressure - 10.1325 dbar)

    Returns
    -------
    SP : array
         Practical Salinity [psu (PSS-78), unitless]

    Examples
    --------
    TODO

    See Also
    --------
    TODO

    Notes
    -----
    TODO

    References
    ----------
    .. [1] Culkin and Smith, 1980:  Determination of the Concentration of
    Potassium Chloride Solution Having the Same Electrical Conductivity, at
    15C and Infinite Frequency, as Standard Seawater of Salinity 35.0000
    (Chlorinity 19.37394), IEEE J. Oceanic Eng, 5, 22-23.

    .. [2] Hill, K.D., T.M. Dauphinee & D.J. Woods, 1986: The extension of the
    Practical Salinity Scale 1978 to low salinities. IEEE J. Oceanic Eng., 11,
    109 - 112.

    .. [3] IOC, SCOR and IAPSO, 2010: The international thermodynamic equation
    of seawater - 2010: Calculation and use of thermodynamic properties.
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO (English), 196 pp.  Appendix E.

    .. [4] Unesco, 1983: Algorithms for computation of fundamental properties
    of seawater.  Unesco Technical Papers in Marine Science, 44, 53 pp.

    Modifications:
    2011-04-01. Paul Barker, Trevor McDougall and Rich Pawlowicz.
    """


    t68 = t * 1.00024
    ft68 = (t68 - 15) / (1 + k * (t68 - 15))

    # The dimensionless conductivity ratio, R, is the conductivity input, C,
    # divided by the present estimate of C(SP=35, t_68=15, p=0) which is
    # 42.9140 mS/cm (=4.29140 S/m), (Culkin and Smith, 1980).

    R = 0.023302418791070513 * C  # 0.023302418791070513 = 1./42.9140

    # rt_lc corresponds to rt as defined in the UNESCO 44 (1983) routines.
    rt_lc = c[0] + (c[1] + (c[2] + (c[3] + c[4] * t68) * t68) * t68) * t68
    Rp = (1 + (p * (e[0] + e[1] * p + e[2] * p ** 2)) /
         (1 + d[0] * t68 + d[1] * t68 ** 2 + (d[2] + d[3] * t68) * R))
    Rt = R / (Rp * rt_lc)

    if Rt < 0:
        Rt = np.nan
    Rtx = np.sqrt(Rt)

    SP = a[0] + (a[1] + (a[2] + (a[3] + (a[4] + a[5] * Rtx) * Rtx) * Rtx) *
         Rtx) * Rtx + ft68 * (b[0] + (b[1] + (b[2] + (b[3] + (b[4] + b[5] *
         Rtx) * Rtx) * Rtx) * Rtx) * Rtx)

    # The following section of the code is designed for SP < 2 based on the
    # Hill et al. (1986) algorithm.  This algorithm is adjusted so that it is
    # exactly equal to the PSS-78 algorithm at SP = 2.

    if SP < 2:
        Hill_ratio = Hill_ratio_at_SP2(t)
        x = 400 * Rt
        sqrty = 10 * Rtx
        part1 = 1 + x * (1.5 + x)
        part2 = 1 + sqrty * (1 + sqrty * (1 + sqrty))
        SP_Hill_raw = SP - a[0] / part1 - b[0] * ft68 / part2
        SP = Hill_ratio * SP_Hill_raw

    SP = np.maximum(SP, 0)  # Ensure that SP is non-negative.

    return SP

