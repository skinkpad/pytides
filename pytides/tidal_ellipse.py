"""
Python version of Zhigang Xu's MATLAB code, which can be obtained from

https://www.mathworks.com/matlabcentral/fileexchange/347-tidal_ellipse

or, indirectly, from

https://sea-mat.github.io/sea-mat/

The programs convert to and from linear and elliptical tidal parameters.

Documentation is available from

https://www.researchgate.net/publication/260427282_Ellipse_Parameters_Conversion_and_Vertical_Profiles_for_Tidal_Currents_in_Matlab

#Authorship Copyright:
#
#    The author of this program retains the copyright of this program, while
# you are welcome to use and distribute this program as long as you credit
# the author properly and respect the program name itself. Particularly,
# you are expected to retain the original author's name in this original
# version of the program or any of its modified version that you might make.
# You are also expected not to essentially change the name of the programs
# except for adding possible extension for your own version you might create,
# e.g. app2ep_xx is acceptable.  Any suggestions are welcome and enjoy my
# program(s)!
#
#
#Author Info:
#_______________________________________________________________________
#  Zhigang Xu, Ph.D.
#  (pronounced as Tsi Gahng Hsu)
#  Research Scientist
#  Coastal Circulation
#  Bedford Institute of Oceanography
#  1 Challenge Dr.
#  P.O. Box 1006                    Phone  (902) 426-2307 (o)
#  Dartmouth, Nova Scotia           Fax    (902) 426-7827
#  CANADA B2Y 4A2                   email xuz@dfo-mpo.gc.ca
#_______________________________________________________________________
#
#Release Date: Nov. 2000

"""

from math import pi
import numpy as np


def angle(z):
    """Equivalent of MATLAB angle function."""
    return np.arctan2(z.imag, z.real)


def ap2ep(Au, PHIu, Av, PHIv, return_full=False):
    """
    Convert tidal amplitude and phase lag (ap-) parameters into tidal ellipse
    (ep-) parameters. Please refer to ep2app for its inverse function.

    Usage:

    SEMA,  ECC, INC, PHA = ap2ep(Au, PHIu, Av, PHIv)

    where:

    Au, PHIu, Av, PHIv are the amplitudes and phase lags (in degrees) of
    u- and v- tidal current components. They can be vectors or
    matrices or multidimensional arrays.

    Any number of dimensions are allowed as long as your computer
    resource can handle.

    Parameters
    ----------

    SEMA : array_like or float
        Semi-major axes, or the maximum speed
    ECC :  array_like or float
        Eccentricity, the ratio of semi-minor axis over
       the semi-major axis; its negative value indicates that the ellipse
       is traversed in clockwise direction.
    INC :  array_like or float
        Inclination, the angles (in degrees) between the semi-major
        axes and u-axis.
    PHA :  array_like or float
        Phase angles, the time (in angles and in degrees) when the
        tidal currents reach their maximum speeds,  (i.e.
        PHA=omega*tmax).

       These four ep-parameters will have the same dimensionality
       (i.e., vectors, or matrices) as the input ap-parameters.

    return_full : bool
        If True, return SEMA,  ECC, INC, PHA, Wp, THETAp, wp, Wm, THETAm

    Document:   tidal_ellipse.ps

    Revisions: May  2002, by Zhigang Xu,  --- adopting Foreman's northern
    semi major axis convention.

    For a given ellipse, its semi-major axis is undetermined by 180. If we borrow
    Foreman's terminology to call a semi major axis whose direction lies in a range of
    [0, 180) as the northern semi-major axis and otherwise as a southern semi major
    axis, one has freedom to pick up either northern or southern one as the semi major
    axis without affecting anything else. Foreman (1977) resolves the ambiguity by
    always taking the northern one as the semi-major axis. This revision is made to
    adopt Foreman's convention. Note the definition of the phase, PHA, is still
    defined as the angle between the initial current vector, but when converted into
    the maximum current time, it may not give the time when the maximum current first
    happens; it may give the second time that the current reaches the maximum
    (obviously, the 1st and 2nd maximum current times are half tidal period apart)
    depending on where the initial current vector happen to be and its rotating sense.

    Version 2, May 2002
    """

    # Assume the input phase lags are in degrees and convert them in radians.

    PHIu = PHIu / 180 * pi
    PHIv = PHIv / 180 * pi

    # Make complex amplitudes for u and v

    u = Au * np.exp(-1j * PHIu)
    v = Av * np.exp(-1j * PHIv)

    # Calculate complex radius of anticlockwise and clockwise circles:

    wp = (u + 1j * v) / 2  # for anticlockwise circles
    wm = conj(u - 1j * v) / 2  # for clockwise circles
    # and their amplitudes and angles
    Wp = np.abs(wp)
    Wm = np.abs(wm)
    THETAp = angle(wp)
    THETAm = angle(wm)

    # calculate ep-parameters (ellipse parameters)

    SEMA = Wp + Wm  # Semi  Major Axis, or maximum speed
    SEMI = Wp - Wm  # Semin Minor Axis, or minimum speed
    ECC = SEMI / SEMA  # Eccentricity

    PHA = (THETAm - THETAp) / 2  # Phase angle, the time (in angle) when
    # the velocity reaches the maximum
    INC = (THETAm + THETAp) / 2  # Inclination, the angle between the
    # semi major axis and x-axis (or u-axis).

    # convert to degrees for output

    PHA = PHA / pi * 180
    INC = INC / pi * 180
    THETAp = THETAp / pi * 180
    THETAm = THETAm / pi * 180

    #map the resultant angles to the range of [0, 360].
    # Note that the numpy equivalent of MATLAB mod is remainder
    # https://numpy.org/doc/stable/reference/generated/numpy.remainder.html
    #PHA=mod(PHA+360, 360)
    #INC=mod(INC+360, 360)

    PHA = np.remainder(PHA + 360, 360)
    INC = np.remainder(INC + 360, 360)

    # Mar. 2, 2002 Revision by Zhigang Xu    (REVISION_1)
    # Change the southern major axes to northern major axes to conform the tidal
    # analysis convention  (cf. Foreman, 1977, p. 13, Manual For Tidal Currents
    # Analysis Prediction, available in www.ios.bc.ca/ios/osap/people/foreman.htm)

    k = np.fix(INC / 180)
    INC = INC - k * 180
    PHA = PHA + k * 180
    #PHA = mod(PHA, 360)
    PHA = np.remainder(PHA, 360)

    if return_full:
        return SEMA, ECC, INC, PHA, Wp, THETAp, wp, Wm, THETAm

    return SEMA, ECC, INC, PHA


def ep2ap(SEMA, INC, PHA, ECC=None, SEMI=None, return_full=False):
    """
    Convert tidal ellipse parameters into amplitude and phase lag parameters.
    Its inverse is app2ep.m. Please refer to app2ep for the meaning of the
    inputs and outputs.

    Note the order of the parameters is changed from MATLAB code and optionally
    the semiminor axis can be specified.

    Parameters
    ----------
    SEMA : array_like or float
        Semimajor axis
    INC : array_like or float
        Ellipse inclination in degrees [0,180]
    PHA : array_like or float
        Phase in degrees
    ECC : array_like or float or None
        eccentricity. Not necessary if SEMI is given
    SEMI : array_like or float or None
        Semiminor axis must be given if ECC not specified.

    Returns
    -------
    Au, PHIu, Av, PHIv

    if return_full is True, the following additional parameters are returned
        Wp, THETAp, wp, Wm, THETAm, wm


    Zhigang Xu
    Oct. 20, 2000

    Document:  tidal_ellipse.ps
    """
    if ECC is None:
        if SEMI is None:
            raise Exception('SEMI must be given if ECC is None')
        ECC = SEMI / SEMA

    Wp = (1 + ECC) / 2 * SEMA
    Wm = (1 - ECC) / 2 * SEMA
    THETAp = INC - PHA
    THETAm = INC + PHA

    # convert degrees into radians

    THETAp = THETAp / 180 * pi
    THETAm = THETAm / 180 * pi

    # Calculate wp and wm.

    wp = Wp * np.exp(1j * THETAp)
    wm = Wm * np.exp(1j * THETAm)

    # Calculate cAu, cAv --- complex amplitude of u and v

    cAu = wp + np.conj(wm)
    cAv = -1j * (wp - np.conj(wm))
    Au = np.abs(cAu)
    Av = np.abs(cAv)
    PHIu = -angle(cAu) * 180 / pi
    PHIv = -angle(cAv) * 180 / pi

    # flip angles in the range of [-180 0) to the range of [180 360).

    PHIu = np.where(PHIu < 0, PHIu + 360, PHIu)
    PHIv = np.where(PHIv < 0, PHIv + 360, PHIv)

    if return_full:
        return Au, PHIu, Av, PHIv, Wp, THETAp, wp, Wm, THETAm, wm

    return Au, PHIu, Av, PHIv
