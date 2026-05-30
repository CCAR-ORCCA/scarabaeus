# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Instrument, ArrayWFrame

#--------------------#
#  Class Definition  #
#--------------------#
class Antenna(Instrument):
    """ Defines an antenna for receiving and transmitting radiometric data for navigation.

    The turn-around ratio :math:`M_2` is the carrier frequency multiplier
    applied by the spacecraft transponder.  For the Doppler observable the
    transponder receives the uplink at :math:`f_U` and retransmits at
    :math:`f_D = M_2\\,f_U`.  The default X-band value is

    .. math::

        M_2 = \\frac{880}{749} \\approx 1.1749

    Parameters
    ----------
    name : str
        The name of the Antenna object.

    turn_around_ratio : float, optional
        Transponder ratio between received and transmitted frequencies. Defaults to Xband value from Moyer: ``880.0 / 749.0``.

    pos_craft_cntrd : ArrayWFrame, optional
        Antenna shift position-vector. Defaults to ``None``.

    spice_id : int or str, optional
        The NAIF ID code of the object. May be given as an integer ID or as a string. Defaults to ``None``.

    See Also
    --------
    scarabaeus.Instrument : Parent class for all spacecraft instruments.
    scarabaeus.DopplerIdeal : Uses turn_ratio to model the Doppler observable.
    scarabaeus.DopplerReal : Uses turn_ratio in the RTLT Doppler formulation.
    """
    #-------------#
    # Constructor #
    #-------------#
    def __init__(self,
                 name : str,
                 turn_ratio : float = 880.0/749.0,
                 pos_craft_centrd : ArrayWFrame = None,
                 spice_id : int | str = None):
        # parent class initialization
        super().__init__(name, spice_id, measurement_type = None)

        # rest of the class properties
        self._turn_ratio = turn_ratio
    
    #----------------------#
    # region    Properties #
    #----------------------#
    @property
    def turn_ratio(self) -> float:
        """Turn-around ratio between transmitted and received frequency at the transponder."""
        return self._turn_ratio
    
    # endregion Properties #
    #----------------------#

    #----------------#
    # region Methods #
    #----------------#