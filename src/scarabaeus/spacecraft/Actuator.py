# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Spacecraft

#--------------------#
#  Class Definition  #
#--------------------#
class Actuator(Spacecraft):
    """ Any spacecraft component that applies thrust, torque, or other forces.

    Parameters
    ----------
    name : str
        The name of the Actuator object.

    spice_id : int or str, optional
        The NAIF ID code of the object. Defaults to ``None``.

    Notes
    -----
    Actuator is a lightweight subclass of :class:`~scarabaeus.Spacecraft`.
    It inherits all spacecraft state and SPICE-registration methods while
    adding no additional properties, leaving specialisation to concrete
    subclasses (e.g. thrusters).

    See Also
    --------
    scarabaeus.Spacecraft : Parent class providing SPICE registration and mass tracking.
    scarabaeus.Instrument : Sibling subclass for sensing instruments.
    """
    #--------------------#
    # region Constructor #
    #--------------------#
    def __init__(self, name : str, spice_id : int | str = None):
        # parent class initialization
        super().__init__(name, spice_id, tot_mass = None)   
    
    #----------------------#
    # region    Properties #
    #----------------------#
    
    # endregion Properties #
    #----------------------#

    #----------------#
    # region Methods #
    #----------------#