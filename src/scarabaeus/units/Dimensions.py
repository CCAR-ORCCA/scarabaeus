# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
import numpy as np
import numpy.typing as npt


# --------------------#
#  Class Definition  #
# --------------------#
class Dimensions:
    """ Defines relationships between fundamental physical quantities. Provides
        dimensional information for the main purpose of creating units.

        These quantities are derived from three of the SI unit system's base
        units [[1]_]: Mass from the kilogram, Length from the meter, and Time from
        the second. Additionaly, a fourth "dimension" — Angle — is included due
        to its relevance to many unitary applications despite the inherent
        non-dimensionality of angles.

        Summarily, these quantities are:

        1. Mass
        2. Length
        3. Time
        4. Angle

        And their power relations are stored in a 1x4 power vector where each
        element corresponds to the above dimensions in the same order.

        Parameters
        ----------
        dim_pwrs : list or numpy.ndarray
            A 1x4 vector of integers representing the power relations to each
            of the four physical dimensions defined by the Dimensions
            class as:

            [Mass, Length, Time, Angle]

            For example, the power vector of force would be
            given as:

            [1, 1, -2, 0]

            Where force is defined as Mass multiplied by Length per
            Time squared. Angles do not contribute to this dimension
            and so are given a zero power.

        See Also
        --------
        scarabaeus.Units       : Utilizes Dimensions in its construction.
        scarabaeus.ArrayWUnits : Dependent on Units and Dimensions.

        Notes
        -----
        The property :attr:`name` is automatically generated upon intialization as a
        string that either provides the name of the dimension if one exists (e.g.
        square length is known as area) or, in the case no name exists, a description
        of the dimensional relations in words. For example, assume that velocity is
        not a named dimension: :attr:`name` would then be assigned ``'Length per Time'`` .

        Dimensions determines if a name exists by mapping the given power vector
        to one in its internal library of named dimensional combinations. The
        method :func:`disp_named_dims` displays a formatted list of all
        existing named dimensions and their respective power vectors.

        Even if a list is passed to ``dim_pwrs`` as an argument during construction,
        Dimensions will automatically convert it to an ``np.array`` when ingesting
        and storing to the property :attr:`powers` . This means that when :attr:`powers` is
        requested, no matter the type passed on construction, a 1x4 ``np.array``
        will always be returned.

        References
        ----------
        .. [1] NIST,
          https://www.nist.gov/pml/owm/metric-si/si-units

        Examples
        --------
        Dimensions objects can be constructed with ``list`` or ``np.array`` objects. 
        The names of Dimensions objects are automatically generated upong construction.

        >>> import scarabaeus as scb
        >>> area_dim = scb.Dimensions([0, 2, 0, 0])             # initialize with a list...
        >>> print(area_dim) # known (named) dimension
        Area
        >>> comp_dim = scb.Dimensions(np.array([3, 0, -2, 1]))  # ... or a numpy array
        >>> print(comp_dim) # unnamed compound dimension
        Cubic Mass times Angle per Square Time
    """

    ## class constants
    # names of each base dimension
    _base_dimensions = ["Mass", "Length", "Time", "Angle"]

    # powers associated with each base dimension
    _base_powers = {
        "Mass": np.array([1, 0, 0, 0]),
        "Length": np.array([0, 1, 0, 0]),
        "Time": np.array([0, 0, 1, 0]),
        "Angle": np.array([0, 0, 0, 1]),
    }

    # base dimension names and, additionaly, commonly occuring named dimension combinations
    _named_dimensions = _base_powers
    _named_dimensions.update(
        {
            "Area": np.array([0, 2, 0, 0]),
            "Volume": np.array([0, 3, 0, 0]),
            "Velocity": np.array([0, 1, -1, 0]),
            "Acceleration": np.array([0, 1, -2, 0]),
            "Force": np.array([1, 1, -2, 0]),
            "Angular Velocity": np.array([0, 0, -1, 1]),
            "Angular Acceleration": np.array([0, 0, -2, 1]),
            "Angular Momentum": np.array([1, 2, -1, 0]),
            "Specific Angular Momentum": np.array([0, 2, -1, 0]),
            "Area to Mass Ratio": np.array([-1, 2, 0, 0]),
            "Gravitational Parameter": np.array([0, 3, -2, 0]),
            "Dimensionless": np.array([0, 0, 0, 0]),
        }
    )

    # region Constructor
    def __init__(self, dim_pwrs: list | np.ndarray):
        # assign properties
        self.powers = dim_pwrs  # input validation and conditioning in property setter
        self._name = None  # name will be created and saved on request

    # region Properties
    @property
    def powers(self) -> np.ndarray:
        """ 1x4 vector describing the Dimension object's power relation to each
            of the four base dimensions.
        """
        return self._powers

    @powers.setter
    def powers(self, input_val):
        ## input validation and conditioning
        # must be a list or numpy array
        if not ((isinstance(input_val, list)) or (isinstance(input_val, np.ndarray))):
            bad_in_err = (
                f"Argument [dim_pwrs] must be of type list or "
                f"np.array. Received: {type(input_val)}."
            )
            raise ValueError(bad_in_err)

        # convert list inputs to numpy arrays -> ensures powers_array always np.array
        if isinstance(input_val, list):
            input_val = np.array(input_val)

        # must be 1x4
        if input_val.shape != (4,):
            bad_shpe_err = (
                f"Argument [dim_pwrs] must be a 1x4 array. "
                f"Received shape of [{input_val.shape}]"
            )
            raise ValueError(bad_shpe_err)

        # elements must be numeric (int or float)
        if not np.issubdtype(input_val.dtype, np.number):
            raise TypeError("The elements in the input powers must be numeric (int or float).")

        # valid input
        self._powers = input_val

    @property
    def name(self) -> str:
        """ The dimension that the Dimensions object represents. For named dimensions
            (e.g. Area or Force), that is their common name. For unnamed dimensions,
            that is their power relation in words (e.g. Length per Time or Mass times Angle).
        """
        # if name exists, return it
        if self._name is not None:
            return self._name

        # otherwise generate and return
        else:
            self._name = self._get_dim_name(self.powers)
            return self._name

    # endregion Properties 

    # region Operators
    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        ## input validation
        if not isinstance(other, self.__class__):
            bad_type_err = (
                "Cannot verify equality of Dimensions object with "
                "non-Dimensions object. Received object of type: "
                f"{type(other)}"
            )
            raise TypeError(bad_type_err)

        # valid input
        if np.array_equal(self.powers, other.powers):
            # powers equal -> return true
            return True
        else:
            # powers not equal -> return false
            return False

    def __ne__(self, other) -> bool:
        # opposite of equality
        return not self.__eq__(other)

    # region Methods
    @classmethod
    def _get_dim_name(cls, in_power: np.ndarray) -> str:
        """ Internal method that returns a name or description of given dimensional array. """
        # first check if the given power exists in the library
        for i, dim in enumerate(cls._named_dimensions.values()):
            if (in_power == dim).all():
                # matches -> return using name from library
                matching_name = list(cls._named_dimensions.keys())[i]
                return matching_name

        # doesn't match -> build a name using the given powers
        # look at each power and create a string description per dimension
        pos_descs = []
        neg_descs = []
        for i, pwr in enumerate(in_power):
            if pwr != 0:
                # power is not zero -> build description
                match abs(pwr):
                    # check any special cases for higher order powers
                    case 1:
                        pwr_order = cls._base_dimensions[i]
                    case 2:
                        pwr_order = f"Square {cls._base_dimensions[i]}"
                    case 3:
                        pwr_order = f"Cubic {cls._base_dimensions[i]}"
                    case _:
                        pwr_order = f"{cls._base_dimensions[i]} to the {abs(pwr)}th"

                # finish dimension description
                if pwr > 0:
                    pos_descs.append(pwr_order)
                else:
                    neg_descs.append(f"per {pwr_order}")
            else:
                # power is zero -> does not need a description
                pass

        # add "times" to separate positive powers
        for i, desc in enumerate(pos_descs):
            if i != len(pos_descs) - 1:
                pos_descs[i] = f"{desc} times"
            else:
                pass
        
        # combine each dimension's description into one string and return
        return " ".join(pos_descs + neg_descs)

    @classmethod
    def def_new_named_dim(cls, name: str, powers: np.ndarray | list) -> None:
        """ Adds a new named dimension to the Dimensions library.

            Parameters
            ----------
            name : str
                The name of the new dimension to be saved in the Dimensions library.

            powers : numpy array or list
                The dimensional powers corresponding to the given name.

            Returns
            -------
            None

            Notes
            -----
            When defining a new named dimension, remember to follow the naming convention
            adopted by the class' built-in named dimensions: capitalize the first letter of
            each word in name.

            Examples
            --------
            >>> import scarabaeus as scb
            >>> unnamed = scb.Dimensions([1 ,1, 1, 1])
            >>> print(unnamed)  # unrecognized dimension
            Mass times Length times Time times Angle
            >>> scb.Dimensions.def_new_named_dim('My New Dim', [1, 1, 1, 1])   # add to dim library
            >>> new_dim = scb.Dimensions([1 ,1, 1, 1])
            >>> print(new_dim)  # will register as the name we gave it
            My New Dim
        """
        ## input validation
        # must be a list or numpy array
        if not ((isinstance(powers, list)) or (isinstance(powers, np.ndarray))):
            bad_in_err = (
                f"Argument [dim_pwrs] must be of type list or "
                f"np.array. Received: {type(powers)}."
            )
            raise ValueError(bad_in_err)

        # convert list inputs to numpy arrays; ensures powers_array always np.array
        if isinstance(powers, list):
            powers = np.array(powers)

        # must be 1x4
        if powers.shape != (4,):
            bad_shpe_err = (
                f"Argument [dim_pwrs] must be a 1x4 array. "
                f"Received shape of [{powers.shape}]"
            )
            raise ValueError(bad_shpe_err)

        # elements must be all integers
        if not np.issubdtype(powers.dtype, np.integer):
            raise TypeError("The elements in the input powers must be of type int.")

        # powers can't already exist
        for i, dim in enumerate(cls._named_dimensions.values()):
            if (powers == dim).all():
                exists_err = (
                    "Argument [powers] invalid. Equivalent dimensional array "
                    "already assigned to existing named dimension."
                )
                raise ValueError(exists_err)

        # name can't already exist
        if name in cls._named_dimensions.keys():
            exists_err = (
                "Argument [name] invalid. Requested name already "
                "assigned to existing named dimension."
            )
            raise ValueError(exists_err)

        # add to named dimensions library
        cls._named_dimensions.update({name: powers})

    @classmethod
    def disp_named_dims(cls) -> None:
        """ Displays all existing named dimensions and their respective dimensional power vectors.

            Includes base dimensions.

            Returns
            -------
            None
        """
        # display header
        head_str = f'|{" " * 39}List of Named Dimensions{" " * 39}|'
        print(f'|{"=" * (len(head_str) - 2)}|')
        print(head_str)
        print(f'|{"=" * (len(head_str) - 2)}|')

        # loop through all named dimensions and format their information accordingly
        for i in range(0, len(cls._named_dimensions.values()), 2):
            # get first column information
            first_name = f"Name  : {list(cls._named_dimensions.keys())[i]}"
            first_pwrs = f"Powers: {list(cls._named_dimensions.values())[i]}"

            # second column information
            try:
                secnd_name = list(cls._named_dimensions.keys())[i + 1]
            except:
                # can't index i+1, set to nothing
                secnd_name = ""
                secnd_pwrs = ""
            else:
                # can index i+1, set to values of i+1
                secnd_name = f"Name  : {list(cls._named_dimensions.keys())[i + 1]}"
                secnd_pwrs = f"Powers: {list(cls._named_dimensions.values())[i + 1]}"

            # display column or columns accordingly
            print(
                f'| {first_name}"{" " * (int(len(head_str) / 2) - len(first_name) - 1)}| '
                f'{secnd_name}"{" " * (int(len(head_str) / 2) - len(secnd_name) - 6)}|'
            )
            print(
                f'| {first_pwrs}{" " * (int(len(head_str) / 2) - len(str(first_pwrs)))}| '
                f'{secnd_pwrs}{" " * (int(len(head_str) / 2) - len(str(secnd_pwrs)) - 5)}|'
            )
            if i < len(cls._named_dimensions.values()) - 2:
                print(f'|{"-" * (len(head_str) - 2)}|')

        # close table
        print(f'|{"=" * (len(head_str) - 2)}|')
