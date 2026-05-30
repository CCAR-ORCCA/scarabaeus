# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Units, ArrayWUnits, Frame, EpochArray

import scarabaeus.utils.NumpyWrapper as np

import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# class definition
class ArrayWFrame():
    """ Defines a a value or values with associated units and reference frame.

        Accepts two calling conventions:

        * ``ArrayWFrame(awu, frame)`` — wrap an existing ArrayWUnits in a frame.
        * ``ArrayWFrame(array, units, frame)`` — wrap a raw array with units and frame.

        Parameters
        ----------
        array : ArrayWUnits
            Value or values with associated units to be defined in the 
            given reference frame.

        units_or_frame : Units or Frame
            Units or Frame objects to be linked with the object.

        frame : Frame, optional
            Frame object to be linked with the ArrayWFrame. 
            Defaults to ``None``.

        See Also
        --------
        scarabaeus.Frame : define a Frame object that carry information about a SPICE-defined frame with origin and orientation
        scarabaeus.Units : define Units to be attached to numpy arrays
        scarabaeus.ArrayWUnits : define an ArrayWUnits that carry information about numpy arrays and units linked to their content

        Examples
        --------
        Create an ArrayWFrame by assigning a Frame to an ArrayWUnits:

        >>> import scarabaeus as scb
        >>> from pathlib import Path
        >>> mk_path = Path('path/to/example_mk.tm')
        >>> scb.SpiceManager.load_kernel_from_mkfile(str(mk_path))  # furnish metakernel
        >>> km = scb.Units.get_units('km')  # define units
        >>> J2000 = scb.Frame('J2000')  # define reference frame
        >>> pos_vec = scb.ArrayWUnits([1, 2, 3], km)    # define position
        >>> pos_J2000 = scb.ArrayWFrame(pos_vec, J2000) # define position in J2000
    """

    # region Constructor
    def __init__(self, array : ArrayWUnits | np.ndarray | float, units_or_frame : Units | Frame, frame : Frame = None):
        # Get list of common frames
        from scarabaeus import SpiceManager
        common_frames = SpiceManager.list_all_frames()

        # Check if frame is None
        if frame is None:
            if not isinstance(array, (ArrayWUnits, np.ndarray, float)):
                raise TypeError(
                    "array must be an ArrayWUnits or a numpy array when frame is not specified."
                )

            if isinstance(units_or_frame, Frame):
                if units_or_frame.name not in common_frames:
                    raise ValueError(
                        f"'{units_or_frame.name}' is not in the list of recognized frames. "
                        "Please create a new frame."
                    )
                self._quantity = array  # array is expected to be an ArrayWUnits
                self._frame = units_or_frame
            else:  # units_or_frame is an ArrayWUnits
                if isinstance(array, ArrayWUnits):
                    self._quantity = array
                else:
                    self._quantity = ArrayWUnits(
                        array, units_or_frame
                    )  # It's always a scalar!
                self._frame = None  # Explicitly assign None to _frame
        else:
            if not isinstance(frame, Frame):
                raise TypeError("frame must be a Frame object.")
            if not isinstance(units_or_frame, Units):
                raise TypeError(
                    "units_or_frame must be an ArrayWUnits object when frame is specified."
                )
            if frame.name not in common_frames:
                raise ValueError(
                    f"'{frame.name}' is not in the list of recognized frames. "
                    "Please create a new frame."
                )
            self._quantity = ArrayWUnits(array, units_or_frame)
            self._frame = frame

        # Get the shape of the quantity
        self._shape = self.quantity.shape

    # region Properties
    @property
    def quantity(self) -> ArrayWUnits:
        """ The quantity or quantities linked to the Frame. """
        return self._quantity

    @property
    def frame(self) -> Frame:
        """ The associated frame of the ArrayWFrame. """
        return self._frame

    @property
    def shape(self) -> np.ndarray:
        """ The shape of the ArrayWFrame. """
        return self._shape

    # endregion Properties

    # region Operators
    def __repr__(self):
        return "{0} ({1})".format(self._quantity, self._frame)

    def __operator_frame_check(self, other : Self):
        """ Internal method that ensures frames are compatible before 
            performing operations.
        """
        if self.frame != other.frame:
            incomp_frm_err = (f'Frame {self.frame} incompatible for mathematical operations with frame ', 
                              f'{other.frame}.')
            raise ValueError(incomp_frm_err)
        
    def __setitem__(self, index, item):
        """ Set a sub-array in the ArrayWFrame object. """
        # make sure given item can be set
        if isinstance(item, ArrayWFrame):
            if self.frame != item.frame:
                # frames must be the same
                bad_frm_err = (f'Cannot assign ArrayWFrame with associated frame {item.frame} ',
                               f'to ArrayWFrame with associated frame {self.frame}.')
                raise ValueError(bad_frm_err)
        else:
            # can't set non-AWF
            not_awf_err = (f'Cannot assign object of type {type(item)} to index within '
                            'ArrayWFrame.')
            raise TypeError(not_awf_err)

        # set the item in the given index
        self._quantity[index] = item.quantity
        
    def __getitem__(self, index):
          return ArrayWFrame(self.quantity[index], self.frame)

    def __add__(self, other : Self):
        # can only add AWF with AWF
        if not isinstance(other, ArrayWFrame):
            not_awf_err = ('Cannot add ArrayWFrame with object of type ',
                           f'{type(other)}.')
            raise TypeError(not_awf_err)

        # frames must be compatible
        self.__operator_frame_check(other)

        # perform operations on AWUs
        array = self.quantity + other.quantity

        # return as an AWF
        return ArrayWFrame(array, self.frame)

    def __neg__(self) -> Self:
        return ArrayWFrame(-self.quantity, self.frame)

    def __pow__(self, other) -> Self:
        # can only raise power for integers or floats
        if not isinstance(other, int): 
            if not isinstance(other, float):
                cant_exp_err = (f'Cannot raise ArrayWFrame to an object of type {type(other)}. '
                                'Exponentiation is only valid for integers or floats.')
                raise TypeError(cant_exp_err)

        # exponentiate the AWU and return as AWF
        return ArrayWFrame(self.quantity**other, self.frame)

    def __mul__(self, other : int | float | Self):
        # different logic if depending on what the AWF is being multiplied by
        if isinstance(other, ArrayWFrame):
            # frames must be compatible
            self.__operator_frame_check(other)

            # multiply the AWUs
            array = self.quantity.__mul__(other.quantity)
        
        elif isinstance(other, float) or isinstance(other, int):
            # multiply the AWU by the given value
            array = self.quantity.__mul__(other)
        
        else:
            cant_mul_err = (f'Cannot multiply ArrayWFrame by object of type {type(other)}. '
                             'Multiplication is only valid by an integer, float, numpy array ',
                             'of equal size, or another ArrayWFrame associated with the same Frame.')
            raise TypeError(cant_mul_err)

        # return the product as an AWF
        return ArrayWFrame(array, self.frame)

    def __matmul__(self, other : Self):
        if isinstance(other, ArrayWFrame):
            # frames must be compatible
            self.__operator_frame_check(other)

            array =  self.quantity @ other.quantity
        
        else:
            cant_mul_err = ( 'Cannot perform matrix multiplication of ArrayWFrame by ',
                            f'object of type {type(other)}. Matrix multiplication is ',
                             'only valid between another ArrayWFrame associated with the ',
                             'same Frame.')
            raise TypeError(cant_mul_err)
        
        # return as an AWF
        return ArrayWFrame(array, self.frame)

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return other * (self**-1)
    
    def __eq__(self, other : Self):
        # can only equate AWF with AWF
        if not isinstance(other, ArrayWFrame):
            not_awf_err = ('Cannot equate ArrayWFrame with object of type ',
                           f'{type(other)}.')
            raise TypeError(not_awf_err)

        # frames must be compatible
        self.__operator_frame_check(other)

        # perform operations on AWU
        return (self.quantity == other.quantity)
    
    def __ne__(self, other):
        # re-use equality logic
        equality_result = self.__eq__(other)

        # invert
        return np.invert(equality_result)

    def __gt__(self, other):
        # can only compare AWF with AWF
        if not isinstance(other, ArrayWFrame):
            not_awf_err = ('Cannot compare ArrayWFrame with object of type ',
                           f'{type(other)}.')
            raise TypeError(not_awf_err)

        # frames must be compatible
        self.__operator_frame_check(other)

        # perform operations on AWU
        return (self.quantity > other.quantity)

    def __ge__(self, other):
        # can only compare AWF with AWF
        if not isinstance(other, ArrayWFrame):
            not_awf_err = ('Cannot compare ArrayWFrame with object of type ',
                           f'{type(other)}.')
            raise TypeError(not_awf_err)

        # frames must be compatible
        self.__operator_frame_check(other)

        # perform operations on AWU
        return (self.quantity >= other.quantity)

    def __lt__(self, other):
        # can only compare AWF with AWF
        if not isinstance(other, ArrayWFrame):
            not_awf_err = ('Cannot compare ArrayWFrame with object of type ',
                           f'{type(other)}.')
            raise TypeError(not_awf_err)

        # frames must be compatible
        self.__operator_frame_check(other)

        # perform operations on AWU
        return (self.quantity < other.quantity)

    def __le__(self, other):
        # can only compare AWF with AWF
        if not isinstance(other, ArrayWFrame):
            not_awf_err = ('Cannot compare ArrayWFrame with object of type ',
                           f'{type(other)}.')
            raise TypeError(not_awf_err)

        # frames must be compatible
        self.__operator_frame_check(other)

        # perform operations on AWU
        return (self.quantity <= other.quantity)

    def cross(self, other : Self) -> Self:
        """ Cross product operation between two ArrayWFrame objects. """
        # can only cross with other AWFs
        if not isinstance(self, ArrayWFrame):
            cant_cross_err = ( 'Cannot perform cross product between ArrayWFrame and ',
                              f'{type(other)} object. Cross product is only defined '
                               'between two ArrayWFrame objects.')
            raise TypeError(cant_cross_err)

        # frames must be compatible
        self.__operator_frame_check(other)

        # preform operation on AWUs and return as AWF
        return ArrayWFrame(self.quantity.cross(other.quantity), self.frame)

    def summation(self) -> Self:
        """ Sum all elements in the ArrayWFrame object, if units allow. """
        # perform operation on AWU
        array = self.quantity.summation()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    def norm(self) -> Self:
        """ Matrix or vector norm operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.norm()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    def unitary(self) -> Self:
        """ Converts a given ArrayWFrame vector into a unit vector with the same units.

            Only valid for vectorial ArrayWFrame.
        """
        # perform operation on AWU
        array = self.quantity.unitary()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    def transpose(self) -> Self:
        """ Transpose operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.transpose()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    T = transpose
    """ Same as self.transpose()

        See Also
        --------
        scarabaeus.ArrayWFrame.transpose
    """

    def inverse(self) -> Self:
        """ Inverse operator for ArrayWFrame objects.

            Only valid for square matrices.
        """
        # perform operation on AWU
        array = self.quantity.inverse()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    def pseudo_inverse(self) -> Self:
        """ Pseudo-inverse operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.pseudo_inverse()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    def exp(self) -> Self:
        """ Exponential operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.exp()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    # Trascendental Exponential2
    def exp2(self) -> Self:
        """ Exponential base 2 operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.exp2()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    # Trascendental Logarithm10
    def log10(self) -> Self:
        """ Log base 10 operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.log10()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    # Trascendental Logarithm2
    def log2(self) -> Self:
        """ Log base 2 operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.log2()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    # Trascendental Logarithm
    def log(self) -> Self:
        """ Natural log operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.log()

        # return as AWF
        return ArrayWFrame(array, self.frame)
    
    def sin(self) -> Self:
        """ Trigonometric sine operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.sin()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    def cos(self) -> Self:
        """  Trigonometric cosine operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.cos()

        # return as AWF
        return ArrayWFrame(array, self.frame)

    def tan(self) -> Self:
        """ Trigonometric tangent operator for ArrayWFrame objects. """
        # perform operation on AWU
        array = self.quantity.tan()

        # return as AWF
        return ArrayWFrame(array, self.frame)
    
    # endregion Operators 

    # region Methods
    def append(self, other : Self, axis : int = None) -> Self:
        """  Append another ArrayWFrame to this one along the given axis.

            Parameters
            ----------
            other : ArrayWFrame
                The ArrayWFrame to append. Must share the same reference frame.
            axis : int, optional
                Axis along which to append. Passed directly to
                :meth:`ArrayWUnits.append`. Defaults to ``None`` (flatten).

            Returns
            -------
            out : ArrayWFrame
                A new ArrayWFrame containing the concatenated values in the same frame.
        """
        # can only append AWF in the same frame
        if isinstance(other, ArrayWFrame):
            if self.frame != other.frame:
                bad_frm_err = (f'Cannot append ArrayWFrame with associated frame {other.frame} ',
                               f'to ArrayWFrame with associated frame {self.frame}.')
                raise ValueError(bad_frm_err)
        
        else:
            not_awf_err = (f'Cannot append object of type {type(other)} to ArrayWFrame.')
            raise TypeError(not_awf_err)
        
        # append given AWF and return
        return ArrayWFrame(self.quantity.append(other.quantity, axis), self.frame)

    def convert_to(self, frame_target: Frame, epoch: EpochArray = None) -> None:
        """ Convert an entire AWF from its current frame to a target frame at given epochs.

            Parameters
            ----------
            frame_target : Frame
                The target frame to convert to.

            epoch : EpochArray, optional
                Given an array of ET at which the transformations hold. Required if the AWF has time-dependent data.

            Raises
            ------
            ValueError
                If the transformation requirements are not met (shape mismatch, incompatible frames, etc.).
        """
        shape = self.shape  # Get the shape of the quantity

        # Single vector case: (3,)
        if shape == (3,):
            # Ensure epoch is either None or a single value
            if epoch is not None and epoch.size != 1:
                raise ValueError(
                    "Epoch array should have exactly one element for (3,) vector conversion."
                )

            # Compute the rotation matrix
            R = Frame.get_DCM(
                self.frame,
                frame_target,
                epoch[0] if epoch is not None else self.frame.epoch,
            )

            # Apply transformation
            transformed_quantity = R @ self.quantity.values

        # Multiple vectors case: (3, N)
        elif shape[0] == 3:
            n_elements = shape[1]

            if epoch is None:
                raise ValueError("Epoch array is required for (3, N) transformations.")

            if epoch.size != n_elements:
                raise ValueError(
                    f"Mismatch: AWF has {n_elements} elements, but {epoch.size} epochs were given."
                )

            # Convert each time step
            transformed_quantity = np.zeros_like(self.quantity.values)
            for ii in range(n_elements):
                R = Frame.get_DCM(self.frame, frame_target, epoch[ii])
                transformed_quantity[:, ii] = R @ self.quantity.values[:, ii]

        else:
            raise ValueError(f"Unsupported AWF shape: {shape}. Must be (3,) or (3, N).")

        # Update frame and quantity
        self._frame = frame_target
        self._quantity = ArrayWUnits(transformed_quantity, self.quantity.units)
