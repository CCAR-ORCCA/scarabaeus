# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import Dimensions, Units

import scarabaeus.utils.NumpyWrapper as np
import numpy
import numpy.typing as npt

import sys
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
import operator
import warnings

# define units
unitless, rad = Units.get_units(['unitless', 'rad'])

# class definition
class ArrayWUnits:
    """ Defines a representation of numerical values and their associated units.

        The values and units are attached and computed together during mathematical 
        operations. Supports scalar and matrix definitions.

        Note that "scalar" definitions are still treated as 1x1 matrices internally.

        Parameters
        ----------
        values : int or float or list or numpy.ndarray
            The numerical value or values of the associated object. Accepts scalar 
            or matrix/vector inputs:

            * ``int`` or ``float`` - a scalar value
            * ``np.ndarray`` - a matrix or vector of values

        units : Unit or list or numpy.ndarray
            The units associated with the given numerical value or values. If a single 
            Unit object is passed, then all numerical values are assumed to be the 
            same unit.
            
            To assign separate units to individual elements in a given numerical 
            matrix/vector, pass either a ``list`` or ``np.ndarray`` with the same 
            size as the numerical values. Each element of this ``list`` or ``np.ndarray`` 
            should be the desired Unit of the corresponding element in the given 
            numerical values.

        See Also
        --------
        scarabaeus.Units : Scarabaeus-defined representation of physical units.
        scarabaeus.Dimensions : Scarabaeus-defined representation of physical dimensions.
        scarabaeus.ArrayWFrame : Attaches a reference frame to ArrayWUnits.

        Notes
        -----
        While both ``values`` and ``units`` can optionally be given as multiple different object 
        types, internally they are always converted to ``np.ndarray`` objects to ensure numpy functionality.

        Examples
        --------
        Create a scalar value with associated units:

        >>> import scarabaeus as scb
        >>> km, sec = scb.Units.get_units(['km', 'sec'])  # define units
        >>> ten_km = scb.ArrayWUnits(10, km)
        >>> print(ten_km)
        10 km
        
        Create a vector of values with different units:

        >>> import numpy as np
        >>> nums = np.array([1, 2, 3, 4]).reshape(-1, 1)
        >>> units = np.array([km, km, km/sec, km/sec]).reshape(-1, 1)
        >>> state_vec2D = scb.ArrayWUnits(nums, units)
        >>> print(state_vec2D)
        [[1]
         [2]
         [3]
         [4]] [non-homogeneous]
    """
    # region Constructor
    def __init__(self, values : int | float | list | npt.NDArray, 
                 units : Units | list[Units] | numpy.ndarray[Units] | None):
        ## construction validation and conditioning
        # must be a valid object type
        if not (isinstance(values, int) or isinstance(values, float) or
                isinstance(values, list) or isinstance(values, np.ndarray) or
                isinstance(values, np.integer) or isinstance(values,np.autograd_types)):
            bad_type_err = ('Argument [values] must be an object of one of the following '
                            'types: int, float, list, np.ndarry. Receveived: '
                            f'{type(values)}.')
            raise TypeError(bad_type_err)

        # condition values to numpy array for simplicity
        if isinstance(values, list):
            values = np.asarray(values, dtype = float)

        elif isinstance(values, int) or isinstance(values, float):
            values = np.asarray([values], dtype = float)

        # extract additional data
        shape = values.shape
        size  = values.size

        # assign class properties
        self.values = values    # input validation in setter

        # NOTE: self._homogeneous_units flag is set in the units setter as well
        self.units  = units     # input validation in setter
        
        self._shape = shape
        self._size  = size

    # region Properties
    @property
    def values(self) -> np.ndarray:
        """ The numerical value or values associated with the object. """
        if self._values.size == 1:
            # return a single values if it's a 0-dimensional array (scalar)
            return self._values.flatten()[0]
        
        else:
            # otherwise return the whole array
            return self._values
    
    @values.setter
    def values(self, input_val : int | float | np.ndarray):
        ## input validation
        # condition to numpy array
        if not isinstance(input_val, np.ndarray):
            input_val = np.array(input_val)

        # all elements must be numerical
        if not np.issubdtype(input_val.dtype, np.number):
            not_num_err = ('All elements of argument [values] must be of type float or '
                           f'int. Received: {input_val.dtype}.')
            raise TypeError(not_num_err)
        
        # can't be an empty array
        if input_val.size == 0:
            raise Exception('Argument [values] must be a non-empty numpy.ndarray instance.')
        
        # valid input
        self._values = input_val

    @property
    def units(self) -> Units | numpy.ndarray[Units]:
        """ The unit or units associated with the object.

            The type is dependent on the homogeneity of the associated units:

            * Units object if the object is a scalar value or all values in the 
              numerical matrix/vector are assigned the same units.
            * Numpy array of Units objects if the values in the numerical 
              matrix/vector are not assigned the same units.
        """
        if self._units.size == 1:
            # return a Units object if it's a 0-dimensional array (scalar)
            return self._units.flat[0]
        
        else:
            # otherwise return the whole array
            return self._units
    
    @units.setter
    def units(self, input_val):
        ## input validation and conditioning 
        # must be a valid type -> perform different logic depending on type
        if isinstance(input_val, type(None)):
            # special case None -> assign unitless...
            self._units = numpy.array(unitless)    # numpy array for parity with other options

            # ...and set homogeneity flag
            self._homogeneous_units = True

        elif isinstance(input_val, Units):
            # single Unit object -> assign property...
            self._units = numpy.array(input_val)   # numpy array for parity with other options

            # ...and set homogeneity flag
            self._homogeneous_units = True

        elif (isinstance(input_val, list) or isinstance(input_val, numpy.ndarray)):
            ## list or numpy array of Unit objects -> further validation required
            # force numpy array to simplify things
            if isinstance(input_val, list):
                input_val = numpy.array(input_val)
            
            ## more input validation
            # all elements must be Units objects
            rebuild_none_as_unitless = False
            for element in input_val.flatten():
                if isinstance(element, type(None)):
                    # set flag to rebuid with all Nones as unitless Units
                    rebuild_none_as_unitless = True

                elif not isinstance(element, Units):
                    not_unit_err = ('All elements of argument [units] must be of type '
                                    f'Unit. Received: {[type(element) for element in input_val]}')
                    raise TypeError(not_unit_err)

            # reassign any None elements as unitless Units elements
            rebuilt = []
            if rebuild_none_as_unitless:
                # create flat array and replace all Nones with unitless
                for element in input_val.flatten():
                    if isinstance(element, type(None)):
                        rebuilt.append(unitless)
                    else:
                        rebuilt.append(element)
                        
                # reshape back to original
                input_val = numpy.array(rebuilt).reshape(input_val.shape)
            
            # check for homogeneity
            if numpy.all(input_val == input_val.flat[0]):    # .flat[0] always gives the zeroth element regardless of matrix dimensions
                # all units are equal -> set homogeneity flag accordingly...
                self._homogeneous_units = True

                # ...and assign all of the units as a single Unit to save space
                self._units = numpy.array(input_val.flat[0])  # numpy array for parity with other options

            else:
                ## units are not equal ->  set homogeneity flag accordingly...
                self._homogeneous_units = False

                # ...and assign units as an array
                self._units = input_val

        else:
            # invalid type -> raise error
            bad_type_err = ('Argument [units] must be an object of one of the '
                            'following types: Unit, list[Unit], numpy.ndarray[Unit], or '
                            f'None. Received: {type(input_val)}.')
            raise TypeError(bad_type_err)

    @property
    def shape(self) -> tuple[int, int]:
        """ The number of columns and rows within the ArrayWUnits object. """
        return self._shape

    @property
    def size(self) -> int:
        """ The number of elements contained within the ArrayWUnits object. """
        return self._size
    
    @property
    def homogeneous_units(self) -> bool:
        """ Boolean flag indicating whether or not all units 
            are homogeneous.
            
            ``True`` if all units are homogeneous. Otherwise ``False``.
        """
        return self._homogeneous_units
    
    # endregion Properties

    # region Operators
    def __repr__(self):
        if self.homogeneous_units:
            ## homogeneous units -> can show beside the values
            if self._values.size == 1:
                # single values -> only need one unit 
                return f'{self.values} {self.units}'
            else:
                ## multiple values -> note that 1 through n elements equal the unit
                # extract shape data
                m, n = np.atleast_2d(self.values).shape

                # format according to shape
                match (m, n):
                    case (1, _):
                        # homogeneous row vector
                        return f'{self.values} [{self.units} ... {self.units}]'
                    
                    case (_, 1):
                        # homogeneous column vector
                        # initialize string to print
                        val_cnt = self.values.size
                        to_print = []

                        # print units vector beside values
                        for i, val in enumerate(self.values):
                            # different formatting depending on the row
                            if i == 0:
                                # first row
                                if val_cnt == 3:
                                    # 3 rows -> top of units is also top of values
                                    to_print.append(f'[{val}  [[{self.units}]\n')
                                else:
                                    # more than 3 rows -> don't start units yet
                                    to_print.append(f'[{val}\n')

                            elif i == (val_cnt - 1):
                                # last row
                                to_print.append(f' {val}]  [{self.units}]]')

                            else:
                                # intermediate row(s)
                                if i == (val_cnt - 3):
                                    # start of units
                                    to_print.append(f' {val}  [[{self.units}]\n')
                                elif i == (val_cnt - 2):
                                    # middle of units
                                    to_print.append(f' {val}   [{"⋮".center(len(self._units.flat[0].name))}]\n')
                                else:
                                    to_print.append(f' {val}\n')

                        # return the finished string
                        return ''.join(to_print)
                    
                    case (_, _):
                        # homogeneous matrix
                        # initialize strings to print
                        _, val_cnt = self.values.shape
                        to_print = []

                        horz = ' ... '
                        vert = '⋮'.center(len(self._units.flat[0].name))

                        # print units matrix beside values
                        for i, val in enumerate(self.values):
                            # different formatting depending on the row
                            if i == 0:
                                # first row
                                if val_cnt == 3:
                                    # 3 rows -> top of units is also top of values
                                    to_print.append(f'[{val}  [[{self.units}{horz}{self.units}]\n')
                                else:
                                    # more than 3 rows -> don't start units yet
                                    to_print.append(f'[{val}\n')

                            elif i == (val_cnt - 1):
                                # last row
                                to_print.append(f' {val}]  [{self.units}{horz}{self.units}]]')

                            else:
                                # intermediate row(s)
                                if i == (val_cnt - 3):
                                    # start of units
                                    to_print.append(f' {val}  [[{self.units}{horz}{self.units}]\n')
                                elif i == (val_cnt - 2):
                                    # middle of units
                                    to_print.append(f' {val}   [{vert}  ⋱  {vert}]\n')
                                else:
                                    to_print.append(f' {val}\n')

                        # return the finished string
                        return ''.join(to_print)
        else:
            # non-homogeneous units -> show each unit beside values
            # NOTE: for now just noting that it's non-homogeneous. need to implement unit key pull for this to work properly
            return f'{self.values} [non-homogeneous]'

    def __str__(self):
        return self.__repr__()

    # Get a sub-array from the ArrayWUnits instance
    def __getitem__(self, index):
        # special cases
        if self.shape == ():
            if index == 0:
                return self
            else:
                raise IndexError("ArrayWUnits index out of range.")
            
        # return the indexed element
        if isinstance(self._units, numpy.ndarray):
            if self.homogeneous_units:
                # homogeneous units -> all elements have the same units
                return ArrayWUnits(self._values[index], self._units)
            
            else:
                # non-homogeneous units -> index units as well
                return ArrayWUnits(self._values[index], self._units[index])

    def __setitem__(self, key, awu_in):
        ## input validation
        # Check that the input argument is an ArrayWUnits instance
        if not isinstance(awu_in, self.__class__):
            raise TypeError('Set-item operation requires an input argument of type ArrayWUnits.')

        # Check that the input array is numerical
        if not ArrayWUnits._is_array_numerical(awu_in._values):
            raise TypeError("The input ArrayWUnits must contain only numerical values.")

        # If the input array has the same units as the queried array,
        # then only the values in the queried array need to be updated
        same_units = (self._units == awu_in._units)

        if isinstance(same_units, bool):
            # make iterable for single values
            same_units = numpy.array([same_units])

        if same_units:
            self._values = np.set(self._values,key,awu_in.values)

        # For all other cases, units are updated and type-casting is performed again
        else:
            # Update numerical array
            self.values = np.set(self.values,key,awu_in.values)

            # Update units array
            if isinstance(self.units, Units):
                # singular unit -> assign without index
                self.units = awu_in.units

            else:
                # array of units -> assign to given index
                self.units[key] = awu_in.units

    # Compute output powers array, depending on the operator
    def units_arithmetic(self, other, operation):
        """ Internal method holding all """
        if isinstance(self._units, numpy.ndarray):
            # ensure compatible objects
            if not isinstance(other, ArrayWUnits):
                compatible_err = ('Mathematical operations only valid for ArrayWunits objects with '
                                  f'other objects of type ArrayWUnits. Received: {type(other)}.')
                raise TypeError(compatible_err)
            
            # perform necessary math depending on the operation type
            match operation:
                case operator.mul:
                    # multiply units
                    new_units = self.units * other.units

                    # multiply values
                    new_vals  = self.values * other.values

                    # return new AWU object
                    return ArrayWUnits(new_vals, new_units)
                
                case operator.truediv:
                    # divide units
                    new_units = self.units / other.units

                    # multiply values
                    new_vals  = self.values / other.values

                    # return new AWU object
                    return ArrayWUnits(new_vals, new_units)

    def __add__(self, other):
        ## input validation
        # can only add another AWU
        if not isinstance(other, self.__class__):
            not_awu_err = ('Addition operation is only defined between two ArrayWUnits '
                           f'objects. Received: {type(other)}.')
            raise TypeError(not_awu_err)

        # can only add compatible matrix shapes
        if not self._are_shapes_compatible(other):
            raise ValueError('Shapes are incompatible for addition.')

        # can only add AWU's with the same dimensions
        if self._units.size == 1:
            if not self.units == other.units:
                bad_dim_err = ('Addition operator is only defined for ArrayWUnits objects '
                               'containing like units. Received: '
                               f'{self.units} + {other.units}.')
                raise ValueError(bad_dim_err)
        else:
            for i, unit in enumerate(self._units):
                if unit.dimensions != other._units[i].dimensions:
                    bad_dim_err = ('Addition operator is only defined for ArrayWUnits objects '
                                    'containing like units. Received: '
                                    f'{unit} + {other._units[i]}.')
                    raise ValueError(bad_dim_err)
        
        # valid operation -> return addition
        # can't add units -> keep the same
        new_units = self.units

        # add values
        new_vals  = self.values + other.values

        # return self with updated values
        return ArrayWUnits(new_vals, new_units)

    def __neg__(self):
        return ArrayWUnits(-1 * self.values, self.units)

    # Element-wise power operator
    def __pow__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(
                'Power operation for ArrayWUnits objects only defined '
                'for exponents of type int or float. Received object of type: '
                f'{type(other)}.'
            )

        new_values = self._values.astype(float) ** other
        new_units  = self.units ** other

        return ArrayWUnits(new_values, new_units)

    def sqrt(self) -> Self:
        """ Returns the element-wise square root of the ArrayWUnits object
            (equivalent to ``awu ** 0.5``).
        """
        return self ** 0.5

    # Element-wise multiplication operator
    def __mul__(self, other):
        # Check instance
        if isinstance(other, self.__class__):
            return ArrayWUnits(self._values * other._values, self._units * other._units)
        
        elif isinstance(other, int) or isinstance(other, float):
            # Case ArrayWUnits * int or ArrayWUnits * float
            return ArrayWUnits(other * self.values, self.units)
        
        else:
            cant_mul_err = ('Multiplication operation for ArrayWUnits is only defined '
                            'with objects of type int, float, or ArrayWUnits.')
            raise TypeError(cant_mul_err)

    # Matrix multiplication (+ special dot product) operator
    def __matmul__(self, other):
        ## input validation
        # For now, matmul only supports matrix multiplication between arrays with dimension <= 2
        if not isinstance(self, self.__class__) or not isinstance(other, self.__class__):
            raise TypeError('The two input arguments of matrix multiplication must be of type ArrayWUnits.')

        # Check on the input dimension
        n_self, m_self   = ArrayWUnits._get_shape(self)
        n_other, m_other = ArrayWUnits._get_shape(other)

        # Check that the dimensions match for the matrix-multiplication operation
        if n_self == n_other == 1:  # dot product of 2 (1,m1)X(1,m2) row vectors
            if m_self != m_other:
                raise Exception('The shapes of the two operands do not match for vector multiplication.')
            shapeOut = (1, 1)

        else:
            if m_self != n_other:   # generic matmul product of 2 (n1xm1)X(n2xm2) matrices
                raise Exception('The shapes of the two operands do not match for matrix multiplication.')
            shapeOut = (n_self, m_other)

        if n_self == n_other == 1:
            arrOut = (self * other).summation()

        else:
            arrOut = ArrayWUnits.empty(shapeOut)
            # Perform matrix multiplication for each row-column pair
            for i in range(n_self):
                for j in range(m_other):
                    # Query this iteration's row-column pair
                    rowIter = self[i, :]

                    if n_other == 1 and m_other > 1:
                        colIter = other[j]

                    else:
                        colIter = other[:, j]

                    # Perform row-column dot product
                    arrOut[i, j] = (rowIter * colIter).summation()

        # return operation
        return arrOut

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __eq__(self, other):
        """ Overloading the element-wise equality operator of an ArrayWUnits object.
            
            Using the property that, if x = y, then x-y = 0.

            Note that instances representing the same quantity with different units are 
            considered equal.
        """
        ## input validation
        # equating to None -> not equal
        if other == None:
            return False
        
        # ensure that both objects being compared are of type ArrayWUnits
        if not isinstance(other, self.__class__):
            raise TypeError("Equality operation requires both instances to be of type ArrayWUnits.")

        # apply equality check depending on matrix dimension
        if self._values.size == 1:
            # 1x1 -> check units and values directly
            return ( (self.values == other.values) and (self.units == other.units) )
        
        else:
            ## NxM -> check by element
            # prepare units for comparison by element
            if self.homogeneous_units:
                # create a list of the same units with length equal to the values
                self_units_check = self.units * np.ones(len(self.values))
            else:
                # flatten list of units
                self_units_check = self.units.flatten()

            if other.homogeneous_units:
                # create a list of the same units with length equal to the values
                other_units_check = other.units * np.ones(len(other.values))
            else:
                # flatten list of units
                other_units_check = other.units.flatten()
            
            # compare each element
            equal = []
            for self_val, other_val, self_unit, other_unit in zip(self.values, other.values, self_units_check, other_units_check):
                if (np.array_equal(self_val, other_val)) and (self_unit.dimensions == other_unit.dimensions):
                    equal.append(True)
                else:
                    equal.append(False)
            
            return equal

    def __ne__(self, other):
        # re-use equality logic
        equality_result = self.__eq__(other)

        # invert
        return np.invert(equality_result)

    def __gt__(self, other : Self):
        ## input validation
        # Check that the operands have the same unit dimensions
        if numpy.any(self.units.dimensions.powers != other.units.dimensions.powers):
            raise Exception('Inequality operation requires the ArrayWUnits operands to have the same unit dimensions.')

        # return operation
        return (self - other).values > 0.0

    def __ge__(self, other : Self):
        # Check that the operands have the same unit dimensions
        if numpy.any(self.units.dimensions.powers != other.units.dimensions.powers):
            raise Exception(
                "Inequality operation requires the ArrayWUnits operands to have the same unit dimensions."
            )

        return (self - other).values >= 0.0

    def __lt__(self, other : Self):
        # Check that the operands have the same unit dimensions
        if numpy.any(self.units.dimensions.powers != other.units.dimensions.powers):
            raise Exception('Inequality operation requires the ArrayWUnits operands to have the same unit dimensions.')

        return (self - other).values < 0.0

    def __le__(self, other : Self):
        # Check that the operands have the same unit dimensions
        if numpy.any(self.units.dimensions.powers != other.units.dimensions.powers):
            raise Exception('Inequality operation requires the ArrayWUnits operands to have the same unit dimensions.')

        return (self - other).values <= 0.0

    def cross(self, other : Self) -> Self:
        """ Cross product operation between two ArrayWUnits objects. """
        ## input validation
        # must cross with an ArrayWUnits object
        if not isinstance(other, self.__class__):
            raise TypeError('Cross-product operation is only defined between two ArrayWUnits objects')

        # both must have homegeneous units
        if not (self.homogeneous_units and other.homogeneous_units):
            raise Exception('Cross-product operation is only defined for ArrayWUnits with homogeneous units')

        # both must be vectors
        if not (len(self.shape) == 1 and len(other.shape) == 1):
            raise Exception('Cross-product operation is only defined for vectorial ArrayWUnits objects')
        
        # perform cross product
        # multiply units
        units_out  = self.units * other.units

        # utilize numpy for cross product itself
        values_out = np.cross(self.values, other.values)

        # construct and return
        cross_product = ArrayWUnits(values_out, units_out)
        
        return cross_product

    def summation(self) -> Self:
        """ Sum all elements in the ArrayWUnits object, if units allow. """
        if self.homogeneous_units:
            # homogeneous units -> can sum values
            return ArrayWUnits(np.sum(self.values), self.units)
        else:
            # non-homogeneous units -> cannot sum
            cant_sum_err = ('Cannot perform summation operation on ArrayWUnits '
                            'object containing non-homogeneous units.')
            raise ValueError(cant_sum_err)

    def norm(self) -> Self:
        """ Matrix or vector norm operator for ArrayWUnits objects. """
        if self.homogeneous_units:
            # units are all the same -> return normalized value with the same units
            return ArrayWUnits(np.linalg.norm(self.values), self.units)
        else:
            # non-homogeneous units -> return normalized value with no units
            return ArrayWUnits(np.linalg.norm(self.values), unitless)

    def unitary(self) -> Self:
        """ Converts a given ArrayWUnits vector into a unit vector with the same units.

            Only valid for vectorial ArrayWUnits.
        """
        # get matrix dimensions
        n, m = self.atleast_2d(self).shape

        if self.homogeneous_units:
            if (n == 1) or (m == 1):
                # column or row vector -> unitize and return
                return (self.norm() ** - 1) * self
            else:
                # not a vector -> can't unitize
                non_1d_err = ('Cannot unitize a non-1-D ArrayWUnits vector. '
                              f'Received: {n} x {m} vector.')
                raise ValueError(non_1d_err)
        else:
            # cannot unitize non-homogenous units
            non_hom_err = ('Cannot unitize an ArrayWUnits objects with '
                           'non-homogeneous units.')
            raise ValueError(non_hom_err)

    def transpose(self) -> Self:
        """ Transpose operator for ArrayWUnits objects.
        """
        ## input validation
        # must have valid matrix dimensions
        if len(self.shape) <= 2:
            n, m = ArrayWUnits._get_shape(self)
        else:
            raise ValueError('Transpose operation defined only for scalar, 1D, and 2D ArrayWunits instances.')

        # find transpose and return
        # first the units
        if self.homogeneous_units:
            # all units the same -> keep them
            new_units = self.units

        else:
            ## units are not the same -> reassign their positions
            # initialize dimensions and scales
            new_dims = numpy.zeros((m, n, self.units.dimensions.powers.shape[-1]), dtype = numpy.integer)
            new_scls = numpy.zeros((m, n, self.units.dimensions.powers.shape[-1]), dtype = numpy.floating)
            if n == 1:
                for col in range(m):
                    new_dims[col][0] = self.units.dimensions.powers[col]
                    new_scls[col][0] = self.units.scales[col]
            else:
                for row in range(n):
                    for col in range(m):
                        new_dims[col][row] = self.units.dimensions.powers[row, col, :]
                        new_scls[col][row] = self.units.scales[row, col, :]

            new_units = Units(Dimensions(new_dims), new_scls)

        # then the values
        if n == m == 1:
            return ArrayWUnits(self.values, new_units)
        
        elif n == 1 and m > 1:
            return ArrayWUnits(self.values.reshape((-1, 1)), new_units)
        
        elif len(self.shape) > 1:
            if m == 1:
                return ArrayWUnits(self.values.transpose()[0], new_units)  # Work with row vectors
            else:
                return ArrayWUnits(self.values.transpose(), new_units)  # Work with matrices and col vectors
            
    T = transpose
    """ Same as self.transpose()

        See Also
        --------
        scarabaeus.ArrayWUnits.transpose
    """

    def inverse(self) -> Self:
        """ Inverse operator for ArrayWUnits objects.

            Only valid for square matrices.
        """
        if isinstance(self.units, Units):
            ## input validation
            # can't invert non-homogeneous awu's
            if not self.homogeneous_units:
                non_hom_err = ('Inverse operation is only defined for ArrayWUnits '
                               'objects with homogeneous units.')
                raise ValueError(non_hom_err)
            
            # can't invert non-square matrices
            if self.shape[0] != self.shape[1]:
                non_sqr_err = ('Inverse operation is only defined for square '
                               'matrices.')
                raise ValueError(non_sqr_err)
            
            # invert and return
            # units first
            inv_dims  = Dimensions(-self.units.dimensions.powers)       # inverted powers
            inv_units = Units(inv_dims, self.units.scales)              # same scaling

            # now values
            inv_vals  = np.linalg.inv(self.values)

            # return
            return ArrayWUnits(inv_vals, inv_units)

    def pseudo_inverse(self) -> Self:
        """ Pseudo-inverse operator for ArrayWUnits objects. """
        # Checks
        if not self.homogeneous_units:
            raise Exception("Pseudo-inverse operation is only defined for ArrayWUnits instances with homogeneous units.")

        # Invert units
        inv_dims  = Dimensions(-self.units.dimensions.powers)       # inverted powers
        inv_units = Units(inv_dims, self.units.scales)              # same scaling

        return ArrayWUnits(np.linalg.pinv(self._values), inv_units)
    
    # Trascendental Exponential
    # OSS: Methods implemented w/o changing the units (i.e., defined over dimensional variables)
    def exp(self) -> Self:
        """ Exponential operator for ArrayWUnits objects. """
        return ArrayWUnits(np.exp(self._values), self._units)

    # Trascendental Exponential2
    def exp2(self) -> Self:
        """ Exponential base 2 operator for ArrayWUnits objects. """
        return ArrayWUnits(np.exp2(self._values), self._units)

    # Trascendental Logarithm10
    def log10(self) -> Self:
        """ Log base 10 operator for ArrayWUnits objects. """
        return ArrayWUnits(np.log10(self._values), self._units)

    # Trascendental Logarithm2
    def log2(self) -> Self:
        """ Log base 2 operator for ArrayWUnits objects. """
        return ArrayWUnits(np.log2(self._values), self._units)

    # Trascendental Logarithm
    def log(self) -> Self:
        """ Natural log operator for ArrayWUnits objects. """
        return ArrayWUnits(np.log(self._values), self._units)
    
    def sin(self) -> Self:
        """ Trigonometric sine operator for ArrayWUnits objects. """
        ## input validation
        # must be homogeneous units
        if not self.homogeneous_units:
            raise Exception('Sine function is only defined for ArrayWUnits instances with homogeneous units')
        
        # all units must be radians
        if self.units != rad:
            raise ValueError("Sine function only accepts an input in radians")
        
        # perform operation and return
        sine_values = np.sin(self._values)
        return ArrayWUnits(sine_values, unitless)

    def cos(self) -> Self:
        """ Trigonometric cosine operator for ArrayWUnits objects. """
        if isinstance(self._units.item(0), Units):
            ## input validation
            if not self.homogeneous_units:
                non_hom_err = ('Cosine function is only defined for ArrayWUnits '
                               'instances with homoegeneous units.')
                raise ValueError(non_hom_err)
            
            if self.units != rad:
                not_rad_err = ('Cosine function is only defined for ArrayWUnits '
                               'objects with units of radians. Current units: '
                               f'{self.units}.')
                raise ValueError(not_rad_err)
            
            # perform operation and return
            cos_values = np.cos(self._values)
            return ArrayWUnits(cos_values, unitless)

    def tan(self) -> Self:
        """ Trigonometric tangent operator for ArrayWUnits objects. """
        ## input validation
        if not self.homogeneous_units:
            non_hom_err = ('Tangent function is only defined for ArrayWUnits '
                           'objects with homogeneous units.')
            raise TypeError(non_hom_err)
        
        if self.units != rad:
            non_rad_err = ('Cannot perform tangent operation on ArrayWUnits '
                           'object with units in radians. Received: '
                           f'{self.units}.')
            raise ValueError(non_rad_err)
        
        # return operation
        tan_values = np.tan(self._values)
        return ArrayWUnits(tan_values, unitless)
    
    # endregion Operators

    # region Methods
    def _get_shape(self):
        """ Return the (n, m) shape of this ArrayWUnits as a 2-tuple.

            Scalars return ``(1, 1)``, 1-D arrays return ``(1, size)``, and
            2-D arrays return ``(rows, cols)``.  Used internally to normalise
            shapes for transpose and dot-product operations.

            Returns
            -------
            shape : tuple[int, int]
                ``(n, m)`` where *n* is the number of rows and *m* the number of
                columns in the 2-D sense.
        """
        # Checks
        if len(self.shape) == 0:
            # Scalars
            n = 1
            m = 1
        elif len(self.shape) == 1:
            # Only applicable to 1D row vectors defined from np.array()
            n = 1
            m = self.shape[0]
        elif len(self.shape) == 2:
            # Column vectors and matrices
            n = self.shape[0]
            m = self.shape[1]

        return (n, m)


    @staticmethod
    def _is_array_numerical(array_in : np.ndarray) -> bool:
        """ Checks if a given array contains numerical values or not.

            Parameters
            ----------
            array_in : np.ndarray
                The array to examine.

            Returns
            -------
            is_numerical : bool
                The numerical status of the examined array. Returns ``True`` if 
                the array is numerical and ``False`` if it is not.
        """
        return np.issubdtype(array_in.dtype, np.number)
    
    @staticmethod
    def empty(shape : int | tuple[int, int]) -> Self:
        """ Initializes a new ArrayWUnits object of the given shape without 
            initializing value entries. Units are set as unitless.

            ArrayWUnits equivalent of Numpy ``empty()`` function.

            Parameters
            ----------
            shape : int | tuple[int, int]
                The shape of the empty ArrayWUnits. One-dimensional arrays 
                should be given as integers while n-dimensional arrays should be 
                given as a tuple of integers.

            Returns
            -------
            out : ArrayWUnits
                An ArrayWUnits object of given shape with empty values and unitless 
                units.

            References
            ----------
            .. [1] Numpy,
              https://numpy.org/doc/2.1/reference/generated/numpy.empty.html
        """
        # return empty values with unitless units
        return ArrayWUnits(np.empty(shape), unitless)

    # def atleast_1d(self, awu_in) -> Self:
    #     """ Recast an ArrayWUnits so it has at least one dimension.

    #         ArrayWUnits equivalent of ``numpy.atleast_1d()``.

    #         Parameters
    #         ----------
    #         awu_in : ArrayWUnits
    #             The ArrayWUnits to recast.

    #         Returns
    #         -------
    #         out : ArrayWUnits
    #             The recast ArrayWUnits with at least one dimension.

    #         Notes
    #         -----
    #         Not yet implemented — raises ``NotImplementedError`` when called.
    #     """
    #     pass

    def atleast_2d(self, awu_in : Self = None) -> Self:
        """ Rescast the ArrayWUnits object as arrays with atleast two dimensions.

            ArrayWUnits equivalent of Numpy ``atleast_2d()`` function.

            Parameters
            ----------
            awu_in : ArrayWUnits, optional
                The AWU to recast.

            Returns
            -------
            out : ArrayWUnits
                The recast AWU.

            References
            ----------
            .. [1] Numpy,
              https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html
        """
        ## input validation and conditioning
        # given an object to recast -> must be a an awu
        if (not isinstance(awu_in, ArrayWUnits)) and (awu_in != None):
            not_awu_err = ('Cannot cast a non-ArrayWUnits object as '
                           'an ArrayWUnits object of two dimensions.')
            raise TypeError(not_awu_err)
        
        # no object passed -> recast self
        else:
            awu_in = self

        # recast and return
        if awu_in.homogeneous_units:
            ## homogeneous awus return non-arrays -> convert to array and recast
            # values
            new_vals  = np.atleast_2d(np.array(awu_in.values))

            # units
            new_units = numpy.atleast_2d(numpy.array(awu_in.units))

        else:
            ## non-hom awus return arrays of units -> recast
            # values
            new_vals  = np.atleast_2d(awu_in.values)

            # units
            new_units = numpy.atleast_2d(awu_in.units)

        # return new awu
        return ArrayWUnits(new_vals, new_units)

    # def atleast_3d(self, awu_in) -> Self:
    #     """ Recast an ArrayWUnits so it has at least three dimensions.

    #         ArrayWUnits equivalent of ``numpy.atleast_3d()``.

    #         Parameters
    #         ----------
    #         awu_in : ArrayWUnits
    #             The ArrayWUnits to recast.

    #         Returns
    #         -------
    #         out : ArrayWUnits
    #             The recast ArrayWUnits with at least three dimensions.

    #         Notes
    #         -----
    #         Not yet implemented — raises ``NotImplementedError`` when called.
    #     """
    #     pass

    # @staticmethod
    # def eye(n : int, units : Units | list[Units] | numpy.ndarray[Units] = None,
    #         diag_units : Units | list[Units] | numpy.ndarray[Units] = None,
    #         m : int = None, k : int = None) -> Self:
    #     """ Return an identity ArrayWUnits of shape (n, m).

    #         ArrayWUnits equivalent of ``numpy.eye()``.

    #         Parameters
    #         ----------
    #         n : int
    #             Number of rows in the output identity matrix.
    #         units : Units or list[Units] or numpy.ndarray[Units], optional
    #             Units assigned to all elements.  A single ``Units`` object creates a
    #             homogeneous array; a list or array assigns per-element units.
    #             Defaults to ``None`` (unitless).
    #         diag_units : Units or list[Units] or numpy.ndarray[Units], optional
    #             Units for the diagonal elements only; off-diagonal elements are
    #             set to unitless.  Mutually exclusive with *units*.
    #         m : int, optional
    #             Number of columns. Defaults to *n* (square matrix).
    #         k : int, optional
    #             Diagonal index (0 = main diagonal, positive = upper, negative =
    #             lower). Defaults to ``0``.

    #         Returns
    #         -------
    #         out : ArrayWUnits
    #             An (n, m) identity matrix wrapped in an ArrayWUnits.

    #         Notes
    #         -----
    #         Not yet implemented — raises ``NotImplementedError`` when called.

    #         References
    #         ----------
    #         https://numpy.org/devdocs/reference/generated/numpy.eye.html
    #     """
    #     pass

    def _are_shapes_compatible(self, other) -> bool:
        """ Check if the shapes of self and other are compatible to perform operations
            between them.
            
            Currently, ArrayWUnits supports three cases of operations:

                1) Operations between N-D arrays with the same shape
                2) Operations between an N-D array and a scalar
                3) Operations between a N-D array and a 1-D array
                   whose size is equal to the size of the last dimension
                   of the N-D array. This case comes in handy, for instance,
                   to perform operations between a 3x1 array and a 3xN matrix.
            
            Returns
            -------
            compatible : bool
                ``True`` if shapes are compatible, ``False`` if not.
        """
        # cast shapes as 2D for simplicity
        self_shape  = self.atleast_2d(self).shape
        other_shape = self.atleast_2d(other).shape

        # Check if the two arrays have the same shape
        if self_shape == other_shape:
            shapesMatch = True
            isSelfSubarray = False
            isOtherSubarray = False

        # Check if self is a 1D array whose size is equal to the size of the last dimension of
        # other
        elif (self.values.ndim == 1) and (self.size == other_shape[-1] or self.size == 1):
            shapesMatch = False
            isSelfSubarray = True
            isOtherSubarray = False

        # Check if other is a 1D array whose size is equal to the size of the last dimension of
        # self
        elif (other.values.ndim == 1) and (other.size == self_shape[-1] or other.size == 1):
            shapesMatch = False
            isSelfSubarray = False
            isOtherSubarray = True

        # Otherwise, the two arrays are not compatible to perform operations
        else:
            raise Exception('The shapes of the ArrayWUnits operands are not compatible to perform operations')

        return True

    # def homogenize(self) -> None:
    #     """ Convert this ArrayWUnits to a homogeneous-units representation.

    #         When all elements share the same unit, the internal storage can be
    #         compacted to a single scalar unit rather than an array of per-element
    #         units, which improves performance for arithmetic operations.

    #         Returns
    #         -------
    #         None
    #             Modifies the object in place.

    #         Notes
    #         -----
    #         Not yet implemented — raises ``NotImplementedError`` when called.
    #     """
    #     pass
       
    def append(self, other : Self, axis : int = None) -> Self:
        """ Return an ArrayWUnits object with given ArrayWUnits object 
            appended to the back.

            ArrayWUnits equivalent of Numpy ``append()`` function.

            Parameters
            ----------
            other : ArrayWUnits
                The ArrayWUnits object to be appended.
            
            axis : int, optional
                The axis along which given ArrayWUnits objects are 
                added. If not given, both self and given ArrayWUnits 
                object are flattened before appending. Defaults to 
                ``None``.

            Returns
            -------
            out : ArrayWUnits
                ArrayWUnits object with given ArrayWUnits appended to 
                the end. If ``axis`` is ``None``, ``out`` is is a 
                flattened ArrayWUnits object.

            References
            ----------
            .. [1] Numpy,
              https://numpy.org/doc/2.1/reference/generated/numpy.append.html

            Examples
            --------
            Append two ArrayWUnits vectors:

            >>> import scarabaeus as scb
            >>> kg, km = scb.Units.get_units(['kg', 'km'])  # define units
            >>> one = scb.ArrayWUnits([[1], [2], [3]], kg) # create AWU's
            >>> two = scb.ArrayWUnits(4, kg)
            >>> one.append(two) # no axis specified gives a flattened row vector
            [1. 2. 3. 4.] [kg ... kg]

            Append compatible vector to matrix

            >>> import numpy as np
            >>> mat_units = np.array([[kg, km, km], [km, kg, km], [km, km, kg]])
            >>> mat = scb.ArrayWUnits(np.eye(3), mat_units)
            >>> print(mat)
            [[1. 0. 0.]
             [0. 1. 0.]
             [0. 0. 1.]] [non-homogeneous]
            >>> row = scb.ArrayWUnits(np.ones((1, 3)), km)
            >>> print(row)
            [[1. 1. 1.]] [km ... km]
            >>> appended = mat.append(row, 0)   # append to first axis
            >>> print(appended)
            [[1. 0. 0.]
             [0. 1. 0.]
             [0. 0. 1.]
             [1. 1. 1.]] [non-homogeneous]
        """
        ## input validation
        if not isinstance(other, self.__class__):
            not_awu_err = (f'Cannot append object of type {type(other)} '
                           'to an ArrayWUnits object.')
            raise TypeError(not_awu_err)
        
        # return new AWU
        ## create array of units if self or other homogeneous because they're stored as singular if so
        # self
        if self.homogeneous_units:
            self_units_arr = self.units * numpy.ones(self.shape)
        else:
            self_units_arr = numpy.array(self.units)

        # other
        if other.homogeneous_units:
            other_units_arr = other.units * numpy.ones(other.shape)
        else:
            other_units_arr =numpy.array(other.units)

        try:
            # make sure appending is possible
            new_vals  = np.append(self._values   , other._values   , axis)
            new_units = numpy.append(self_units_arr, other_units_arr, axis)

        except:
            # append error -> notify with awu error instead of np error for clarity
            cant_app_err = ('Cannot append ArrayWUnits objects with '
                            'incompatible shapes.')
            raise ValueError(cant_app_err)
        
        else:
            # valid append -> return appended awu
            return ArrayWUnits(new_vals, new_units)

    # NOTE: redundant?
    def _get_shape(self):
        """ Retrieve the shape of an input ArrayWUnits. """
        ## input validation
        if len(self.shape) == 0:
            # Scalars
            n = 1
            m = 1
        elif len(self.shape) == 1:
            # Only applicable to 1D row vectors defined from np.array()
            n = 1
            m = self.shape[0]
        elif len(self.shape) == 2:
            # Column vectors and matrices
            n = self.shape[0]
            m = self.shape[1]

        return (n, m)

    def angle_between(self, other):
        # NOTE: CONSIDER REMOVING THIS FROM AWU AS IT IS NOT DEPENDENT ON AWU
        """ Angle between two vectors. """
        if isinstance(self._units.flat[0], Units):
            ## input validation
            if not self.homogeneous_units:
                non_hom_err = ('Angle function is only defined for '
                               'ArrayWUnits instances with homogeneous units.')
                raise ValueError(non_hom_err)
            
            # perform operation
            # calculate cosine of angle between vectors
            cos_ang = (self @ other) / (self.norm() * other.norm())

            # return
            return ArrayWUnits(np.arccos(cos_ang.values), rad)

    def convert_to(self, to_convert_to : Units) -> Self:
        """ Convert all elements of an ArrayWUnits object to the given 
            units.

            Valid for ArrayWUnits objects containing non-homogeneous units 
            as long as those units are all of the same dimension.

            Parameters
            ----------
            to_convert_to : Units
                The desired unit the ArrayWUnits is to be converted to.

            Returns
            -------
            converted : ArrayWUnits
                The ArrayWUnits object with converted units.

            Examples
            --------
            Convert 0.05 N to mN and back to N:

            >>> import scarabaeus as scb
            >>> N, mN = scb.Units.get_units(['N', 'mN'])  # define units
            >>> p05N = scb.ArrayWUnits(0.05, N)           # 0.05 N
            >>> print(p05N)
            0.05 N
            >>> p05N_in_mN = p05N.convert_to(mN)          # convert to mN
            >>> print(p05N_in_mN)
            50.0 mN
            >>> back_to_p05N = p05N_in_mN.convert_to(N)   # and back to N
            >>> print(back_to_p05N)
            0.05 N
        """
        ## input validation 
        ## requested conversion must be valid type
        # ensure iterable so that every unit can be checked for non-homogeneous conversions
        if not isinstance(to_convert_to, (list, np.ndarray)):
            validate_convert_to = [to_convert_to]
        else:
            validate_convert_to = to_convert_to
        
        # validate each requested conversion
        for to_validate in validate_convert_to:
            if (not isinstance(to_validate, Units)):
                bad_type_err = (f'Cannot convert {self.units} to {to_validate}. '
                                'Argument [to_convert_to] must be a Units object or '
                                f'list of Units objects. Received: {type(to_validate)}.')
                raise TypeError(bad_type_err)

            # units must have the same dimensions
            if self.homogeneous_units:
                # only one dimension to check
                if self._units.flat[0].dimensions != to_validate.dimensions:
                    bad_dim_err = ('Cannot convert Units object with dimensions '
                                    f'{self.units.dimensions} to Units object with '
                                    f'dimensions {to_validate.dimensions}.')
                    raise ValueError(bad_dim_err)
            else:
                # need to make sure all units are the same dimensions
                for unit, conv_unit in zip(self.units, validate_convert_to):
                    if unit.dimensions != conv_unit.dimensions:
                        bad_dim_err = ('Cannot convert Units object with dimensions '
                                        f'{unit.dimensions} to Units object with '
                                        f'dimensions {conv_unit.dimensions}.')
                        raise ValueError(bad_dim_err)
        
        # convert to requested unit and return
        if self.homogeneous_units:
            # same units everywhere
            self_scl = self._units.flat[0].scales
            conv_scl = to_convert_to.scales
            self_dims = self._units.flat[0].dimensions.powers

            # perform sum of logarithms to avoid machine error losses
            ratio = self._difference_by_sum_of_logs(self_dims,self_scl, conv_scl)

            new_vals = ratio * self.values

        else:
            # look at each unit
            new_vals = []
            for self_unit, conv_unit, val in zip(self.units, to_convert_to, self.values):
                # find ratio
                ratio = self._difference_by_sum_of_logs(self_unit.dimensions.powers,self_unit.scales, conv_unit.scales)
                new_vals.append(ratio * val)

        # return as new unit object
        return ArrayWUnits(new_vals, to_convert_to)
    
    def _difference_by_sum_of_logs(self,self_dims, self_scl, conv_scl) -> float:
        """ Internal method to reduce code repetition in ArrayWUnits.convert_to().

            Converts the given self and conversion scales to integer represenations 
            of their exponents, finds the difference between current and desired exponents, 
            then sums this difference before returning back as an exponential.
            
            Equivalent to taking the product of the scales, but for large numbers this can 
            lead to inaccuracies due to floating point errors. By peforming the multiplications 
            as additions, these inaccuracies are avoided.
        """
        # get exponents of each unit
        self_exps = np.log10(self_scl.copy(), out = self_scl.copy().astype(float), where = self_scl != 0)
        conv_exps = np.log10(conv_scl.copy(), out = self_scl.copy().astype(float), where = conv_scl != 0)

        # find difference between current exponents and requested exponents
        exp_diff = self_exps - conv_exps

        # preallocate
        non_base_10s = []
        sum = 0.0
        # self_dims = self._units.flat[0].dimensions.powers

        # find ratio or exponential ratio for each dimension
        for i, (exp, sign) in enumerate(zip(exp_diff, self_dims)):
            if sign > 0:
                if float(exp).is_integer():
                    # base 10 -> save for sum of logs
                    sum += exp
                else:
                    # not base 10 -> get direct ratio
                    non_base_10s.append(self_scl[i] / conv_scl[i])
            elif sign < 0:
                if float(exp).is_integer():
                    # base 10 -> save for sum of logs
                    sum -= exp
                else:
                    # not base 10 -> get direct ratio
                    non_base_10s.append(1.0 / (self_scl[i] / conv_scl[i]))
        
        # take the product of all direct ratios
        non_base_10_ratio = np.prod(non_base_10s)

        # restore to exponential
        base_10_ratio = 10.0**sum

        # combine and return full ratios
        return base_10_ratio * non_base_10_ratio
    
    @staticmethod
    def get_value_in_target_units(quantity, target_units:Units) -> np.ndarray:
        """ Method to extract a value of an AWU quantity if its units match the desired target units.
            If that's not the case, a warning is displayed and a conversion is attempted. 
            If that also fails, then an error is generated on the quantity of interest not having the proper units.

            Parameters
            ----------
            quantity : ArrayWUnits
                An ArrayWUnits object containing a value and unit (e.g. quantity.values and quantity.units)

            target_units : Units
                The target unit to check on the quantity
                
            Returns
            -------
            value : numpy.ndarray
                A numpy array corresponding to quantity.values, if the units match
        """
        if target_units == unitless and isinstance(quantity, (float, int)):
            # target is unitless and quantity is already unitless
            value = quantity
        elif quantity.units == target_units:
            value = quantity.values
        else:
            warnings.warn(
            f"Invalid units for {quantity}."
            f"Compatible with {target_units}, but got {quantity.units}."
            f"Trying automatic conversion to {target_units}.",
            UserWarning
        )
            try:
                # Attempt conversion to the expected units
                quantity_converted = quantity.to(target_units)
                if quantity_converted.units == target_units:
                    value = quantity_converted.values
                else: 
                    raise ValueError(
                        f"Automatic conversion attempted but failed."
                    )    
            except Exception:
                raise ValueError(
                    f"Automatic conversion attempted but failed."
                )
        
        return value