# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (    # import all necessary SCB classes
    constants,
    Units,
    ArrayWUnits,
    Frame,
    ArrayWFrame,
    EpochArray
)

from pathlib import Path    # import additional libraries with a break between SCB imports
import numpy as np

# define units and frames
km, kg, sec = Units.get_units(['km', 'kg', 'sec'])
J2000 = Frame('J2000')

# class definition
class TemplateClass():
    """ Template class for Scarabaeus developers.

        An example Python class to illustrate the style guide adopted by 
        Scarabaeus (SCB) developers and provide a starting point when creating 
        new modules.

        This is the second description paragraph. This template provides 
        examples for each docstring section, although not every section must always 
        be included. See the Notes section for an overview of the docstring 
        structure in order and recomendations for when and when not to include 
        them.

        Parameters
        ----------
        input_a : float
            The first input, not optional. We noted its type with the : float 
            after its name.
        input_b : str, optional
            A second optional input. Make sure to note what the default 
            value is using double back ticks. Defaults to ``None``.

        Raises
        ------
        CustomError
            Raised when we trigger the custom error. Only need to fill this 
            section out if we've created a custom error, otherwise this should 
            be omitted from the docstring.

        See Also
        --------
        scarabaeus.Units : provide a small reason to see the also.

        Notes
        -----
        The format of the class docstring follows the order:

        * name of the class
        * high-level description of the class in a sentence or two
        * if necessary, provide a more detailed description of the class in the
          second paragraph
        * Parameters section
        * Raises section - optional, usually not included. Only necessary 
          if the class contains custom errors
        * See Also section - optional, only included if you want to point to 
          any other SCB classes that are relevant
        * Notes section - optional, only included if you want to provide 
          additional information that would make the general description overly 
          long or complicated.
        * References section - optional, only included if the class 
          uses concepts from a source that needs to be cited
        * Examples - provide an example or two of using the class if it's small. 
          For large/complex classes, omit this section but make sure that you 
          provide examples in function docstrings.

        References
        ----------
        .. [1] Numpydoc,
          https://numpydoc.readthedocs.io/en/latest/example.html
            
        Examples
        --------
        Written in doctest format:
        
        >>> import scarabaeus as scb
        >>> example = scb.TemplateClass(1.0)
        >>> print(example)
        SCB Template Class with input a: 1.0 and input b: RECEIVED NONE FOR b
    """
    # region Constructor
    def __init__(self, input_a: float, input_b: str = None):
        # NOTE: we don't put a docstring on the init, otherwise it will override 
        #       the class docstring.
        ## save inputs
        # here we are performing input validation by passing them to their 
        # setters before actually assigning the property (no _)
        self._validate_a(input_a)
        self._validate_and_condition_b(input_b)

        ## save additional parameters if necessary
        # since we're setting these, we can set their private attributes directly
        # with the _
        self._public_prop = 2.0     # this property is readable by users
        self._priv_att    = False   # this attribute is private

    # region Properties
    @property
    def input_a(self) -> float:
        """ Explain what the property is. """
        return self._input_a
    
    def _validate_a(self, a) -> None:
        """ Internal method to validate input_a. """
        # must be of type float
        if not isinstance(a, float):
            raise TypeError(f'Received invalid type {type(a)} for input_a. '
                            'Must be of float.')
        
        # not doing anymore checking/conditioning -> set property
        self._input_a = a

    @property
    def input_b(self) -> str | None:
        """ Explain what the property is.
            
            For properties that need more than one line, make sure 
            to leave a vertical space between the short and long description. 
            Also note that we use the -> pointer to hint what type the property 
            is. This is important for documentation as well as for inline 
            autocompletion.
        """
        return self._input_b
    
    def _validate_and_condition_b(self, b) -> None:
        """ Internal method to validate and condition input_b. 
            In addition to checking the type, this will condition 
            the input depending on the received type.
        """
        match b:
            case str():
                # received string input -> more input handling
                if b < 0.0:
                    # negative -> must be positive
                    raise ValueError('Received a negative value for input_b.')
                
                if b == 5.2:
                    # special custom error
                    err_str = ('To illustrate a custom error, cannot accept '
                               'an input_b value of 5.2.')
                    raise type("CustomError", (Exception,), {})(err_str)

                # valid input -> assign property
                self._input_b = b
            case None:
                # received None -> assuming default
                self._input_b = 'RECEIVED NONE FOR b'
            case _:
                # didn't receive string or None -> bad type
                raise TypeError(f'Received invalid type {type(b)} for input_b. '
                                'Must be str or None.')

    @property
    def public_prop(self) -> float:
        """ Some additional public property of the class. 
            
            Even if it doesn't have a setter, still need to provide 
            a description of it if we want it visible to the user.
        """
        return self.public_prop
    
    # NOTE: do not assign a @property decorator to private attributes, for example 
    #       self._priv_att
    
    # endregion Properties

    # region Operators
    def __repr__(self):
        # most other operators don't need to be overridden. __repr__ 
        # almost always should be though so that printing the class can 
        # give us more info. We can see in the Examples section of the 
        # class docstring that printing TemplateClass will return info 
        # about its input variables, so let's do that here
        return f'SCB Template Class with input a: {self._input_a} and input b: None'

    # endregion Operators

    # region Methods
    def template_method(self, input_a: ArrayWUnits) -> float:
        """ Multiplies an :class:`~scarabaeus.units.ArrayWUnits` and returns its 
            values multiplied by the internal property :attr:`public_prop`.
        
            This is an example method for the example class TemplateClass.
            It takes 

            Parameters
            ----------
            input_a : :class:`~scarabaeus.units.ArrayWUnits`
                The ArrayWUnits to multiply by.

            Returns
            -------
            multiplied: float
                The given multiplied result.

            Notes
            -----
            Notes go here if you have them.

            Examples
            --------
            Here is an example:

            >>> import scarabaeus as scb
            >>> kg = scb.Units.get_units('kg')
            >>> a = scb.ArrayWUnits(10, kg)
            >>> print(a)
            10 kg
            >>> template = scb.TemplateClass(5.0)
            >>> print(template.public_prop)
            2.0
            >>> ans = template.template_method(a)
            >>> print(ans)
            20.0
        """
        # input handling
        if not isinstance(input_a, ArrayWUnits):
            raise TypeError(f'Received invalid type {type(input_a)} for input_a. '
                            'Must be ArrayWUnits.')
        
        # multiply value by property and return
        return input_a.values * self._public_prop