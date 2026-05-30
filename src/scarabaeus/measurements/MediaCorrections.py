# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from scarabaeus import (
    constants,
    Measurement,
    Units,
    Frame,
    ArrayWUnits,
    SpiceManager,
    Instrument,
    ArrayWFrame,
)
import scarabaeus as scb
import scarabaeus.utils.NumpyWrapper as np
import copy as copy
import os
import re
from typing import Optional, Union, Dict, List
from pathlib import Path
from scipy.integrate import quad

PathLike = Union[str, Path]

# Generate common units and frames
spc_pos_units, spc_time_units, spc_angle_units = Units.get_units(["km", "sec", "rad"])
J2000, ITRF93, ECLIPJ2000, IAUEARTH = Frame.generate_common_frames()


# --------------------#
#  Class Definition  #
# --------------------#
class MediaCorrections(Measurement):
    """ Computes path delays due to tropospheric and ionospheric media corrections.

    This class is a subclass of :class:`~scarabaeus.Measurement` and provides
    correction models for signal propagation through Earth's atmosphere.
    It is consumed by :class:`~scarabaeus.DopplerReal` and
    :class:`~scarabaeus.SequentialRangingReal`.

    **Tropospheric delay (Chao mapping)**

    The total one-way excess tropospheric path length is the sum of seasonal
    and non-seasonal zenith delays mapped to slant via the Chao functions
    (Estefan 1995; Moyer eq. 10-8 to 10-10):

    .. math::

        \\Delta\\rho_{tropo} = R_{dry}(\\varepsilon)\\,\\Delta\\tau_{dry}
                             + R_{wet}(\\varepsilon)\\,\\Delta\\tau_{wet}

    where the dry and wet mapping functions are

    .. math::

        R_{dry} = \\frac{1}{\\sin\\varepsilon
                  + \\dfrac{A_{dry}}{\\tan\\varepsilon + B_{dry}}},
        \\qquad
        R_{wet} = \\frac{1}{\\sin\\varepsilon
                  + \\dfrac{A_{wet}}{\\tan\\varepsilon + B_{wet}}}

    with constants :math:`A_{dry}=0.00143`, :math:`B_{dry}=0.0445`,
    :math:`A_{wet}=0.00035`, :math:`B_{wet}=0.017`.  The zenith delays are
    represented by Fourier (seasonal) or normalized polynomial (non-seasonal)
    expansions loaded from JSON files at construction time.

    **Ionospheric delay**

    The dispersive ionospheric group delay scales as :math:`f^{-2}`:

    .. math::

        \\Delta\\rho_{iono} \\propto \\frac{1}{f^2}\\int N_e\\,ds

    where :math:`f` is the carrier frequency and :math:`N_e` is the electron
    column density.

    **Sign conventions**

    * :class:`~scarabaeus.DopplerReal`: ionosphere enters with a **negative**
      sign (Moyer eq. 10-28, 10-29).
    * :class:`~scarabaeus.SequentialRangingReal`: ionosphere enters with a
      **positive** sign (Moyer eq. 10-27).

    Parameters
    ----------
    name : str
        The name of the media correction model object.
    instrument : Instrument
        The observation instrument or ground station to which the corrections apply.
    tropo_seasonal_file_path : PathLike, optional
        Path to the JSON file containing seasonal tropospheric correction data.
        Defaults to ``None``.
    tropo_non_seasonal_file_path : PathLike, optional
        Path to the JSON file containing non-seasonal tropospheric correction data.
        Defaults to ``None``.
    iono_file_path : PathLike, optional
        Path to the JSON file containing ionospheric correction data.
        Defaults to ``None``.

    See Also
    --------
    scarabaeus.DopplerReal : Real Doppler model that consumes MediaCorrections.
    scarabaeus.SequentialRangingReal : Real ranging model that consumes MediaCorrections.

    References
    ----------
    .. [1] https://descanso.jpl.nasa.gov/monograph/series2/Descanso2_all.pdf
    .. [2] https://ntrs.nasa.gov/api/citations/19950013586/downloads/19950013586.pdf
    .. [3] https://pds-geosciences.wustl.edu/insight/urn-nasa-pds-insight_documents/document_rise/trk-2-23-revc-l5_ion_tro.pdf
    """

    # Inclusive DSN antenna-number ranges for each complex, See [2, 3]
    _DSN_RANGES = {
        "C10": range(11, 31),  # 11–30 inclusive
        "C40": range(31, 51),  # 31–50 inclusive
        "C60": range(51, 71),  # 51–70 inclusive
    }

    # -------------------#
    # region Constructor #
    # -------------------#
    def __init__(
        self,
        name: str,
        instrument: Instrument,
        tropo_seasonal_file_path: Optional[PathLike] = None,
        tropo_non_seasonal_file_path: Optional[PathLike] = None,
        iono_file_path: Optional[PathLike] = None,
    ):
        super().__init__(name, instrument)

        # Initialize attributes
        self._name = name
        self._instrument = instrument

        # Automatically determine DSN site from antenna number
        self.dsn_site = self._match_dsn_site()

        # Path for seasonal tropospheric coefficients
        self._tropo_seasonal_file_path: Optional[Path] = None
        if tropo_seasonal_file_path is not None:
            self.tropo_seasonal_file_path = tropo_seasonal_file_path

        # Path for non-seasonal tropospheric coefficients
        self._tropo_non_seasonal_file_path: Optional[Path] = None
        if tropo_non_seasonal_file_path is not None:
            self.tropo_non_seasonal_file_path = tropo_non_seasonal_file_path

        # Path for ionospheric coefficients
        self._iono_file_path: Optional[Path] = None
        if iono_file_path is not None:
            self.iono_file_path = iono_file_path

    @property
    def tropo_seasonal_file_path(self) -> Optional[Path]:
        """Absolute path to the selected troposphere JSON, if set."""
        return self._tropo_seasonal_file_path

    @property
    def tropo_non_seasonal_file_path(self) -> Optional[Path]:
        """Absolute path to the selected troposphere JSON, if set."""
        return self._tropo_non_seasonal_file_path

    @property
    def iono_file_path(self) -> Optional[Path]:
        """Absolute path to the selected ionosphere JSON, if set."""
        return self._iono_file_path

    @tropo_seasonal_file_path.setter
    def tropo_seasonal_file_path(self, value: PathLike) -> None:
        p = Path(value).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Seasonal troposphere file not found: {p}")
        if p.suffix.lower() != ".json":
            raise ValueError(
                f"Expected a .json seasonal troposphere file, got: {p.name}"
            )
        self._tropo_seasonal_file_path = p.resolve()

    @tropo_non_seasonal_file_path.setter
    def tropo_non_seasonal_file_path(self, value: PathLike) -> None:
        p = Path(value).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Non-Seasonal troposphere file not found: {p}")
        if p.suffix.lower() != ".json":
            raise ValueError(
                f"Expected a .json non-seasonal troposphere file, got: {p.name}"
            )
        self._tropo_non_seasonal_file_path = p.resolve()

    @iono_file_path.setter
    def iono_file_path(self, value: PathLike) -> None:
        p = Path(value).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Ionospheric file not found: {p}")
        if p.suffix.lower() != ".json":
            raise ValueError(f"Expected a .json ionospheric file, got: {p.name}")
        self._iono_file_path = p.resolve()

    # ---------------#
    # region Methods #
    # ---------------#

    def _extract_antenna_number(self) -> int:
        """
        Extract the DSN antenna number associated with the current instrument.

        The method first attempts to use the ``spice_id`` attribute of the
        instrument if it exists and is an integer. In this case, the returned
        number is the last two digits of the DSN identifier (e.g., ``399024``
        → antenna number ``24``). If ``spice_id`` is not available, the method
        falls back to parsing the trailing digits from the instrument name.

        Returns
        -------
        antenna_number : int
            The numeric identifier of the DSN antenna (e.g., 14, 24, 35).

        Raises
        ------
        ValueError
            If neither a valid ``spice_id`` nor trailing digits in the
            instrument name can be found.

        Notes
        -----
        - For instruments with a SPICE ID (e.g., ``399024``), the antenna
        number is computed modulo 100.
        - For instruments identified by name (e.g., ``"DSS14"``), the antenna
        number is parsed directly from the trailing digits.

        Examples
        --------
        .. code-block:: python

            # Example with spice_id
            gs = scb.GroundStation("DSS-24")
            gs.spice_id = 399024
            antenna_number = gs._extract_antenna_number()   # returns 24

            # Example with name parsing
            gs = scb.GroundStation("DSS14")
            antenna_number = gs._extract_antenna_number()   # returns 14
        """
        # Prefer a numeric spice_id if your Instrument provides it
        spice_id: Optional[int] = getattr(self._instrument, "spice_id", None)
        if isinstance(spice_id, int):
            return (
                spice_id % 100
            )  # return the spice_id of the GSS witouth the 399 as prefix

        # Fallback: parse trailing digits from instrument name (e.g., 'DSS14' -> 14)
        m = re.search(r"(\d+)$", getattr(self._instrument, "name", ""))
        if not m:
            raise ValueError(
                f"Cannot determine antenna number from instrument name '{self._instrument.name}'. "
                "Expected trailing digits (e.g., '...14')."
            )
        return int(m.group(1))

    def _match_dsn_site(self) -> str:
        """
        Map the DSN antenna number to its corresponding complex site code.

        The antenna number is first extracted via :meth:`_extract_antenna_number`.
        It is then mapped into one of the defined DSN complexes: ``C10``,
        ``C40``, or ``C60``.

        Returns
        -------
        site : str
            DSN site label corresponding to the antenna number. One of
            ``"C10"``, ``"C40"``, or ``"C60"``.

        Raises
        ------
        ValueError
            If the antenna number does not fall within any of the configured
            DSN ranges.

        Notes
        -----
        The DSN antenna ranges are defined internally by the class attribute
        ``_DSN_RANGES``:

        - ``C10`` → antennas 11–30
        - ``C40`` → antennas 31–50
        - ``C60`` → antennas 51–70

        Examples
        --------
        .. code-block:: python

            media_corr = scb.MediaCorrections(
                name="GS1 Doppler Media Corrections",
                instrument=scb.GroundStation("DSS-24"),
            )
            site = media_corr._match_dsn_site()  # returns "C10"
        """
        antenna_num = self._extract_antenna_number()

        for site, rng in self._DSN_RANGES.items():
            if antenna_num in rng:
                return site

        # If no mapping found, raise a clear error
        raise ValueError(
            f"No DSN site mapping for antenna number {antenna_num}. "
            f"Expected one of ranges: { {k: (r.start, r.stop - 1) for k, r in self._DSN_RANGES.items()} }"
        )

    @staticmethod
    def find_tro_metadata(
        folder: str, query_year: int, query_doy: int
    ) -> Optional[Dict]:
        """
        Locate troposphere metadata files covering a given year and day-of-year.

        This scans the specified folder for files named according to the pattern
        ``YYYY_DDD_YYYY_DDD_tro.json``. Each file encodes the start and end
        year/DOY interval it covers. The method returns the metadata dictionary
        for the file whose interval contains the provided query year/day.
        Intervals that cross year boundaries (e.g. ``2017_335_2018_001_tro.json``)
        are handled correctly.

        Parameters
        ----------
        folder : str
            Path to the folder containing the troposphere metadata files.

        query_year : int
            Year of interest in four-digit format (e.g., 2017).

        query_doy : int
            Day-of-year (1–366) within `query_year`.

        Returns
        -------
        metadata : dict or None
            Dictionary describing the matching file with fields:

            - ``"file"`` : filename of the tro file
            - ``"start_year"``, ``"start_doy"`` : start of coverage interval
            - ``"end_year"``, ``"end_doy"`` : end of coverage interval

            Returns ``None`` if no file covers the query date.

        Raises
        ------
        None

        Notes
        -----
        If multiple files cover the same query date, the one with the tightest
        interval span is returned.

        Examples
        --------
        .. code-block:: python

            meta = MediaCorrections.find_tro_metadata(
                folder="data/tropo",
                query_year=2017,
                query_doy=346
            )
            print(meta["file"])
            # -> "2017_335_2018_001_tro.json"
        """
        pat = re.compile(
            r"(?P<y1>\d{4})_(?P<d1>\d{3})_(?P<y2>\d{4})_(?P<d2>\d{3})_tro\.json$"
        )
        metadata: List[Dict] = []
        for fname in os.listdir(folder):
            m = pat.match(fname)
            if not m:
                continue
            y1, d1, y2, d2 = map(int, (m["y1"], m["d1"], m["y2"], m["d2"]))
            # skip obviously bad ranges
            if (y2, d2) < (y1, d1):
                continue
            metadata.append(
                {
                    "file": fname,
                    "start_year": y1,
                    "start_doy": d1,
                    "end_year": y2,
                    "end_doy": d2,
                }
            )
        q = (query_year, query_doy)
        # If multiple files cover the same day, prefer the tightest window.
        best = None
        best_span = None
        for meta in metadata:
            start = (meta["start_year"], meta["start_doy"])
            end = (meta["end_year"], meta["end_doy"])
            if start <= q <= end:
                span = (end[0] - start[0], end[1] - start[1])  # coarse tie-breaker
                if best is None or span < best_span:
                    best, best_span = meta, span
        return best

    @staticmethod
    def find_ion_metadata(
        folder: str, query_year: int, query_doy: int
    ) -> Optional[Dict]:
        """
        Locate ionosphere metadata files covering a given year and day-of-year.

        This scans the specified folder for files named with any prefix followed by
        ``YYYY_DDD_YYYY_DDD_ion.json`` (e.g., ``orex_beno_2018_213_2018_244_ion.json``).
        The method returns the metadata dictionary for the file whose interval
        contains the provided query year/day. Intervals crossing year boundaries
        are handled correctly.

        Parameters
        ----------
        folder : str
            Path to the folder containing ionosphere metadata files.
        query_year : int
            Year of interest in four-digit format (e.g., 2018).
        query_doy : int
            Day-of-year (1–366) within `query_year`.

        Returns
        -------
        metadata : dict or None
            Dictionary describing the matching file with fields:
            - "file" : filename
            - "start_year", "start_doy" : start of coverage interval
            - "end_year",   "end_doy"   : end of coverage interval
            Returns None if no file covers the query date.

        Notes
        -----
        If multiple files cover the same query date, the one with the tightest
        interval span is returned.
        """
        # Accept any leading prefix, but require ...YYYY_DDD_YYYY_DDD_ion.json at the end
        pat = re.compile(
            r".*?(?P<y1>\d{4})_(?P<d1>\d{3})_(?P<y2>\d{4})_(?P<d2>\d{3})_ion\.json$"
        )
        metadata: List[Dict] = []
        for fname in os.listdir(folder):
            m = pat.match(fname)
            if not m:
                continue
            y1, d1, y2, d2 = map(int, (m["y1"], m["d1"], m["y2"], m["d2"]))
            # Skip obviously bad ranges
            if (y2, d2) < (y1, d1):
                continue
            metadata.append(
                {
                    "file": fname,
                    "start_year": y1,
                    "start_doy": d1,
                    "end_year": y2,
                    "end_doy": d2,
                }
            )
        q = (query_year, query_doy)
        # Prefer the tightest window if multiple files include the query date
        best = None
        best_span = None
        for meta in metadata:
            start = (meta["start_year"], meta["start_doy"])
            end = (meta["end_year"], meta["end_doy"])
            if start <= q <= end:
                span = (end[0] - start[0], end[1] - start[1])  # coarse tie-breaker
                if best is None or span < best_span:
                    best, best_span = meta, span
        return best

    def mediafun_trig(media_data, query_time_et) -> float:
        """
        Evaluate media correction function using a trigonometric series expansion.
        Applicable for seasonal, non-seasonal, and ionospheric corrections.

        Parameters
        ----------
        media_data : dict
            Dictionary containing the trigonometric coefficients (`C1`–`C10`)
            and the interval bounds `start_et` and `end_et`. Coefficients
            are expected to be convertible to float.
        query_time_et : float
            Ephemeris Time (ET) at which the correction is to be evaluated.

        Returns
        -------
        delta_sum : float
            Media correction value delay at the requested time, expressed in meters.

        Raises
        ------
        KeyError
            If one or more required coefficients (`C1`–`C10`) or ET bounds
            (`start_et`, `end_et`) are missing from `media_data`.

        Notes
        -----
        The function evaluates a truncated Fourier series of the form:

        .. math::

            \\Delta(t) = C_1
                    + C_2 \\cos(X) + C_3 \\sin(X)
                    + C_4 \\cos(2X) + C_5 \\sin(2X)
                    + C_6 \\cos(3X) + C_7 \\sin(3X)
                    + C_8 \\cos(4X) + C_9 \\sin(4X)

        where :math:`X = 2 \\pi (t - A)/P`, with :math:`A` being the
        `start_et` and :math:`P` the fundamental period `C1`.

        Examples
        --------
        .. code-block:: python

            # Example evaluation
            et = scb.SpiceManager.utc2et("2010-01-15 12:00:00 UTC")
            correction = scb.MediaCorrections.mediafun_trig(media_data, et)
        """
        C = np.array(
            [media_data[f"C{i}"] for i in range(1, 11)], dtype=np.float64
        )  # Create list with coefficients 1-10 (convert all to floats)
        A = media_data["start_et"]
        B = media_data["end_et"]
        P = C[0]  # Period
        X = 2 * np.pi * ((float(query_time_et) - A) / P)
        delta_sum = (
            C[1]
            + C[2] * np.cos(X)
            + C[3] * np.sin(X)
            + C[4] * np.cos(2 * X)
            + C[5] * np.sin(2 * X)
            + C[6] * np.cos(3 * X)
            + C[7] * np.sin(3 * X)
            + C[8] * np.cos(4 * X)
            + C[9] * np.sin(4 * X)
        )
        return delta_sum

    def mediafun_nrmpow(media_data, query_time_et) -> float:
        """
        Evaluate media correction function using a normalized power series expansion.
        Applicable for seasonal, non-seasonal, and ionospheric corrections.

        Parameters
        ----------
        media_data : dict
            Dictionary containing the polynomial coefficients (`C1`–`C10`)
            and the interval bounds `start_et` and `end_et`. Coefficients
            are expected to be convertible to float.
        query_time_et : float
            Ephemeris Time (ET) at which the correction is to be evaluated.

        Returns
        -------
        delta_sum : float
            Media correction value delay at the requested time, expressed in meters.

        Raises
        ------
        KeyError
            If one or more required coefficients (`C1`–`C10`) or ET bounds
            (`start_et`, `end_et`) are missing from `media_data`.

        Notes
        -----
        The function evaluates a normalized power series of the form:

        .. math::

            X = 2 \\frac{t - A}{B - A} - 1

            \\Delta(t) = C_0 X^0 + C_1 X^1 + C_2 X^2 + \\ldots + C_9 X^9

        where :math:`A` and :math:`B` are the start and end ephemeris times
        of the interval, respectively.

        Examples
        --------
        .. code-block:: python

            # Example evaluation
            et = scb.SpiceManager.utc2et("2010-01-15 12:00:00 UTC")
            correction = scb.MediaCorrections.mediafun_nrmpow(media_data, et)
        """
        C = np.array(
            [media_data[f"C{i}"] for i in range(1, 11)], dtype=np.float64
        )  # Create list with coefficients 1-10 (convert all to floats)
        A = media_data["start_et"]
        B = media_data["end_et"]
        X = 2 * ((float(query_time_et) - A) / (B - A)) - 1
        delta_sum = (
            C[0] * X**0
            + C[1] * X**1
            + C[2] * X**2
            + C[3] * X**3
            + C[4] * X**4
            + C[5] * X**5
            + C[6] * X**6
            + C[7] * X**7
            + C[8] * X**8
            + C[9] * X**9
        )
        return delta_sum

    def _mediafun_constant(media_data, query_time) -> float:
        """
        Evaluate media correction function using a constant.
        Applicable for seasonal, non-seasonal, and ionospheric corrections.

        Parameters
        ----------
        media_data : dict
            Dictionary containing the polynomial coefficients (`C1`–`C10`).
            In this case, only `C1` is used. The coefficients are not cast
            to floats, as the data may include NaN values.

        query_time_et : float
            Ephemeris Time (ET) at which the correction is to be evaluated.

        Returns
        -------
        delta_sum : float or Any
            Constant media correction value taken directly from `C1`.

        Raises
        ------
        KeyError
            If `C1` is missing from `media_data`.

        Notes
        -----
        This function returns the first coefficient (`C1`) without further
        transformation, representing a constant correction across the
        interval.

        See Also
        --------
        mediafun_trig : Seasonal correction using trigonometric expansion.
        mediafun_nrmpow : Non-seasonal correction using normalized power series.

        Examples
        --------
        .. code-block:: python

            # Example evaluation
            correction = scb.MediaCorrections._mediafun_constant(media_data)
        """
        C = [
            media_data[f"C{i}"] for i in range(1, 11)
        ]  # Create list with coefficients 1-10 (No conversion to floats because data includes NaN)
        delta_sum = C[0]
        return delta_sum

    @staticmethod
    def _compute_delta_static(aux_data, query_time_et) -> float:
        """
        Compute the static media delay contribution for a given dataset.

        Depending on the ``type`` field in the input dictionary, this method dispatches
        the evaluation to the appropriate function (trigonometric, normalized power series,
        or constant). If ``aux_data`` is not provided, it returns 0.0.

        Parameters
        ----------
        aux_data : dict or None
            Dictionary containing coefficients and metadata for the media correction.
            Must include a ``"type"`` field with one of the supported types
            (``"TRIG"``, ``"NRMPOW"``, or ``"CONST"``).

        query_time_et : float
            Ephemeris Time (ET, seconds past J2000) at which to evaluate the media
            correction.

        Returns
        -------
        delta : float
            The computed static media delay contribution [m].


        Examples
        --------
        .. code-block:: python

            aux = {"type": "TRIG", "C1": 1.0, "C2": 0.5, "start_et": 100, "end_et": 200}
            et = 150.0
            delay = media_correction._compute_delta_static(aux, et)
            print(delay)
        """
        # Map type string to function
        MEDIA_FUN_MAP = {
            "TRIG": MediaCorrections.mediafun_trig,
            "NRMPOW": MediaCorrections.mediafun_nrmpow,
            "CONST": MediaCorrections._mediafun_constant,
        }
        if not aux_data:
            return 0
        f = MEDIA_FUN_MAP.get(aux_data.get("type"))
        return f(aux_data, query_time_et)

    def chao_mapping(E: float) -> tuple[float, float]:
        """
        Compute the tropospheric mapping functions for dry and wet delays using the Chao model.

        Parameters
        ----------
        E : float
            Elevation angle of the observation [rad].

        Returns
        -------
        R_dry : float
            Dry mapping function value at the given elevation angle.

        R_wet : float
            Wet mapping function value at the given elevation angle.

        Notes
        -----
        The mapping functions are based on:

        - Estefan (1995), equations (7)–(8a,b).
        - Moyer (2000), equations (10–8 to 10–10).

        The dry and wet components are computed as:

        .. math::

            R_{dry} = \\frac{1}{\\sin(E) + \\frac{A_{dry}}{\\tan(E) + B_{dry}}}

            R_{wet} = \\frac{1}{\\sin(E) + \\frac{A_{wet}}{\\tan(E) + B_{wet}}}

        with constants :math:`A_{dry} = 0.00143`, :math:`B_{dry} = 0.0445`,
        :math:`A_{wet} = 0.00035`, and :math:`B_{wet} = 0.017`.

        References
        ----------
        .. [1] Estefan, J. A. (1995). Tropospheric Delay Effects on GPS Measurements.
        .. [2] Moyer, T. D. (2000). Formulation for Observed and Computed Values
            of Deep Space Network Data Types for Navigation. JPL 00-7.

        Examples
        --------
        .. code-block:: python

            # Example: compute mapping functions at 30 degrees elevation
            import numpy as np
            E = np.deg2rad(30.0)
            R_dry, R_wet = scb.MediaCorrections.chao_mapping(E)
        """

        # Define constants for summation
        A_dry, B_dry, A_wet, B_wet = (
            0.00143,
            0.0445,
            0.00035,
            0.017,
        )  # Estefan 8a & 8b or Moyer 10-9 & 10-10
        # Calculate wet and dry mapping given by Estefan eq. 7 or Moyer 10-8
        R_dry = 1 / (np.sin(E) + A_dry / (np.tan(E) + B_dry))
        R_wet = 1 / (np.sin(E) + A_wet / (np.tan(E) + B_wet))
        return R_dry, R_wet

    def _csp_time_to_et(day: str, time: str) -> float:
        """
        Convert a compact CSP-style date and time string into SPICE Ephemeris Time (ET).

        Parameters
        ----------
        day : str
            Date string in the format ``YY/MM/DD`` where the year may be two digits.
            Years in the range 70–99 are mapped to 1970–1999, while all other
            two-digit years are mapped to 2000+.
        time : str
            Time string in the format ``HH:MM`` or ``HH:MM:SS``. If seconds are
            omitted, ``:00`` is appended automatically.

        Returns
        -------
        et : float
            Ephemeris Time (ET) in seconds past J2000.

        Raises
        ------
        ValueError
            If the input strings cannot be parsed into a valid date/time.
        SpiceyError
            If the generated UTC string cannot be converted by SPICE.

        Notes
        -----
        The function builds a SPICE-compatible UTC string of the form:

        .. code-block::

            YYYY MON DD HH:MM:SS UTC

        where ``MON`` is the three-letter uppercase month abbreviation.
        The resulting UTC string is passed to ``SpiceManager.str2et``.

        Examples
        --------
        .. code-block:: python

            # Example: convert 1 Feb 2019 00:00 UTC to ET
            et = scb.MediaCorrections._csp_time_to_et("19/02/01", "00:00")

            # Example: convert 31 Mar 2019 23:59:00 UTC to ET
            et = scb.MediaCorrections._csp_time_to_et("19/03/31", "23:59")
        """
        # Expand the date into a SPICE-friendly UTC string
        # First normalize time string
        if len(time.split(":")) == 2:
            time = time + ":00"  # append seconds if missing
        # Parse the day string
        yy, mm, dd = day.split("/")
        yy = int(yy)
        mm = int(mm)
        dd = int(dd)
        # Map two-digit year to full year
        if 70 <= yy <= 99:
            yyyy = 1900 + yy
        else:
            yyyy = 2000 + yy
        # Build UTC string in a format SPICE accepts (YYYY MON DD HH:MM:SS UTC)
        MONTH_ABBR = [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]
        utc_str = f"{yyyy} {MONTH_ABBR[mm-1]} {dd:02d} {time} UTC"
        return SpiceManager.str2et(utc_str)

    def get_media_data_for_et(
        dict_input: dict, site_query: str, wetdry_query: Optional[str], query_et: float
    ) -> Optional[dict]:
        """
        Extract media-correction coefficients for a given DSN site and epoch,
        with optional wet/dry filtering. Works for both troposphere tables
        (with 'wet_or_dry' and possibly 'type') and ionosphere tables (which
        may omit those fields).

        Parameters
        ----------
        dict_input : dict
            Table-like dictionary with row-aligned lists:
            required: 'dsn', 'start_day', 'start_time', 'end_day', 'end_time'
            optional: 'wet_or_dry', 'type', any 'C#' coefficient fields, etc.
        site_query : str
            DSN site code to filter (e.g., "C40").
        wetdry_query : str or None
            If provided and 'wet_or_dry' exists, filter by this mode ("WET"/"DRY").
            If None or the column is missing, only the site filter is applied.
        query_et : float
            Ephemeris Time (ET) at which the values are requested.

        Returns
        -------
        dict or None
            A single-interval dictionary slice whose [start_et, end_et] contains
            query_et. All row-wise fields are reduced to the selected row; non
            row-wise/scalar fields are carried as-is. Returns None if no match.
        """
        # ---- Basic row count from 'dsn' (required) ----
        if "dsn" not in dict_input:
            return None
        num_rows = len(dict_input["dsn"])
        if num_rows == 0:
            return None

        have_wetdry = ("wet_or_dry" in dict_input) and (wetdry_query is not None)

        # ---- Row selection indices ----
        keep_idx = []
        for i in range(num_rows):
            if dict_input["dsn"][i] != site_query:
                continue
            if have_wetdry:
                if (
                    str(dict_input["wet_or_dry"][i]).upper()
                    != str(wetdry_query).upper()
                ):
                    continue
            keep_idx.append(i)

        if not keep_idx:
            return None

        # ---- Build a filtered view (slice row-wise lists, keep scalars as-is) ----
        filtered = {}
        for key, vals in dict_input.items():
            if isinstance(vals, list) and len(vals) == num_rows:
                filtered[key] = [vals[i] for i in keep_idx]
            else:
                filtered[key] = vals  # non row-wise or mismatched length → carry as-is

        # ---- Convert UTC bounds to ET (row-wise over kept indices) ----
        try:
            start_ets = [
                MediaCorrections._csp_time_to_et(d0, t0)
                for d0, t0 in zip(filtered["start_day"], filtered["start_time"])
            ]
            end_ets = [
                MediaCorrections._csp_time_to_et(d1, t1)
                for d1, t1 in zip(filtered["end_day"], filtered["end_time"])
            ]
        except KeyError:
            # Required time columns missing
            return None

        filtered["start_et"] = start_ets
        filtered["end_et"] = end_ets

        # ---- Pick the interval containing query_et ----
        for i, (s, e) in enumerate(zip(start_ets, end_ets)):
            if s <= query_et <= e:
                # Reduce row-wise lists to the chosen row; keep scalars as-is
                single = {
                    k: (v[i] if isinstance(v, list) and len(v) == len(start_ets) else v)
                    for k, v in filtered.items()
                }
                return single

        return None

    def computed_rtlt_delays(
        self,
        query_time_et: float,
        target_spice_id: str,
        frequency: float | None = None,
        gs_delta_awf: ArrayWFrame | None = None,
    ) -> tuple[float, float]:
        """
        Compute a radio signal path delay due to Earth's media at a given epoch.

        This method evaluates the **tropospheric** contribution (seasonal + non-seasonal,
        dry + wet) mapped by the Chao functions at the station elevation angle.
        If an ionospheric dataset is available (``self.iono_file_path``), the ionospheric
        term may also be included (currently not implemented).

        Parameters
        ----------
        query_time_et : float
            SPICE Ephemeris Time (seconds past J2000) at which to evaluate the delay.

        target_spice_id : str
            SPICE name or NAIF ID of the tracked target (e.g., spacecraft) used to
            compute the elevation angle at the DSN station.

        frequency : float, optional
            Uplink or downlink frequency in Hz. Required to compute ionospheric
            group delay (∝ 1/frequency²). If None, iono contribution is skipped.

        gs_delta_awf: ArrayWFrame | None
            If not None, this is the delta vector for the ground station in the SSB space-time relativistic frame of reference.


        Returns
        -------
        p_total : float
            Total media path delay in meters (troposphere + ionosphere if available).
            The troposphere term sums seasonal and non-seasonal dry/wet components and
            applies the Chao mapping functions at the current elevation angle.

        Raises
        ------
        KeyError
            If required keys are missing from the loaded JSON files (e.g., ``"aux_tropo"``).
        FileNotFoundError
            If one of the provided JSON file paths does not exist.
        ValueError
            If the JSON content is malformed or contains invalid time ranges.

        Notes
        -----
        - The method expects JSON files at the paths
        ``self.tropo_seasonal_file_path`` and ``self.tropo_non_seasonal_file_path``.
        Each should expose an ``"aux_tropo"`` table compatible with
        :meth:`MediaCorrections.get_media_data_for_et`.
        - The elevation angle is computed via :meth:`SpiceManager.get_elevation_angle`
        using the DSN station name from ``self.instrument.name`` and the provided
        target identifier.
        - Mapping to slant delay is performed using :meth:`MediaCorrections.chao_mapping`
        (dry and wet components).
        - Ionospheric delay handling is scaffolded (``flag_iono``) but not yet implemented.

        See Also
        --------
        MediaCorrections.get_media_data_for_et
            Interval selection and coefficient extraction for a given site/mode/time.
        MediaCorrections.chao_mapping
            Dry/wet mapping functions as a function of elevation angle.
        MediaCorrections.mediafun_trig
            Trigonometric (seasonal) delay evaluator.
        MediaCorrections.mediafun_nrmpow
            Normalized power-series (non-seasonal) delay evaluator.

        Examples
        --------
        .. code-block:: python

            et = scb.SpiceManager.utc2et("2018-12-03 12:00:00 UTC")
            p_media = media_correction.computed_rtlt_delays(
                query_time_et=et,
                target_spice_id="ORX"
            )
        """
        # TODO: incorporate gs_delta_awf (ground-station position delta) at the appropriate
        # points in the formulation. Without this, the ground-station location offset cannot
        # be iteratively estimated within the filter. This is currently not critical and is
        # left unimplemented due to limited time and uncertainty about the correct insertion
        # points in the pipeline.

        # Get auxiliary data
        DSS_name = self.instrument.name
        query_site = self.dsn_site

        tropo_non_seasonal_path = getattr(self, "tropo_non_seasonal_file_path", None)
        tropo_seasonal_path = getattr(self, "tropo_seasonal_file_path", None)
        iono_path = getattr(self, "iono_file_path", None)

        flag_tropo = False
        flag_iono = False
        if tropo_non_seasonal_path is not None and tropo_seasonal_path is not None:
            tropo_seasonal_dict = scb.Utils.load_json(tropo_seasonal_path)
            tropo_dict = scb.Utils.load_json(tropo_non_seasonal_path)
            flag_tropo = True
        if iono_path is not None:
            iono_dict = scb.Utils.load_json(iono_path)
            flag_iono = True

        # Compute the elevation angle E at query_time_et
        elevation_angle_awu = SpiceManager.get_elevation_angle(
            target=target_spice_id,
            et=query_time_et,  # SPICE ET (seconds past J2000) or a UTC string
            station=DSS_name,  # e.g., "DSS-14", "DSS-43", "DSS-63"
            abcorr="LT",  # LT+S
        )
        # Get the elevation angle as float, making sure it's in RAD
        E = ArrayWUnits.get_value_in_target_units(
            elevation_angle_awu, spc_angle_units
        )  # In [rad]

        # Compute tropo contribution
        p_total_tropo = 0
        if flag_tropo:
            # static components (in m) - Extract coefficients and aux data
            aux_data_dry_seasonal = MediaCorrections.get_media_data_for_et(
                tropo_seasonal_dict["aux_tropo"], query_site, "DRY", query_time_et
            )
            aux_data_wet_seasonal = MediaCorrections.get_media_data_for_et(
                tropo_seasonal_dict["aux_tropo"], query_site, "WET", query_time_et
            )
            aux_data_dry_non_seasonal = MediaCorrections.get_media_data_for_et(
                tropo_dict["aux_tropo"], query_site, "DRY", query_time_et
            )
            aux_data_wet_non_seasonal = MediaCorrections.get_media_data_for_et(
                tropo_dict["aux_tropo"], query_site, "WET", query_time_et
            )
            # static components (in m) - Compute delay with appropiate function (TRIG, NRMPOW, CONST)
            delta_dry_seasonal = (
                MediaCorrections._compute_delta_static(
                    aux_data_dry_seasonal, query_time_et
                )
                if aux_data_dry_seasonal
                else 0
            )
            delta_wet_seasonal = (
                MediaCorrections._compute_delta_static(
                    aux_data_wet_seasonal, query_time_et
                )
                if aux_data_wet_seasonal
                else 0
            )
            delta_dry_non_seasonal = (
                MediaCorrections._compute_delta_static(
                    aux_data_dry_non_seasonal, query_time_et
                )
                if aux_data_dry_non_seasonal
                else 0
            )
            delta_wet_non_seasonal = (
                MediaCorrections._compute_delta_static(
                    aux_data_wet_non_seasonal, query_time_et
                )
                if aux_data_wet_non_seasonal
                else 0
            )
            # mapping component (in m) - Compute the dry and wet mappings based on elevation angle
            dry_mapping, wet_mapping = MediaCorrections.chao_mapping(E)
            # Add up coefficient sums for seasonal and non-seasonal
            p_dry = (
                delta_dry_seasonal + delta_dry_non_seasonal
            ) * dry_mapping  # Moyer 10-2
            p_wet = (
                delta_wet_seasonal + delta_wet_non_seasonal
            ) * wet_mapping  # Moyer 10-3
            p_total_tropo = p_dry + p_wet

        # Compute iono contribution
        p_total_iono = 0
        if flag_iono:
            # Extract coefficients and aux data
            aux_data_iono = MediaCorrections.get_media_data_for_et(
                iono_dict["aux_iono"], query_site, None, query_time_et
            )
            # Compute the delay in m
            delta_iono_p1 = (
                MediaCorrections.mediafun_nrmpow(aux_data_iono, query_time_et)
                if aux_data_iono
                else 0.0
            )
            delta_iono_p2 = ((2295 * 1e6) / frequency) ** 2  # Moyer 10-11
            p_total_iono = delta_iono_p1 * delta_iono_p2

        return p_total_tropo, p_total_iono

    @staticmethod
    def _compute_SC_lt(
        sun_to_sc_t2: np.ndarray,
        sun_to_gs_tx: np.ndarray,
        frequency: np.ndarray,
        radii_SC_threshold: float = 100,
    ) -> np.ndarray:
        """
        Compute Solar Corona delays

        Parameters
        ----------
        sun_to_sc_t2 : np.ndarray
            State vector (pos,vel) from the sun to the spacecraft at t2

        sun_to_gs_tx : np.ndarray
            State vector (pos,vel) from the sun to the ground station at tx (e.g. for way == 'dwn' sun_to_gs_t3, for way == 'up' sun_to_gs_t1 )

        frequency: np.ndarray
            Signal frequency, expressed in Hz

        radii_SC_threshold: float
            Minimum Sun radii threshold to perform SC computation. Default to 50 Sun's Radii.

        Returns
        -------
        dt_SC_lt : float
            Solar Corona lightime delay due to group delays

        Notes
        -----
        - To compute the output, the frequency is expected in GHz, as per in Eq. (1) in [1]
        - To group delay output in Eq. (1) in [1] is expressed in microseconds

        References
        ----------
        .. [1] Christian M. Ho (2010). 106, Rev. B Solar Corona and Solar Wind Effects. DSN Teleccomunications Link Design Handbook
        .. [2] Moyer, T. D. (2000). Formulation for Observed and Computed Values
            of Deep Space Network Data Types for Navigation. JPL 00-7.

        """

        # Get the Solar radius R0 in [m]
        R0 = (
            ArrayWUnits.get_value_in_target_units(
                constants.SUN.mean_radius, spc_pos_units
            )
            * 1e3
        )  # [m]

        # Compute L, from [1] Fig.1, [2] Eq. (10-58)
        Lvec = sun_to_sc_t2[:3] - sun_to_gs_tx[:3]
        L = Lvec / np.linalg.norm(Lvec)  # [km]
        # Calculate the point of closest approach of the signal to the Sun, From [2] (10-59)
        pdown_vec = sun_to_gs_tx[:3] - np.dot(sun_to_gs_tx[:3], L) * L  # [km]
        a_min = np.linalg.norm(pdown_vec) * 1e3  # [m]
        # Only continue with the computation if a_min/R0  is below a pre-defined threshold
        dt_SC_lt = 0
        if a_min / R0 < radii_SC_threshold:
            # Calculate the maximum distances from Sun to spacecraft and Sun to GS, From [1], Fig.1
            a_max_sc = np.linalg.norm(sun_to_sc_t2[:3]) * 1e3  # [m]
            a_max_gs = np.linalg.norm(sun_to_gs_tx[:3]) * 1e3  # [m]
            L1 = np.sqrt(a_max_gs**2 - a_min**2)  # [m]
            L2 = np.sqrt(a_max_sc**2 - a_min**2)  # [m]

            # Define a model for Solar wind electron density as function of radial distance a for line integration in [1] Eq. 1
            def _solar_wind_electron_density_func(l, a_min, L_i):
                # From [1], Fig.1
                a = np.sqrt((L_i - l) ** 2 + a_min**2)
                # From [1], eq (2)
                Ne = 2.21e14 * (a / R0) ** (-6) + 1.55e12 * (a / R0) ** (-2.3)
                return Ne

            # Integrate to compute STEC (Slant Total Electron Content)
            STEC_L1, _ = quad(
                _solar_wind_electron_density_func, 0, L1, args=(a_min, L1)
            )  # STEC on L1, from GS to a_min, see [1] Fig. 1
            STEC_L2, _ = quad(
                _solar_wind_electron_density_func, 0, L2, args=(a_min, L2)
            )  # STEC on L2, from SC to a_min, see [1] Fig. 1
            STEC_tot = (
                STEC_L1 + STEC_L2
            )  # Total STEC on one-way path between GS-to-SC or SC-to_GS (according to sun_to_gs_tx input)
            # Convert STEC to group delay, [1] Eq. (1)
            dt_SC_lt = 1.3446e-19 / (frequency * 1e-9) ** 2 * STEC_tot * 1e-6  # [s]

        return dt_SC_lt
