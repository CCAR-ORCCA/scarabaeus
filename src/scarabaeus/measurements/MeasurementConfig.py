# SPDX-FileCopyrightText: 2026 Orbital Research Cluster for Celestial Applications (ORCCA) Lab, University of Colorado at Boulder
# SPDX-License-Identifier: ISC
from dataclasses import dataclass, field
from typing import Any, Optional


# class definition
@dataclass
class MeasurementConfig:
    """
    Settings shared by every stream in a scenario.

    Anything here can also be overridden per stream. Extra keys can be carried
    in ``extra`` so this can grow without changing the signature.

    Attributes
    ----------
    media_folder : str, optional
        Folder holding the tropospheric and ionospheric correction files.
    tropo_seasonal_file_path : str, optional
        Seasonal troposphere file.
    sc_transponder_delay_RU : float
        Spacecraft transponder delay in range units (RU), applied to the real
        radiometric reads (DopplerReal, SequentialRangingReal). Defaults to
        ``0.0``, which takes the delay each tracking block carries, or no
        delay when it has none.

        A plain float on purpose, not an ArrayWUnits: a range unit is not a
        fixed unit of time. One RU lasts ``1 / (range_unit *
        reference_frequency)`` seconds, which depends on the band and on the
        exciter scalars of each pass, so it is only known once a pass is read
        and is not something :class:`~scarabaeus.Units` can convert. The
        reader converts it per pass with
        :meth:`~scarabaeus.Measurement.transponder_delay_seconds`. RU is also
        how missions quote the delay.
    frame : Frame, optional
        Frame the observations are expressed in. ``None`` leaves each model on
        its own default, which is J2000. Set it and
        :meth:`~scarabaeus.MeasurementSpec.build` passes it to every stream's read.
    sigma : dict, optional
        Default measurement sigma per type name or per dataset name, used for
        any stream that does not carry its own.

        A value is needed for every stream, including real data. The sigma
        refit during outlier rejection runs on the residuals, which come from
        the dataset, which cannot be built without a sigma to weight them -- so
        the refit revises an initial value rather than standing in for one. A
        stream left without one reports what is missing (see
        :meth:`~scarabaeus.Measurement._require_sigma`).
    bias : dict, optional
        Measurement bias per type name or dataset name. Defaults to 0.
    extra : dict
        Anything further, offered to every model constructor.

        Offered, not passed: :meth:`~scarabaeus.MeasurementSpec.build` filters each model's
        keyword arguments against its signature, so a key that only one model
        accepts reaches only that model instead of raising TypeError on the
        others. The cost is that a key no model accepts -- a typo, or one meant
        for a different scenario -- would simply do nothing, so ``build`` warns
        about any key it could not give to at least one model.

    See Also
    --------
    scarabaeus.MeasurementRegistry : The measurement types a stream can name.
    scarabaeus.MeasurementStream : One stream, which can override these
        settings.
    """

    media_folder: Optional[str] = None
    tropo_seasonal_file_path: Optional[str] = None
    sc_transponder_delay_RU: float = 0.0
    frame: Any = None
    sigma: dict = field(default_factory=dict)
    bias: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    # region Methods
    def get_sigma(self, type_name: str, dataset_name: str = None):
        """Return the sigma of a stream.

        Parameters
        ----------
        type_name : str
            Measurement type of the stream, a key of the registry.
        dataset_name : str, optional
            Dataset name of the stream, looked up before ``type_name``. Defaults
            to ``None``.

        Returns
        -------
        object or None
            The sigma given for ``dataset_name``, else the one given for
            ``type_name``, else ``None``.
        """
        if dataset_name is not None and dataset_name in self.sigma:
            return self.sigma[dataset_name]
        return self.sigma.get(type_name)

    def get_bias(self, type_name: str, dataset_name: str = None):
        """Return the bias of a stream.

        Parameters
        ----------
        type_name : str
            Measurement type of the stream, a key of the registry.
        dataset_name : str, optional
            Dataset name of the stream, looked up before ``type_name``. Defaults
            to ``None``.

        Returns
        -------
        object
            The bias given for ``dataset_name``, else the one given for
            ``type_name``, else ``0.0``.
        """
        if dataset_name is not None and dataset_name in self.bias:
            return self.bias[dataset_name]
        return self.bias.get(type_name, 0.0)

    # endregion Methods
