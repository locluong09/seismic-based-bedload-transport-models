from abc import ABC, abstractmethod

class SeismicBasedBedloadTransportModel(ABC):
    """
    Abstract base class for seismic-based bedload transport models.
    """

    @abstractmethod
    def forward_psd(self, *args, **kwargs):
        """
        Calculate the power spectral density (PSD) for sediment transport.

        Args: required args
            frequency : Frequency window (Hz).
            grain_size : Grain size diameter (m).
            flow_depth : Water depth (m).
            channel_width : Channel width (m).
            slope_angle : Channel slope (-).
            source_receiver_distance : Distance from the river thalweg to a seismic sensor (m).
            qb: bedload flux (m2/s)
        Returns:
            Power-spectral density (PSD) of seismic velocity signal.
        """
        pass

    @abstractmethod
    def inverse_bedload(self, *args, **kwargs):
        """
        Inverse calculation for bedload transport based on observed seismic PSD.

        Args: required args
            PSD_observe: observed PSD
            frequency : Frequency window (Hz).
            grain_size : Grain size diameter (m).
            flow_depth : Water depth (m).
            channel_width : Channel width (m).
            slope_angle : Channel slope (-).
            source_receiver_distance : Distance from the river thalweg to a seismic sensor (m).
            bedload flux qb is set to be 1 for inverting mode.
        Returns:
            Inverse back bedload flux from seismic PSD.
        """
        pass
